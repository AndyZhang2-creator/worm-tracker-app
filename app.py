"""
app.py — FastAPI backend for the Worm Tail Tracker.

Endpoints:
  POST /api/analyze         single video
  POST /api/analyze-batch   multiple videos (+ CSV token)
  GET  /api/download/{token} download the batch CSV
  GET  /api/model-status    is the model loaded?

The trained YOLO-pose weights (model/best.pt) will very likely NOT exist the
first time this runs — labeling/training happens separately. So:
  * `ultralytics` is imported INSIDE the loader, never at module level.
  * Loading is lazy, on first request.
  * A missing/broken model yields a clean 503 with an actionable message.

Model contract (fixed — see build spec §2):
  * single class `worm`
  * 2 keypoints per detection, index 0 = head, index 1 = tail
  * results[0].boxes.conf       -> per-detection box confidence
  * results[0].keypoints.xy[i]  -> (2,2) pixel (x,y) for [head, tail]
  * results[0].keypoints.conf[i]-> (2,) per-keypoint confidence; None => 1.0
"""

import csv
import io
import os
import secrets
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from speed_utils import compute_track_speed

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "best.pt"

app = FastAPI(title="Worm Tail Tracker")

# In-memory store of generated CSVs, keyed by a random token. Lives for the
# process lifetime — this is a single-user local tool, no need for anything
# fancier than a dict.
_CSV_STORE: dict[str, str] = {}

# Lazily-populated model handle and the last load error (if any).
_MODEL = None
_MODEL_LOAD_ERROR: str | None = None


def _model_path() -> Path:
    """Resolve the weights path, honoring the WORM_MODEL_PATH override."""
    override = os.environ.get("WORM_MODEL_PATH")
    return Path(override) if override else DEFAULT_MODEL_PATH


def _missing_model_detail() -> str:
    return (
        f"No model found at {_model_path()}. Export your trained Ultralytics "
        f"YOLO-pose weights (best.pt) to that location, or set the "
        f"WORM_MODEL_PATH environment variable to point at them, then retry."
    )


def get_model():
    """
    Lazily load the YOLO model on first use.

    Returns the loaded model, or raises HTTPException(503) with an actionable
    message if the weights are missing or fail to load. `ultralytics` is
    imported here (not at module scope) so the server starts cleanly with no
    model present.
    """
    global _MODEL, _MODEL_LOAD_ERROR

    if _MODEL is not None:
        return _MODEL

    path = _model_path()
    if not path.exists():
        _MODEL_LOAD_ERROR = _missing_model_detail()
        raise HTTPException(status_code=503, detail=_MODEL_LOAD_ERROR)

    try:
        from ultralytics import YOLO  # imported lazily, by design (spec §6)

        _MODEL = YOLO(str(path))
        _MODEL_LOAD_ERROR = None
        return _MODEL
    except Exception as exc:  # noqa: BLE001 - surface any load failure as 503
        _MODEL_LOAD_ERROR = (
            f"Failed to load model at {path}: {exc}. Confirm the file is a "
            f"valid Ultralytics YOLO-pose weights export."
        )
        raise HTTPException(status_code=503, detail=_MODEL_LOAD_ERROR) from exc


# --------------------------------------------------------------------------- #
# Core per-video analysis
# --------------------------------------------------------------------------- #

def _extract_tail_keypoint(result):
    """
    From a single-frame YOLO result, return (x, y, conf) for the TAIL keypoint
    of the highest-confidence detection, or None if no worm was detected.

    Keypoint index 1 == tail (index 0 == head), per the fixed model contract.
    """
    kpts = getattr(result, "keypoints", None)
    if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
        return None

    # Choose the most confident detection if several worms appear.
    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
        best_i = int(np.argmax(boxes.conf.cpu().numpy()))
    else:
        best_i = 0

    xy = kpts.xy[best_i].cpu().numpy()  # shape (2, 2): [[hx,hy],[tx,ty]]
    if xy.shape[0] < 2:
        return None

    tail_x, tail_y = float(xy[1][0]), float(xy[1][1])

    # Per-keypoint confidence; if the model didn't emit any, treat as 1.0.
    if kpts.conf is not None:
        tail_conf = float(kpts.conf[best_i].cpu().numpy()[1])
    else:
        tail_conf = 1.0

    return tail_x, tail_y, tail_conf


def analyze_video_file(path: str, sample_fps: float, pcutoff: float, px_per_mm):
    """
    Run inference over one video and return the response dict (spec §5 shape).

    Raises ValueError for un-openable / non-video files so callers can decide
    whether to abort (single) or record a per-video error (batch).
    """
    model = get_model()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open file as a video.")

    # Per-video native fps — never assume a global value (spec §4).
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0
    step = max(1, round(native_fps / sample_fps))
    effective_fps = native_fps / step  # <- THIS goes into compute_track_speed

    xs, ys, confs = [], [], []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Sequential read + skip; only run inference every step-th frame.
        if frame_idx % step == 0:
            results = model(frame, verbose=False)
            r = results[0]
            kp = _extract_tail_keypoint(r)
            if kp is None:
                # Worm not detected this frame -> conf 0 so it falls below any
                # sensible pcutoff and becomes a NaN gap, never a fake point.
                xs.append(np.nan)
                ys.append(np.nan)
                confs.append(0.0)
            else:
                xs.append(kp[0])
                ys.append(kp[1])
                confs.append(kp[2])
        frame_idx += 1
    cap.release()

    frames_analyzed = len(xs)
    if frames_analyzed == 0:
        raise ValueError("No frames could be decoded from this file.")

    metrics = compute_track_speed(
        xs, ys, confs, fps=effective_fps, pcutoff=pcutoff, px_per_mm=px_per_mm
    )

    # time_series_s is one timestamp per speed value (i.e. per interval).
    # speed[i] covers the interval between analyzed frame i and i+1; we label it
    # by the time of the later frame.
    n_speeds = len(metrics["speed_series_px_s"])
    time_series_s = [round((i + 1) / effective_fps, 3) for i in range(n_speeds)]

    response = {
        "video": os.path.basename(path),
        "native_fps": round(float(native_fps), 3),
        "effective_fps": round(float(effective_fps), 3),
        "frames_analyzed": frames_analyzed,
        "frames_tracked_pct": round(metrics["frames_tracked_pct"], 3),
        "avg_speed_px_s": _round_or_none(metrics["avg_speed_px_s"]),
        "max_speed_px_s": _round_or_none(metrics["max_speed_px_s"]),
        "total_distance_px": round(metrics["total_distance_px"], 3),
        "speed_series_px_s": metrics["speed_series_px_s"],
        "time_series_s": time_series_s,
        "avg_speed_mm_s": _round_or_none(metrics.get("avg_speed_mm_s")),
        "max_speed_mm_s": _round_or_none(metrics.get("max_speed_mm_s")),
        "total_distance_mm": _round_or_none(metrics.get("total_distance_mm")),
    }
    return response


def _round_or_none(v):
    return None if v is None else round(float(v), 3)


def _save_upload_to_temp(upload: UploadFile) -> str:
    """Persist an UploadFile to a temp path so OpenCV can open it by path."""
    suffix = Path(upload.filename or "video").suffix or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return tmp_path


# --------------------------------------------------------------------------- #
# CSV building
# --------------------------------------------------------------------------- #

CSV_BASE_COLUMNS = [
    "video",
    "native_fps",
    "frames_analyzed",
    "frames_tracked_pct",
    "avg_speed_px_s",
    "max_speed_px_s",
    "total_distance_px",
]
CSV_MM_COLUMNS = ["avg_speed_mm_s", "max_speed_mm_s", "total_distance_mm"]


def _build_csv(results: list[dict], used_calibration: bool) -> str:
    columns = CSV_BASE_COLUMNS + (CSV_MM_COLUMNS if used_calibration else [])
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for r in results:
        if "error" in r:
            # Failed video: record name, leave the numeric cells blank so the
            # CSV stays a clean rectangle that lines up with the header.
            row = [r.get("video", "")] + ["" for _ in columns[1:]]
        else:
            row = [r.get(col, "") for col in columns]
        writer.writerow(row)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/api/model-status")
def model_status():
    """Report whether the model is (or can be) loaded, without 503-ing."""
    path = _model_path()
    if not path.exists():
        return {"ready": False, "detail": _missing_model_detail()}
    try:
        get_model()
        return {"ready": True}
    except HTTPException as exc:
        return {"ready": False, "detail": exc.detail}


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    sample_fps: float = Form(10.0),
    pcutoff: float = Form(0.5),
    px_per_mm: float | None = Form(None),
):
    tmp_path = _save_upload_to_temp(file)
    try:
        result = analyze_video_file(tmp_path, sample_fps, pcutoff, px_per_mm)
        # Preserve the original uploaded filename rather than the temp name.
        result["video"] = file.filename or result["video"]
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _safe_unlink(tmp_path)


@app.post("/api/analyze-batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    sample_fps: float = Form(10.0),
    pcutoff: float = Form(0.5),
    px_per_mm: float | None = Form(None),
):
    results: list[dict] = []
    for upload in files:
        tmp_path = _save_upload_to_temp(upload)
        try:
            res = analyze_video_file(tmp_path, sample_fps, pcutoff, px_per_mm)
            res["video"] = upload.filename or res["video"]
            results.append(res)
        except HTTPException:
            # Model not ready — this is fatal for the whole batch, not per-video.
            _safe_unlink(tmp_path)
            raise
        except Exception as exc:  # noqa: BLE001 - one bad video must not abort
            results.append({"video": upload.filename or "unknown", "error": str(exc)})
        finally:
            _safe_unlink(tmp_path)

    csv_text = _build_csv(results, used_calibration=bool(px_per_mm))
    token = secrets.token_hex(16)
    _CSV_STORE[token] = csv_text
    return {"results": results, "csv_token": token}


@app.get("/api/download/{token}")
def download_csv(token: str):
    csv_text = _CSV_STORE.get(token)
    if csv_text is None:
        raise HTTPException(status_code=404, detail="Unknown or expired CSV token.")
    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="worm_speed_summary.csv"'
        },
    )


def _safe_unlink(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


# Serve the single-page frontend. Mounted last so it doesn't shadow /api routes.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
