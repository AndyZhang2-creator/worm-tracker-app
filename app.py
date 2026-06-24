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

import base64
import csv
import io
import os
import secrets
import tempfile
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from speed_utils import compute_displacement, compute_track_speed

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
# Roboflow workflow backend (temporary base model)
# --------------------------------------------------------------------------- #

RF_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
# Direct-model id (workspace-public, not secret). When set we call the model
# endpoint directly — this avoids the broken serverless *workflow* (its inner
# `model` step references an undeclared `model_id` input) and works both hosted
# and on a local inference server. Set ROBOFLOW_MODEL_ID="" to use the workflow.
RF_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "c-elegan-detection-5haae/1")
RF_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "andy-zhang-ud8qm")
RF_WORKFLOW = os.environ.get("ROBOFLOW_WORKFLOW_ID", "c-elegan-detection-v1-logic")
RF_IMAGE_INPUT = os.environ.get("ROBOFLOW_IMAGE_INPUT", "image")
# Endpoint selection. This model emits two endpoints ("End-point1"=0,
# "End-Point-2"=1) rather than head/tail, and the two ends of a worm move
# differently. By default we track BOTH per frame and report whichever moved the
# most (greatest displacement from its start) — "the farthest tail". Override to
# pin a specific endpoint by index (WORM_ENDPOINT_INDEX) or by class name
# (ROBOFLOW_TAIL_KEYPOINT) if you'd rather always track the same one.
FORCE_ENDPOINT_NAME = os.environ.get("ROBOFLOW_TAIL_KEYPOINT", "").strip().lower()
_force_idx_raw = os.environ.get("WORM_ENDPOINT_INDEX", "").strip()
FORCE_ENDPOINT_INDEX = int(_force_idx_raw) if _force_idx_raw.lstrip("-").isdigit() else None


def _rf_mode() -> str:
    """'model' for direct-model inference (default), 'workflow' otherwise."""
    return "model" if RF_MODEL_ID else "workflow"


def _rf_base() -> str:
    """Resolve the inference base URL: explicit override, else a mode default."""
    explicit = os.environ.get("ROBOFLOW_API_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    # Direct model works on the hosted serverless endpoint; the workflow path
    # historically targets a local inference server.
    return "https://serverless.roboflow.com" if _rf_mode() == "model" else "http://localhost:9001"


def active_backend() -> str:
    """
    Which detector to use. Explicit WORM_BACKEND wins; otherwise auto: a
    configured Roboflow key implies the (temporary) workflow backend, else the
    permanent local YOLO weights.
    """
    explicit = os.environ.get("WORM_BACKEND", "").strip().lower()
    if explicit in ("ultralytics", "roboflow"):
        return explicit
    return "roboflow" if RF_API_KEY else "ultralytics"


def _roboflow_infer(frame):
    """POST one frame to Roboflow (direct model or workflow); return the JSON."""
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to JPEG-encode frame for Roboflow.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    base = _rf_base()

    if _rf_mode() == "model":
        url = f"{base}/{RF_MODEL_ID}?api_key={RF_API_KEY}"
        kwargs = dict(
            content=b64,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    else:
        url = f"{base}/infer/workflows/{RF_WORKSPACE}/{RF_WORKFLOW}"
        kwargs = dict(
            json={
                "api_key": RF_API_KEY,
                "inputs": {RF_IMAGE_INPUT: {"type": "base64", "value": b64}},
                "use_cache": True,
            }
        )

    last_exc = None
    for _ in range(2):  # one light retry for transient hiccups
        try:
            resp = httpx.post(url, timeout=120, **kwargs)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Roboflow returned HTTP {resp.status_code}: {resp.text[:300]}"
                )
            return resp.json()
        except httpx.HTTPError as exc:
            last_exc = exc
    safe_url = url.split("?")[0]  # never echo the api_key in errors
    raise RuntimeError(f"Could not reach Roboflow at {safe_url}: {last_exc}")


def _find_predictions(obj):
    """
    Depth-first search of an arbitrary workflow response for the list of
    detection dicts (each with x/y and ideally a `keypoints` list). Roboflow
    nests this under named outputs, so we don't hard-code the output key.
    """
    if isinstance(obj, list):
        if obj and isinstance(obj[0], dict) and (
            "keypoints" in obj[0] or ("x" in obj[0] and "y" in obj[0])
        ):
            return obj
        for item in obj:
            found = _find_predictions(item)
            if found:
                return found
    elif isinstance(obj, dict):
        p = obj.get("predictions")
        if isinstance(p, list) and p and isinstance(p[0], dict):
            return p
        for value in obj.values():
            found = _find_predictions(value)
            if found:
                return found
    return None


def _parse_roboflow_keypoints(data):
    """
    Return ALL keypoints of the most confident detection from a Roboflow
    response as a list of (x, y, conf, name) in the model's keypoint order, or
    None if no worm/keypoint was found. Endpoint selection happens later, after
    we've seen how much each one moved across the whole clip.
    """
    preds = _find_predictions(data)
    if not preds:
        return None
    best = max(preds, key=lambda p: p.get("confidence", 0.0) or 0.0)
    kps = best.get("keypoints") or []
    if not kps:
        return None

    out = []
    for kp in kps:
        name = str(kp.get("class_name", kp.get("class", ""))).strip()
        conf = kp.get("confidence")
        out.append((float(kp["x"]), float(kp["y"]),
                    float(conf if conf is not None else 1.0), name))
    return out


def _roboflow_unconfigured_detail() -> str:
    return (
        "Roboflow backend selected but ROBOFLOW_API_KEY is not set. Export "
        "ROBOFLOW_API_KEY (and optionally ROBOFLOW_API_URL, default "
        f"{_rf_base()}) to use the temporary detector."
    )


def make_detector():
    """
    Build a `frame -> list[(x, y, conf, name)] | None` callable for the active
    backend (all keypoints of the best detection). Raises HTTPException(503) if
    that backend isn't ready, so callers get the same clean 503 either way.
    """
    backend = active_backend()
    if backend == "roboflow":
        if not RF_API_KEY:
            raise HTTPException(status_code=503, detail=_roboflow_unconfigured_detail())

        def detect(frame):
            return _parse_roboflow_keypoints(_roboflow_infer(frame))

        return detect

    model = get_model()  # may raise HTTPException(503)

    def detect(frame):
        return _extract_keypoints(model(frame, verbose=False)[0])

    return detect


# --------------------------------------------------------------------------- #
# Core per-video analysis
# --------------------------------------------------------------------------- #

_YOLO_KP_NAMES = ["head", "tail"]  # index 0 = head, 1 = tail (build-spec contract)


def _extract_keypoints(result):
    """
    From a single-frame YOLO result, return ALL keypoints of the highest-
    confidence detection as a list of (x, y, conf, name), or None if no worm was
    detected. Index 0 is named "head", index 1 "tail", per the model contract.
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

    xy = kpts.xy[best_i].cpu().numpy()  # shape (K, 2)
    if xy.shape[0] == 0:
        return None
    kconf = kpts.conf[best_i].cpu().numpy() if kpts.conf is not None else None

    out = []
    for i in range(xy.shape[0]):
        conf = float(kconf[i]) if kconf is not None else 1.0
        name = _YOLO_KP_NAMES[i] if i < len(_YOLO_KP_NAMES) else f"kp{i}"
        out.append((float(xy[i][0]), float(xy[i][1]), conf, name))
    return out


def analyze_video_file(path: str, sample_fps: float, pcutoff: float, px_per_mm):
    """
    Run inference over one video and return the response dict (spec §5 shape).

    Raises ValueError for un-openable / non-video files so callers can decide
    whether to abort (single) or record a per-video error (batch).
    """
    detect = make_detector()  # backend-agnostic; may raise HTTPException(503)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open file as a video.")

    # Per-video native fps — never assume a global value (spec §4).
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0
    step = max(1, round(native_fps / sample_fps))
    effective_fps = native_fps / step  # <- THIS goes into compute_track_speed

    # Sequential read + skip; only run inference every step-th frame. Keep ALL
    # keypoints per frame — we choose which endpoint to report after seeing how
    # much each moved across the whole clip.
    frames_kps = []  # each: list[(x,y,conf,name)] or None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            frames_kps.append(detect(frame))
        frame_idx += 1
    cap.release()

    frames_analyzed = len(frames_kps)
    if frames_analyzed == 0:
        raise ValueError("No frames could be decoded from this file.")

    n_kp = max((len(f) for f in frames_kps if f), default=0)
    if n_kp == 0:
        raise ValueError("No worm/keypoints were ever detected in this video.")

    # Build one (xs, ys, confs) track per keypoint index. A frame where the worm
    # (or that keypoint) is missing becomes a NaN gap (conf 0), never a fake 0.
    endpoint_names = ["" for _ in range(n_kp)]
    tracks = [([], [], []) for _ in range(n_kp)]
    for f in frames_kps:
        for k in range(n_kp):
            kp = f[k] if (f and k < len(f)) else None
            kxs, kys, kconfs = tracks[k]
            if kp is None:
                kxs.append(np.nan); kys.append(np.nan); kconfs.append(0.0)
            else:
                kxs.append(kp[0]); kys.append(kp[1]); kconfs.append(kp[2])
                if not endpoint_names[k] and len(kp) > 3 and kp[3]:
                    endpoint_names[k] = kp[3]

    # Per-endpoint displacement, then choose which endpoint to report: by
    # default the one that moved the most ("the farthest tail").
    per_endpoint = []
    for k in range(n_kp):
        kxs, kys, kconfs = tracks[k]
        kd = compute_displacement(kxs, kys, kconfs, pcutoff=pcutoff, px_per_mm=px_per_mm)
        per_endpoint.append({
            "index": k,
            "name": endpoint_names[k] or f"kp{k}",
            "farthest_displacement_px": _round_or_none(kd["farthest_displacement_px"]),
            "net_displacement_px": _round_or_none(kd["net_displacement_px"]),
        })
    chosen = _choose_endpoint(per_endpoint)
    xs, ys, confs = tracks[chosen]

    metrics = compute_track_speed(
        xs, ys, confs, fps=effective_fps, pcutoff=pcutoff, px_per_mm=px_per_mm
    )
    disp = compute_displacement(xs, ys, confs, pcutoff=pcutoff, px_per_mm=px_per_mm)

    # time_series_s is one timestamp per speed value (i.e. per interval).
    # speed[i] covers the interval between analyzed frame i and i+1; we label it
    # by the time of the later frame.
    n_speeds = len(metrics["speed_series_px_s"])
    time_series_s = [round((i + 1) / effective_fps, 3) for i in range(n_speeds)]

    # When the farthest-reach happened (seconds from start), for the chart.
    far_frame = disp.get("farthest_displacement_frame")
    far_time = round(far_frame / effective_fps, 3) if far_frame is not None else None

    response = {
        "video": os.path.basename(path),
        "backend": active_backend(),
        "native_fps": round(float(native_fps), 3),
        "effective_fps": round(float(effective_fps), 3),
        "frames_analyzed": frames_analyzed,
        "frames_tracked_pct": round(metrics["frames_tracked_pct"], 3),
        "avg_speed_px_s": _round_or_none(metrics["avg_speed_px_s"]),
        "max_speed_px_s": _round_or_none(metrics["max_speed_px_s"]),
        "total_distance_px": round(metrics["total_distance_px"], 3),
        # New: how far the tail got from its start, and start->endpoint.
        "farthest_displacement_px": _round_or_none(disp["farthest_displacement_px"]),
        "net_displacement_px": _round_or_none(disp["net_displacement_px"]),
        "farthest_displacement_time_s": far_time,
        # Which of the worm's endpoints we report: the one that moved the most.
        "tracked_endpoint_index": chosen,
        "tracked_endpoint_name": endpoint_names[chosen] or f"kp{chosen}",
        "endpoints": per_endpoint,
        "speed_series_px_s": metrics["speed_series_px_s"],
        "time_series_s": time_series_s,
        "avg_speed_mm_s": _round_or_none(metrics.get("avg_speed_mm_s")),
        "max_speed_mm_s": _round_or_none(metrics.get("max_speed_mm_s")),
        "total_distance_mm": _round_or_none(metrics.get("total_distance_mm")),
        "farthest_displacement_mm": _round_or_none(disp.get("farthest_displacement_mm")),
        "net_displacement_mm": _round_or_none(disp.get("net_displacement_mm")),
    }
    return response


def _choose_endpoint(per_endpoint: list[dict]) -> int:
    """
    Pick which endpoint index to report. Default: the one that moved the
    farthest from its start (greatest displacement) — "the farthest tail".
    Overridable: a forced class name (ROBOFLOW_TAIL_KEYPOINT) or index
    (WORM_ENDPOINT_INDEX) pins a specific endpoint instead.
    """
    if FORCE_ENDPOINT_NAME:
        for ep in per_endpoint:
            if ep["name"].strip().lower() == FORCE_ENDPOINT_NAME:
                return ep["index"]
    if FORCE_ENDPOINT_INDEX is not None:
        for ep in per_endpoint:
            if ep["index"] == FORCE_ENDPOINT_INDEX:
                return ep["index"]
    # Auto: greatest farthest-displacement (None sorts as -1, so a tracked
    # endpoint always beats an untracked one).
    return max(
        per_endpoint,
        key=lambda ep: ep["farthest_displacement_px"] if ep["farthest_displacement_px"] is not None else -1.0,
    )["index"]


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
    "farthest_displacement_px",
    "net_displacement_px",
]
CSV_MM_COLUMNS = [
    "avg_speed_mm_s",
    "max_speed_mm_s",
    "total_distance_mm",
    "farthest_displacement_mm",
    "net_displacement_mm",
]


def _batch_summary(results: list[dict], used_mm: bool) -> dict:
    """
    Aggregate the headline numbers the user asked for across a batch:
      - average speed the worms are moving (mean of per-video averages)
      - how far the farthest tail moved, and which video that was.
    Failed videos and videos with no trackable speed are excluded from means.
    """
    speed_key = "avg_speed_mm_s" if used_mm else "avg_speed_px_s"
    far_key = "farthest_displacement_mm" if used_mm else "farthest_displacement_px"
    ok = [r for r in results if "error" not in r]

    speeds = [r[speed_key] for r in ok if r.get(speed_key) is not None]
    avg_speed = round(sum(speeds) / len(speeds), 3) if speeds else None

    far_video, far_value = None, None
    for r in ok:
        v = r.get(far_key)
        if v is not None and (far_value is None or v > far_value):
            far_value, far_video = v, r.get("video")

    return {
        "videos_total": len(results),
        "videos_ok": len(ok),
        "videos_failed": len(results) - len(ok),
        "units_speed": "mm/s" if used_mm else "px/s",
        "units_distance": "mm" if used_mm else "px",
        "avg_speed": avg_speed,
        "farthest_displacement": round(far_value, 3) if far_value is not None else None,
        "farthest_displacement_video": far_video,
    }


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
    """Report whether the active backend is ready, without ever 503-ing."""
    backend = active_backend()

    if backend == "roboflow":
        if not RF_API_KEY:
            return {"ready": False, "backend": backend, "detail": _roboflow_unconfigured_detail()}
        base = _rf_base()
        target = f"{RF_MODEL_ID}" if _rf_mode() == "model" else f"workflow {RF_WORKSPACE}/{RF_WORKFLOW}"
        # Light reachability check against the inference server root.
        try:
            httpx.get(base, timeout=5)
            return {"ready": True, "backend": backend, "detail": f"Roboflow {target} @ {base}"}
        except httpx.HTTPError as exc:
            return {
                "ready": False,
                "backend": backend,
                "detail": (
                    f"Roboflow not reachable at {base} ({exc}). Start a local "
                    f"inference server (`inference server start`) or set "
                    f"ROBOFLOW_API_URL to a hosted endpoint."
                ),
            }

    path = _model_path()
    if not path.exists():
        return {"ready": False, "backend": backend, "detail": _missing_model_detail()}
    try:
        get_model()
        return {"ready": True, "backend": backend}
    except HTTPException as exc:
        return {"ready": False, "backend": backend, "detail": exc.detail}


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
    return {
        "results": results,
        "csv_token": token,
        "summary": _batch_summary(results, used_mm=bool(px_per_mm)),
    }


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
