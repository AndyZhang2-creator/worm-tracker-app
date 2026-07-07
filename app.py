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
import json
import os
import queue
import secrets
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from speed_utils import compute_displacement, compute_track_speed

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_MODEL_PATH = BASE_DIR / "model" / "best.pt"


def _load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE pairs from .env without adding another dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env(BASE_DIR / ".env")

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
# Roboflow workflow backend
# --------------------------------------------------------------------------- #

def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


RF_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "").strip()
RF_WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "andy-zhang-ud8qm").strip()
RF_WORKFLOW = os.environ.get("ROBOFLOW_WORKFLOW_ID", "c-elegan-detection-v6-logic").strip()
RF_IMAGE_INPUT = os.environ.get("ROBOFLOW_IMAGE_INPUT", "image").strip()
RF_USE_CACHE = os.environ.get("ROBOFLOW_USE_CACHE", "true").strip().lower() not in ("0", "false", "no")
RF_CONFIDENCE = _float_env("ROBOFLOW_CONFIDENCE", 0.30)
RF_WORKFLOW_TIMEOUT_S = _float_env("ROBOFLOW_WORKFLOW_TIMEOUT_S", 15.0)
RF_MODEL_TIMEOUT_S = _float_env("ROBOFLOW_MODEL_TIMEOUT_S", 120.0)
DEFAULT_PCUTOFF = _float_env("WORM_PCUTOFF", RF_CONFIDENCE)
MAX_WORMS_PER_VIDEO = max(1, int(_float_env("WORM_MAX_TRACKS", 5)))
MAX_MATCH_DISTANCE_PX = _float_env("WORM_MAX_MATCH_DISTANCE_PX", 150.0)
# Light moving-average smoothing of the center trajectory before measuring
# speed. Tracking jitter only ever ADDS to summed path length (displacement is
# always >= 0), so unsmoothed speed reads a bit high; a small window damps that.
# Odd window in analyzed frames; 1 = off. Tune up if speed still overshoots.
SMOOTH_WINDOW = max(1, int(_float_env("WORM_SMOOTH_WINDOW", 3)))
# Overlap tolerance for de-duplicating the model's boxes: if two detections
# overlap by more than this fraction of the smaller box, keep only the most
# confident one. Adjustable per request; 0.9 = merge boxes overlapping >90%.
DEFAULT_OVERLAP_TOLERANCE = min(1.0, max(0.0, _float_env("WORM_OVERLAP_TOLERANCE", 0.9)))
TRACKING_FRAME_MAX_WIDTH = max(320, int(_float_env("WORM_TRACKING_FRAME_MAX_WIDTH", 900)))
TRACKING_FRAME_JPEG_QUALITY = min(95, max(50, int(_float_env("WORM_TRACKING_FRAME_JPEG_QUALITY", 85))))
LIVE_FRAME_MAX_WIDTH = max(320, int(_float_env("WORM_LIVE_FRAME_MAX_WIDTH", 720)))
LIVE_FRAME_JPEG_QUALITY = min(95, max(45, int(_float_env("WORM_LIVE_FRAME_JPEG_QUALITY", 75))))
LIVE_FRAME_EVERY_N = max(1, int(_float_env("WORM_LIVE_FRAME_EVERY_N", 1)))

# Optional escape hatch for the older direct-model backend. The default is now
# the serverless Roboflow workflow requested by the user.
RF_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "c-elegan-detection-5haae/6").strip()
RF_FALLBACK_MODEL_ID = os.environ.get("ROBOFLOW_FALLBACK_MODEL_ID", "c-elegan-detection-5haae/6").strip()
_RF_CLIENT = None

# Endpoint naming helpers kept for per-endpoint displacement diagnostics. The
# headline speed/displacement tracks the worm center, not either endpoint.
FORCE_ENDPOINT_NAME = os.environ.get("ROBOFLOW_TAIL_KEYPOINT", "").strip().lower()
_force_idx_raw = os.environ.get("WORM_ENDPOINT_INDEX", "").strip()
FORCE_ENDPOINT_INDEX = int(_force_idx_raw) if _force_idx_raw.lstrip("-").isdigit() else None


def _rf_mode() -> str:
    """'workflow' by default; 'model' only when ROBOFLOW_MODEL_ID is set."""
    return "model" if RF_MODEL_ID else "workflow"


def _rf_base() -> str:
    """Resolve the inference base URL: explicit override, else a mode default."""
    explicit = os.environ.get("ROBOFLOW_API_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    return "https://serverless.roboflow.com"


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


def _roboflow_client():
    """Return an inference-sdk client when the optional package is installed."""
    global _RF_CLIENT

    if _RF_CLIENT is not None:
        return _RF_CLIENT
    try:
        from inference_sdk import InferenceHTTPClient
    except ImportError:
        return None

    _RF_CLIENT = InferenceHTTPClient(api_url=_rf_base(), api_key=RF_API_KEY)
    return _RF_CLIENT


def _frame_to_temp_jpeg(frame) -> str:
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    ok = cv2.imwrite(path, frame)
    if not ok:
        _safe_unlink(path)
        raise ValueError("Failed to JPEG-encode frame for Roboflow.")
    return path


def _post_roboflow(url: str, timeout: float, attempts: int = 2, **kwargs):
    last_exc = None
    for _ in range(attempts):
        try:
            resp = httpx.post(url, timeout=timeout, **kwargs)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Roboflow returned HTTP {resp.status_code}: {resp.text[:300]}"
                )
            return resp.json()
        except httpx.HTTPError as exc:
            last_exc = exc
    raise RuntimeError(f"Could not reach Roboflow at {url}: {last_exc}")


def _roboflow_workflow_http(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to JPEG-encode frame for Roboflow.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    base = _rf_base()
    url = f"{base}/infer/workflows/{RF_WORKSPACE}/{RF_WORKFLOW}"
    return _post_roboflow(
        url,
        timeout=RF_WORKFLOW_TIMEOUT_S,
        attempts=1,
        json={
            "api_key": RF_API_KEY,
            "inputs": {RF_IMAGE_INPUT: {"type": "base64", "value": b64}},
            "use_cache": RF_USE_CACHE,
        },
    )


def _workflow_fallback_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "InnerWorkflowParameterBindingsUnknownInputError" in text
        or "unknown child workflow input names ['model_id']" in text
        or "ReadTimeout" in text
        or "Could not reach Roboflow" in text
    )


def _roboflow_direct_model_http(frame, model_id: str | None = None):
    model_id = model_id or RF_MODEL_ID or RF_FALLBACK_MODEL_ID
    if not model_id:
        raise RuntimeError("No Roboflow direct model id is configured.")

    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to JPEG-encode frame for Roboflow.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    url = f"{_rf_base()}/{model_id}"
    return _post_roboflow(
        url,
        timeout=RF_MODEL_TIMEOUT_S,
        attempts=2,
        params={"api_key": RF_API_KEY, "confidence": int(RF_CONFIDENCE * 100)},
        content=b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )


def _roboflow_infer(frame):
    """Run one frame through the configured Roboflow workflow/model."""
    if _rf_mode() == "model":
        return _roboflow_direct_model_http(frame)

    client = _roboflow_client()
    if client is None:
        try:
            return _roboflow_workflow_http(frame)
        except RuntimeError as exc:
            if _workflow_fallback_error(exc):
                return _roboflow_direct_model_http(frame, RF_FALLBACK_MODEL_ID)
            raise

    image_path = _frame_to_temp_jpeg(frame)
    try:
        try:
            return client.run_workflow(
                workspace_name=RF_WORKSPACE,
                workflow_id=RF_WORKFLOW,
                images={RF_IMAGE_INPUT: image_path},
                use_cache=RF_USE_CACHE,
            )
        except Exception as exc:  # noqa: BLE001 - SDK wraps server errors
            if _workflow_fallback_error(exc):
                return _roboflow_direct_model_http(frame, RF_FALLBACK_MODEL_ID)
            raise
    finally:
        _safe_unlink(image_path)


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


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_response_image_size(obj):
    """
    Find Roboflow response image dimensions without confusing detection box
    widths/heights for the whole image size.
    """
    if isinstance(obj, list):
        for item in obj:
            found = _find_response_image_size(item)
            if found:
                return found
    elif isinstance(obj, dict):
        image = obj.get("image")
        if isinstance(image, dict):
            w = _safe_float(image.get("width"), 0.0)
            h = _safe_float(image.get("height"), 0.0)
            if w > 0 and h > 0:
                return w, h
        for value in obj.values():
            found = _find_response_image_size(value)
            if found:
                return found
    return None


def _coordinate_scale_for_response(data, frame_shape):
    if frame_shape is None:
        return 1.0, 1.0
    response_size = _find_response_image_size(data)
    if not response_size:
        return 1.0, 1.0
    response_w, response_h = response_size
    frame_h, frame_w = frame_shape[:2]
    if response_w <= 0 or response_h <= 0 or frame_w <= 0 or frame_h <= 0:
        return 1.0, 1.0
    return float(frame_w) / response_w, float(frame_h) / response_h


def _scale_optional(value, scale):
    return None if value is None else float(value) * scale


def _candidate_center(candidate: dict):
    x = candidate.get("x")
    y = candidate.get("y")
    if x is not None and y is not None:
        return float(x), float(y)

    kps = candidate.get("keypoints") or []
    if not kps:
        return None
    xs = [kp[0] for kp in kps if not np.isnan(kp[0])]
    ys = [kp[1] for kp in kps if not np.isnan(kp[1])]
    if not xs or not ys:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def _center_with_conf(candidate: dict):
    """(x, y, detection_confidence) of the detection center, or None."""
    center = _candidate_center(candidate)
    if center is None:
        return None
    return (center[0], center[1], _safe_float(candidate.get("confidence"), 0.0))


def _candidate_box(candidate: dict):
    """(x1, y1, x2, y2) box for a candidate; falls back to its keypoint extent."""
    x, y = candidate.get("x"), candidate.get("y")
    w, h = candidate.get("width"), candidate.get("height")
    if x is not None and y is not None and w and h:
        return (float(x) - w / 2, float(y) - h / 2, float(x) + w / 2, float(y) + h / 2)
    kps = candidate.get("keypoints") or []
    xs = [k[0] for k in kps if not np.isnan(k[0])]
    ys = [k[1] for k in kps if not np.isnan(k[1])]
    if not xs or not ys:
        return None
    pad = 4.0  # give single/colinear keypoints a small footprint
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _box_overlap_ratio(a, b) -> float:
    """
    Overlap as a fraction of the SMALLER box (intersection / min-area). This
    reads as "these two boxes overlap by X%" and catches one box nested inside
    another — the usual signature of the model detecting the same worm twice.
    """
    if a is None or b is None:
        return 0.0
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _dedupe_overlapping(candidates: list[dict], tolerance: float):
    """
    Greedy NMS: walking highest-confidence first, drop any detection that
    overlaps an already-kept one by more than `tolerance` (0..1). So when boxes
    overlap a lot (default >90%), only the most-confident worm survives.
    tolerance >= 1 (or <= 0) disables the filter.
    """
    if not candidates or tolerance is None or tolerance <= 0 or tolerance >= 1:
        return candidates
    ordered = sorted(candidates, key=lambda c: _safe_float(c.get("confidence"), 0.0), reverse=True)
    kept, kept_boxes = [], []
    for candidate in ordered:
        box = _candidate_box(candidate)
        if any(_box_overlap_ratio(box, kb) > tolerance for kb in kept_boxes):
            continue
        kept.append(candidate)
        kept_boxes.append(box)
    return kept


def _parse_roboflow_detections(data, frame_shape=None):
    """
    Return every worm detection at or above RF_CONFIDENCE, sorted by confidence.
    Each candidate carries the full keypoint list so a later pass can associate
    the same worm across frames instead of jumping to the per-frame best match.
    """
    preds = _find_predictions(data)
    if not preds:
        return []

    scale_x, scale_y = _coordinate_scale_for_response(data, frame_shape)
    detections = []
    for pred in preds:
        det_conf = _safe_float(pred.get("confidence"), 0.0)
        if det_conf < RF_CONFIDENCE:
            continue

        kps = []
        for kp in pred.get("keypoints") or []:
            if "x" not in kp or "y" not in kp:
                continue
            name = str(kp.get("class_name", kp.get("class", ""))).strip()
            conf = _safe_float(kp.get("confidence"), det_conf)
            kps.append((float(kp["x"]) * scale_x, float(kp["y"]) * scale_y, conf, name))
        if not kps:
            continue

        candidate = {
            "x": _scale_optional(pred.get("x"), scale_x),
            "y": _scale_optional(pred.get("y"), scale_y),
            "width": _scale_optional(pred.get("width"), scale_x),
            "height": _scale_optional(pred.get("height"), scale_y),
            "confidence": det_conf,
            "class": pred.get("class_name", pred.get("class", "")),
            "keypoints": kps,
        }
        if _candidate_center(candidate) is not None:
            detections.append(candidate)

    return sorted(detections, key=lambda c: c["confidence"], reverse=True)


def _parse_roboflow_keypoints(data, frame_shape=None):
    """
    Backwards-compatible helper: return keypoints for the highest-confidence
    detection after applying the Roboflow confidence floor.
    """
    detections = _parse_roboflow_detections(data, frame_shape=frame_shape)
    return detections[0]["keypoints"] if detections else None


def _roboflow_unconfigured_detail() -> str:
    return (
        "Roboflow backend selected but ROBOFLOW_API_KEY is not set. Export "
        "ROBOFLOW_API_KEY (and optionally ROBOFLOW_API_URL, default "
        f"{_rf_base()}) to use the temporary detector."
    )


def make_detector(overlap_tolerance: float = DEFAULT_OVERLAP_TOLERANCE):
    """
    Build a `frame -> list[candidate]` callable for the active backend. Each
    candidate is one worm detection with all of its keypoints, after dropping
    boxes that overlap an already-kept one by more than `overlap_tolerance`.
    Raises HTTPException(503) if that backend isn't ready.
    """
    backend = active_backend()
    if backend == "roboflow":
        if not RF_API_KEY:
            raise HTTPException(status_code=503, detail=_roboflow_unconfigured_detail())

        def detect(frame):
            return _dedupe_overlapping(
                _parse_roboflow_detections(_roboflow_infer(frame), frame_shape=frame.shape), overlap_tolerance
            )

        return detect

    model = get_model()  # may raise HTTPException(503)

    def detect(frame):
        return _dedupe_overlapping(
            _extract_worm_candidates(model(frame, verbose=False)[0]), overlap_tolerance
        )

    return detect


# --------------------------------------------------------------------------- #
# Core per-video analysis
# --------------------------------------------------------------------------- #

_YOLO_KP_NAMES = ["head", "tail"]  # index 0 = head, 1 = tail (build-spec contract)


def _extract_worm_candidates(result):
    """
    From a single-frame YOLO result, return all worm candidates at or above the
    confidence floor. Index 0 is named "head", index 1 "tail", per the model
    contract.
    """
    kpts = getattr(result, "keypoints", None)
    if kpts is None or kpts.xy is None or len(kpts.xy) == 0:
        return []

    xy_all = kpts.xy.cpu().numpy()  # shape (N, K, 2)
    if xy_all.shape[0] == 0:
        return []
    kconf_all = kpts.conf.cpu().numpy() if kpts.conf is not None else None

    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
        box_confs = boxes.conf.cpu().numpy()
    else:
        box_confs = np.ones((xy_all.shape[0],), dtype=float)
    try:
        box_wh = boxes.xywh.cpu().numpy() if boxes is not None and boxes.xywh is not None else None
    except Exception:  # noqa: BLE001
        box_wh = None

    candidates = []
    for det_i in range(xy_all.shape[0]):
        det_conf = float(box_confs[det_i]) if det_i < len(box_confs) else 1.0
        if det_conf < RF_CONFIDENCE:
            continue

        xy = xy_all[det_i]
        if xy.shape[0] == 0:
            continue
        kconf = kconf_all[det_i] if kconf_all is not None else None
        keypoints = []
        for i in range(xy.shape[0]):
            conf = float(kconf[i]) if kconf is not None else det_conf
            name = _YOLO_KP_NAMES[i] if i < len(_YOLO_KP_NAMES) else f"kp{i}"
            keypoints.append((float(xy[i][0]), float(xy[i][1]), conf, name))

        candidate = {
            "x": float(np.mean(xy[:, 0])),
            "y": float(np.mean(xy[:, 1])),
            "width": float(box_wh[det_i][2]) if box_wh is not None and det_i < len(box_wh) else None,
            "height": float(box_wh[det_i][3]) if box_wh is not None and det_i < len(box_wh) else None,
            "confidence": det_conf,
            "class": "worm",
            "keypoints": keypoints,
        }
        candidates.append(candidate)

    return sorted(candidates, key=lambda c: c["confidence"], reverse=True)


def _extract_keypoints(result):
    """Backwards-compatible helper for older tests/tools."""
    candidates = _extract_worm_candidates(result)
    return candidates[0]["keypoints"] if candidates else None


def _distance_sq(a, b) -> float:
    if a is None or b is None:
        return float("inf")
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _top_seed_candidates(candidates: list[dict], max_tracks: int, pcutoff: float | None = None) -> list[dict]:
    viable = []
    for candidate in candidates or []:
        center = _center_with_conf(candidate)
        if not candidate.get("keypoints") or center is None:
            continue
        if pcutoff is not None and center[2] < pcutoff:
            continue
        viable.append(candidate)
    return sorted(viable, key=lambda c: _safe_float(c.get("confidence"), 0.0), reverse=True)[:max_tracks]


def _assign_candidates_to_tracks(tracks: list[dict], candidates: list[dict], pcutoff: float | None = None) -> None:
    viable = []
    for candidate in candidates or []:
        center = _center_with_conf(candidate)
        if not candidate.get("keypoints") or center is None:
            continue
        if pcutoff is not None and center[2] < pcutoff:
            continue
        viable.append(candidate)
    if not viable:
        for track in tracks:
            track["frames"].append(None)
            track["centers"].append(None)
        return

    pairs = []
    for track_i, track in enumerate(tracks):
        for cand_i, candidate in enumerate(viable):
            pairs.append((
                _distance_sq(track["last_center"], _candidate_center(candidate)),
                -_safe_float(candidate.get("confidence"), 0.0),
                track_i,
                cand_i,
            ))

    assigned_tracks = set()
    assigned_candidates = set()
    assigned = {}
    for _, _, track_i, cand_i in sorted(pairs):
        if track_i in assigned_tracks or cand_i in assigned_candidates:
            continue
        dist_sq = _distance_sq(tracks[track_i]["last_center"], _candidate_center(viable[cand_i]))
        if MAX_MATCH_DISTANCE_PX > 0 and dist_sq > MAX_MATCH_DISTANCE_PX ** 2:
            continue
        assigned_tracks.add(track_i)
        assigned_candidates.add(cand_i)
        assigned[track_i] = viable[cand_i]

    for track_i, track in enumerate(tracks):
        candidate = assigned.get(track_i)
        if candidate is None:
            track["frames"].append(None)
            track["centers"].append(None)
            continue
        track["frames"].append(candidate["keypoints"])
        track["centers"].append(_center_with_conf(candidate))
        track["last_center"] = _candidate_center(candidate)


def _select_same_worm_tracks(
    frames_candidates: list[list[dict]],
    max_tracks: int = MAX_WORMS_PER_VIDEO,
    pcutoff: float | None = DEFAULT_PCUTOFF,
):
    """
    Seed up to `max_tracks` worm identities from the first frame with detections,
    using confidence order. Later frames assign detections one-to-one by nearest
    center so the same identities are followed throughout the video.
    """
    seed_frame = None
    seeds = []
    for frame_i, candidates in enumerate(frames_candidates):
        seeds = _top_seed_candidates(candidates, max_tracks, pcutoff=pcutoff)
        if seeds:
            seed_frame = frame_i
            break

    if seed_frame is None:
        return []

    tracks = []
    for track_i, seed in enumerate(seeds, start=1):
        tracks.append({
            "worm_id": track_i,
            "seed_frame": seed_frame,
            "seed_confidence": _safe_float(seed.get("confidence"), 0.0),
            "last_center": _candidate_center(seed),
            "frames": [None for _ in range(seed_frame)] + [seed["keypoints"]],
            "centers": [None for _ in range(seed_frame)] + [_center_with_conf(seed)],
        })

    for candidates in frames_candidates[seed_frame + 1:]:
        _assign_candidates_to_tracks(tracks, candidates, pcutoff=pcutoff)

    return tracks


def _select_same_worm_track(frames_candidates: list[list[dict]]):
    """Backwards-compatible helper for tests/tools that expect one track."""
    tracks = _select_same_worm_tracks(frames_candidates, max_tracks=1)
    return tracks[0]["frames"] if tracks else []


def _read_video_frame(path: str, frame_index: int):
    """Read a specific source video frame for the visual tracking preview."""
    target = max(0, int(frame_index))

    cap = cv2.VideoCapture(path)
    try:
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            if ok:
                return frame
    finally:
        cap.release()

    # Some codecs ignore random seek; fall back to a sequential read.
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for current in range(target + 1):
            ok, frame = cap.read()
            if not ok:
                return None
            if current == target:
                return frame
    finally:
        cap.release()
    return None


def _kp_xy(kp):
    if kp is None or len(kp) < 2:
        return None
    x, y = float(kp[0]), float(kp[1])
    if np.isnan(x) or np.isnan(y):
        return None
    return x, y


def _track_seed_keypoints(track: dict):
    seed_frame = int(track.get("seed_frame", 0))
    frames = track.get("frames") or []
    if seed_frame < 0 or seed_frame >= len(frames):
        return []
    return frames[seed_frame] or []


def _keypoints_center(keypoints):
    points = [_kp_xy(kp) for kp in (keypoints or [])]
    points = [p for p in points if p is not None]
    if not points:
        return None
    xs, ys = zip(*points)
    return float(np.mean(xs)), float(np.mean(ys))


def _draw_label(image, text: str, x: float, y: float, color, selected: bool) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55 if image.shape[1] >= 520 else 0.45
    thickness = 2 if selected else 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    ix = int(max(0, min(x, image.shape[1] - tw - pad * 2 - 1)))
    iy = int(max(th + pad, min(y, image.shape[0] - baseline - pad - 1)))
    bg = (18, 21, 24) if selected else (32, 38, 43)
    cv2.rectangle(
        image,
        (ix, iy - th - pad),
        (ix + tw + pad * 2, iy + baseline + pad),
        bg,
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (ix, iy - th - pad),
        (ix + tw + pad * 2, iy + baseline + pad),
        color,
        thickness=1,
    )
    cv2.putText(image, text, (ix + pad, iy), font, scale, color, thickness, cv2.LINE_AA)


def _encode_jpeg_data_url(image, quality: int = TRACKING_FRAME_JPEG_QUALITY) -> str | None:
    ok, buf = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _resize_for_preview(frame, max_width: int):
    image = frame.copy()
    scale = 1.0
    if image.shape[1] > max_width:
        scale = max_width / image.shape[1]
        new_size = (max_width, max(1, int(round(image.shape[0] * scale))))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image, scale


WORM_COLORS = [
    (168, 231, 110),  # green  (#6EE7A8)
    (61, 163, 232),   # amber  (#E8A33D)
    (240, 180, 110),  # blue
    (200, 120, 255),  # magenta
    (140, 255, 200),  # teal
]


def _worm_color(worm_id: int):
    return WORM_COLORS[(int(worm_id) - 1) % len(WORM_COLORS)]


class _LiveWormTracker:
    """
    Assigns STABLE worm ids to detections as frames stream in, mirroring the
    batch identity logic (_select_same_worm_tracks): seed ids on the first frame
    with detections (by confidence), then match each later frame one-to-one by
    nearest center. This stops a worm's id from flipping frame to frame in the
    live preview, and the ids line up with the final per-worm results.
    """

    def __init__(self, max_tracks: int):
        self.max_tracks = max_tracks
        self.tracks = []  # [{worm_id, last_center}]
        self.seeded = False

    def update(self, candidates):
        if not self.seeded:
            seeds = _top_seed_candidates(candidates, self.max_tracks)
            if not seeds:
                return []
            self.tracks = [
                {"worm_id": i, "last_center": _candidate_center(seed)}
                for i, seed in enumerate(seeds, start=1)
            ]
            self.seeded = True
            return [(seed, t["worm_id"]) for seed, t in zip(seeds, self.tracks)]

        viable = [c for c in (candidates or []) if c.get("keypoints") and _candidate_center(c) is not None]
        pairs = []
        for ti, track in enumerate(self.tracks):
            for ci, candidate in enumerate(viable):
                pairs.append((
                    _distance_sq(track["last_center"], _candidate_center(candidate)),
                    -_safe_float(candidate.get("confidence"), 0.0),
                    ti, ci,
                ))
        assigned_t, assigned_c, out_map = set(), set(), {}
        for _, _, ti, ci in sorted(pairs):
            if ti in assigned_t or ci in assigned_c:
                continue
            dist_sq = _distance_sq(self.tracks[ti]["last_center"], _candidate_center(viable[ci]))
            if MAX_MATCH_DISTANCE_PX > 0 and dist_sq > MAX_MATCH_DISTANCE_PX ** 2:
                continue
            assigned_t.add(ti); assigned_c.add(ci); out_map[ti] = ci

        labeled = []
        for ti, track in enumerate(self.tracks):
            ci = out_map.get(ti)
            if ci is None:
                continue
            candidate = viable[ci]
            track["last_center"] = _candidate_center(candidate)
            labeled.append((candidate, track["worm_id"]))
        return labeled


def _annotate_live_frame(frame, labeled: list):
    """labeled: list of (candidate, worm_id) carrying STABLE ids across frames."""
    image, scale = _resize_for_preview(frame, LIVE_FRAME_MAX_WIDTH)
    endpoint_color = (61, 163, 232)
    if not labeled:
        _draw_label(image, "no worm detected", 12, 28, (132, 144, 152), False)
        return image

    for candidate, worm_id in labeled:
        color = _worm_color(worm_id)
        points = []
        for kp in candidate.get("keypoints") or []:
            xy = _kp_xy(kp)
            if xy is None:
                continue
            points.append((int(round(xy[0] * scale)), int(round(xy[1] * scale)), kp))

        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(image, points[i][:2], points[i + 1][:2], color, 2, cv2.LINE_AA)
        for px, py, kp in points:
            cv2.circle(image, (px, py), 4, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, (px, py), 6, (18, 21, 24), thickness=1, lineType=cv2.LINE_AA)
            if len(kp) > 3 and kp[3]:
                _draw_label(image, str(kp[3])[:16], px + 7, py - 7, endpoint_color, False)

        # Draw the tracked CENTER: the midpoint of the two endpoints.
        if len(points) >= 2:
            mx = (points[0][0] + points[1][0]) // 2
            my = (points[0][1] + points[1][1]) // 2
            cv2.circle(image, (mx, my), 5, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, (mx, my), 7, color, thickness=2, lineType=cv2.LINE_AA)

        center = _candidate_center(candidate)
        if center is not None:
            conf = _round_or_none(candidate.get("confidence"))
            label = f"worm {worm_id}" + (f" {conf:.2f}" if conf is not None else "")
            cx, cy = center
            _draw_label(image, label, cx * scale + 8, cy * scale - 12, color, True)
    return image


def _live_frame_payload(frame, labeled: list, frame_index: int, analyzed_index: int, native_fps: float):
    annotated = _annotate_live_frame(frame, labeled)
    data_url = _encode_jpeg_data_url(annotated, quality=LIVE_FRAME_JPEG_QUALITY)
    worm_ids = [int(worm_id) for _, worm_id in labeled]
    return {
        "frame_index": int(frame_index),
        "analyzed_index": int(analyzed_index),
        "time_s": round(frame_index / native_fps, 3) if native_fps else None,
        "detections": len(labeled),
        "worm_ids": worm_ids,
        "image": data_url,
    }


def _annotate_tracking_frame(frame, tracks: list[dict], selected_worm_id: int):
    image, scale = _resize_for_preview(frame, TRACKING_FRAME_MAX_WIDTH)

    selected_color = (168, 231, 110)  # #6EE7A8, BGR for OpenCV
    muted_color = (132, 144, 152)
    endpoint_color = (61, 163, 232)  # #E8A33D, BGR

    for track in tracks:
        worm_id = int(track.get("worm_id", 0))
        keypoints = _track_seed_keypoints(track)
        center = _keypoints_center(keypoints)
        if center is None:
            continue

        selected = worm_id == selected_worm_id
        color = selected_color if selected else muted_color
        thickness = 3 if selected else 1
        radius = 5 if selected else 3
        points = []
        for kp in keypoints:
            xy = _kp_xy(kp)
            if xy is None:
                continue
            points.append((int(round(xy[0] * scale)), int(round(xy[1] * scale)), kp))

        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(image, points[i][:2], points[i + 1][:2], color, thickness, cv2.LINE_AA)

        for px, py, kp in points:
            cv2.circle(image, (px, py), radius, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, (px, py), radius + 2, (18, 21, 24), thickness=1, lineType=cv2.LINE_AA)
            if selected and len(kp) > 3 and kp[3]:
                _draw_label(image, str(kp[3])[:16], px + 8, py - 8, endpoint_color, False)

        # Draw the tracked CENTER: the midpoint of the two endpoints.
        if len(points) >= 2:
            mx = (points[0][0] + points[1][0]) // 2
            my = (points[0][1] + points[1][1]) // 2
            cv2.circle(image, (mx, my), radius + 1, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(image, (mx, my), radius + 3, color, thickness=2, lineType=cv2.LINE_AA)

        cx, cy = center
        _draw_label(image, f"worm {worm_id}", cx * scale + 8, cy * scale - 12, color, selected)

    return image


def _tracking_frame_payloads(path: str, tracks: list[dict], native_fps: float, step: int) -> dict[int, dict]:
    if not tracks:
        return {}

    seed_frame = int(tracks[0].get("seed_frame", 0))
    source_frame_index = seed_frame * max(1, int(step))
    frame = _read_video_frame(path, source_frame_index)
    if frame is None:
        return {}

    payloads = {}
    for track in tracks:
        worm_id = int(track["worm_id"])
        annotated = _annotate_tracking_frame(frame, tracks, selected_worm_id=worm_id)
        data_url = _encode_jpeg_data_url(annotated)
        if not data_url:
            continue
        payloads[worm_id] = {
            "tracking_frame_image": data_url,
            "tracking_frame_index": source_frame_index,
            "tracking_frame_analyzed_index": seed_frame,
            "tracking_frame_time_s": round(source_frame_index / native_fps, 3) if native_fps else None,
        }
    return payloads


def _per_second_speed_bins(speed_series_px_s: list, effective_fps: float, px_per_mm):
    """
    Average the worm's speed into one-second buckets. Each speed sample is the
    interval between two analyzed frames, so intervals are duration-weighted when
    they straddle a second boundary. Untracked (None/NaN) intervals are skipped,
    so a bucket's average only reflects the time the worm was actually tracked.
    """
    if not speed_series_px_s or effective_fps <= 0:
        return []

    dt = 1.0 / effective_fps
    total_duration = len(speed_series_px_s) * dt
    n_bins = int(np.ceil(total_duration))
    weighted_sums = [0.0 for _ in range(n_bins)]
    valid_durations = [0.0 for _ in range(n_bins)]

    for i, speed in enumerate(speed_series_px_s):
        if speed is None:
            continue
        value = _safe_float(speed, np.nan)
        if np.isnan(value):
            continue

        interval_start = i * dt
        interval_end = (i + 1) * dt
        first_bin = int(np.floor(interval_start))
        last_bin = int(np.ceil(interval_end))
        for bin_i in range(first_bin, min(last_bin, n_bins)):
            overlap = min(interval_end, bin_i + 1.0) - max(interval_start, float(bin_i))
            if overlap <= 0:
                continue
            weighted_sums[bin_i] += value * overlap
            valid_durations[bin_i] += overlap

    bins = []
    for second in range(n_bins):
        avg_px = (
            weighted_sums[second] / valid_durations[second]
            if valid_durations[second] > 0
            else None
        )
        bucket = {
            "second": second,
            "start_s": round(float(second), 3),
            "end_s": round(float(min(second + 1.0, total_duration)), 3),
            "avg_speed_px_s": _round_or_none(avg_px),
            "tracked_duration_s": round(float(valid_durations[second]), 3),
        }
        if px_per_mm:
            bucket["avg_speed_mm_s"] = (
                _round_or_none(avg_px / px_per_mm) if avg_px is not None else None
            )
        bins.append(bucket)

    return bins


def _smooth_positions(xs, ys, window: int):
    """
    Centered moving average of the position series over an odd `window`,
    ignoring NaN gaps. A position that is itself NaN (untracked) stays NaN, so
    gaps are preserved and the NaN-gap speed logic is unaffected. Damps tracking
    jitter, which otherwise biases speed upward (summed |displacement| only grows
    with noise). window <= 1 returns the series unchanged.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if window is None or window <= 1 or len(xs) == 0:
        return xs, ys
    r = window // 2
    n = len(xs)
    sx, sy = xs.copy(), ys.copy()
    for i in range(n):
        if np.isnan(xs[i]) or i - r < 0 or i + r + 1 > n:
            continue  # keep gaps as gaps; leave array-edge points unsmoothed
        with np.errstate(invalid="ignore"):
            sx[i] = np.nanmean(xs[i - r:i + r + 1])
            sy[i] = np.nanmean(ys[i - r:i + r + 1])
    return sx, sy


def _analyze_worm_track(
    path: str,
    track: dict,
    native_fps: float,
    effective_fps: float,
    frames_analyzed: int,
    pcutoff: float,
    px_per_mm,
    tracked_worm_count: int,
):
    frames_kps = track["frames"]
    n_kp = max((len(f) for f in frames_kps if f), default=0)
    if n_kp == 0:
        return None

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

    # Per-endpoint displacement (kept for reference in the response).
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

    # Track the worm's CENTER — the detection center, which is the midpoint of
    # the two endpoints when both fire and stays on the body when only one does.
    # It doesn't lurch on keypoint dropouts, so speed reflects real body motion.
    # A frame where the worm wasn't detected/matched is still a NaN gap.
    xs, ys, confs = [], [], []
    for c in (track.get("centers") or []):
        if c is None:
            xs.append(np.nan); ys.append(np.nan); confs.append(0.0)
        else:
            xs.append(c[0]); ys.append(c[1]); confs.append(c[2])

    # Damp tracking jitter (which biases speed upward) before measuring.
    xs, ys = _smooth_positions(xs, ys, SMOOTH_WINDOW)

    metrics = compute_track_speed(xs, ys, confs, fps=effective_fps, pcutoff=pcutoff, px_per_mm=px_per_mm)
    disp = compute_displacement(xs, ys, confs, pcutoff=pcutoff, px_per_mm=px_per_mm)
    per_second_speed = _per_second_speed_bins(
        metrics["speed_series_px_s"], effective_fps=effective_fps, px_per_mm=px_per_mm
    )

    # time_series_s is the x-axis for the chart: speed[i] covers the interval
    # between analyzed frame i and i+1, labeled by the time of the later frame.
    n_speeds = len(metrics["speed_series_px_s"])
    time_series_s = [round((i + 1) / effective_fps, 3) for i in range(n_speeds)]

    # When the farthest-reach happened (seconds from start), for the chart marker.
    far_frame = disp.get("farthest_displacement_frame")
    far_time = round(far_frame / effective_fps, 3) if far_frame is not None else None
    seed_frame = int(track.get("seed_frame", 0))

    return {
        "video": os.path.basename(path),
        "worm_id": int(track["worm_id"]),
        "worm_label": f"worm {int(track['worm_id'])}",
        "tracked_worm_count": tracked_worm_count,
        "seed_frame": seed_frame,
        "seed_time_s": round(seed_frame / effective_fps, 3),
        "seed_confidence": _round_or_none(track.get("seed_confidence")),
        "backend": active_backend(),
        "native_fps": round(float(native_fps), 3),
        "effective_fps": round(float(effective_fps), 3),
        "frames_analyzed": frames_analyzed,
        "frames_tracked_pct": round(metrics["frames_tracked_pct"], 3),
        "avg_speed_px_s": _round_or_none(metrics["avg_speed_px_s"]),
        "max_speed_px_s": _round_or_none(metrics["max_speed_px_s"]),
        "total_distance_px": round(metrics["total_distance_px"], 3),
        "farthest_displacement_px": _round_or_none(disp["farthest_displacement_px"]),
        "net_displacement_px": _round_or_none(disp["net_displacement_px"]),
        "farthest_displacement_time_s": far_time,
        "tracked_point": "center",
        "tracked_endpoint_name": "center (worm body center)",
        "endpoints": per_endpoint,
        "speed_series_px_s": metrics["speed_series_px_s"],
        "per_second_speed": per_second_speed,
        "time_series_s": time_series_s,
        "avg_speed_mm_s": _round_or_none(metrics.get("avg_speed_mm_s")),
        "max_speed_mm_s": _round_or_none(metrics.get("max_speed_mm_s")),
        "total_distance_mm": _round_or_none(metrics.get("total_distance_mm")),
        "farthest_displacement_mm": _round_or_none(disp.get("farthest_displacement_mm")),
        "net_displacement_mm": _round_or_none(disp.get("net_displacement_mm")),
    }


def _infer_concurrency() -> int:
    """
    How many frames to push through the detector at once. The hosted Roboflow
    backend is one HTTP round-trip per frame, so a video's frames are
    network-bound and running several in flight cuts wall-clock time sharply.
    The local YOLO model stays sequential by default because concurrent
    inference on a single model handle isn't guaranteed thread-safe. Override
    with WORM_INFER_CONCURRENCY (a positive integer).
    """
    raw = os.environ.get("WORM_INFER_CONCURRENCY", "").strip()
    if raw.isdigit() and int(raw) >= 1:
        return int(raw)
    return 6 if active_backend() == "roboflow" else 1


def analyze_video_file(path: str, sample_fps: float, pcutoff: float, px_per_mm, progress_callback=None, overlap_tolerance: float = DEFAULT_OVERLAP_TOLERANCE):
    """
    Run inference over one video and return one result dict per tracked worm.

    Raises ValueError for un-openable / non-video files so callers can decide
    whether to abort (single) or record a per-video error (batch).
    """
    detect = make_detector(overlap_tolerance)  # backend-agnostic; may raise HTTPException(503)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open file as a video.")

    # Per-video native fps — never assume a global value (spec §4).
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if native_fps <= 0:
        native_fps = 30.0
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0.")
    step = max(1, round(native_fps / sample_fps))
    effective_fps = native_fps / step
    total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames_to_analyze = int(np.ceil(total_frames_raw / step)) if total_frames_raw > 0 else None
    if progress_callback:
        progress_callback({
            "type": "video_start",
            "native_fps": round(float(native_fps), 3),
            "effective_fps": round(float(effective_fps), 3),
            "total_frames": total_frames_raw or None,
            "frames_to_analyze": frames_to_analyze,
            "sample_step": step,
        })

    # Pass 1 — decode only the frames we will actually run inference on, kept in
    # frame order. Decoding is cheap and local; the expensive part is detection.
    sampled = []  # (analyzed_index, source_frame_index, frame)
    frame_idx = 0
    analyzed_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            sampled.append((analyzed_idx, frame_idx, frame))
            analyzed_idx += 1
        frame_idx += 1
    cap.release()

    # Pass 2 — run the detector. The hosted Roboflow backend is one HTTP
    # round-trip per frame, so a batch of frames is network-bound: run several
    # concurrently to cut wall-clock time. Results are consumed strictly IN
    # ORDER, so the stable-identity live tracker and the final track assignment
    # are byte-for-byte identical to the old sequential path — only the network
    # waits overlap. The local model stays sequential by default.
    frames_candidates = [None] * len(sampled)  # each: list[candidate]
    live_tracker = _LiveWormTracker(MAX_WORMS_PER_VIDEO) if progress_callback else None

    def _run_detect(item):
        a_idx, f_idx, frame = item
        return a_idx, f_idx, frame, (detect(frame) or [])

    def _consume(a_idx, f_idx, frame, candidates):
        frames_candidates[a_idx] = candidates
        labeled = live_tracker.update(candidates) if live_tracker else []
        if progress_callback and a_idx % LIVE_FRAME_EVERY_N == 0:
            progress_callback({
                "type": "frame",
                **_live_frame_payload(
                    frame=frame,
                    labeled=labeled,
                    frame_index=f_idx,
                    analyzed_index=a_idx,
                    native_fps=native_fps,
                ),
                "frames_to_analyze": frames_to_analyze,
            })

    concurrency = _infer_concurrency()
    if concurrency > 1 and len(sampled) > 1:
        # ThreadPoolExecutor.map runs up to `concurrency` detections at once but
        # yields results in submission order, so the tracker sees frames in
        # sequence while the network waits overlap.
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for a_idx, f_idx, frame, candidates in pool.map(_run_detect, sampled):
                _consume(a_idx, f_idx, frame, candidates)
    else:
        for item in sampled:
            _consume(*_run_detect(item))

    frames_analyzed = len(frames_candidates)
    if frames_analyzed == 0:
        raise ValueError("No frames could be decoded from this file.")

    worm_tracks = _select_same_worm_tracks(
        frames_candidates, max_tracks=MAX_WORMS_PER_VIDEO, pcutoff=pcutoff
    )
    if not worm_tracks:
        raise ValueError("No worm/keypoints were ever detected in this video.")

    tracking_frame_payloads = _tracking_frame_payloads(
        path=path,
        tracks=worm_tracks,
        native_fps=native_fps,
        step=step,
    )

    results = []
    for track in worm_tracks:
        result = _analyze_worm_track(
            path=path,
            track=track,
            native_fps=native_fps,
            effective_fps=effective_fps,
            frames_analyzed=frames_analyzed,
            pcutoff=pcutoff,
            px_per_mm=px_per_mm,
            tracked_worm_count=len(worm_tracks),
        )
        if result is not None:
            result.update(tracking_frame_payloads.get(result["worm_id"], {}))
            results.append(result)

    if not results:
        raise ValueError("No worm/keypoints were ever detected in this video.")
    if progress_callback:
        progress_callback({
            "type": "video_done",
            "frames_analyzed": frames_analyzed,
            "worm_count": len(results),
        })
    return results


def _choose_endpoint(per_endpoint: list[dict]) -> int:
    """
    Pick which endpoint index to report. Default: the one that moved the
    farthest from its start (greatest displacement).
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
    "worm_id",
    "seed_confidence",
    "seed_frame",
    "seed_time_s",
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


# Every per-worm data point to average across the batch: (key, mm-only?).
_AVERAGE_METRICS = [
    ("frames_tracked_pct", False),
    ("avg_speed_px_s", False),
    ("max_speed_px_s", False),
    ("total_distance_px", False),
    ("farthest_displacement_px", False),
    ("net_displacement_px", False),
    ("avg_speed_mm_s", True),
    ("max_speed_mm_s", True),
    ("total_distance_mm", True),
    ("farthest_displacement_mm", True),
    ("net_displacement_mm", True),
]


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _batch_summary(results: list[dict], used_mm: bool) -> dict:
    """
    Aggregate the headline numbers the user asked for across every tracked
    worm in the batch:
      - average speed the worms are moving (mean of per-worm averages)
      - how far the farthest tracked center moved, and which video that was
      - the mean of every other reported data point (distance, endpoint
        displacement, % tracked, ...), across all videos/worms in the batch.
    Failed videos and worms with no trackable value for a given metric are
    excluded from that metric's mean.
    """
    speed_key = "avg_speed_mm_s" if used_mm else "avg_speed_px_s"
    far_key = "farthest_displacement_mm" if used_mm else "farthest_displacement_px"
    ok = [r for r in results if "error" not in r]

    avg_speed = _mean([r.get(speed_key) for r in ok])

    far_video, far_worm_id, far_value = None, None, None
    for r in ok:
        v = r.get(far_key)
        if v is not None and (far_value is None or v > far_value):
            far_value, far_video, far_worm_id = v, r.get("video"), r.get("worm_id")

    # Mean of every numeric data point across all analyzed worms/videos, in
    # both px and (when calibrated) mm units — not just the headline speed.
    averages = {
        key: _mean([r.get(key) for r in ok])
        for key, mm_only in _AVERAGE_METRICS
        if not mm_only or used_mm
    }

    return {
        "videos_total": len(results),
        "videos_ok": len(ok),
        "videos_failed": len(results) - len(ok),
        "units_speed": "mm/s" if used_mm else "px/s",
        "units_distance": "mm" if used_mm else "px",
        "avg_speed": avg_speed,
        "farthest_displacement": round(far_value, 3) if far_value is not None else None,
        "farthest_displacement_video": far_video,
        "farthest_displacement_worm_id": far_worm_id,
        "averages": averages,
    }


def _per_second_csv_columns(results: list[dict], used_calibration: bool):
    """One CSV column per second bucket (px/s, plus mm/s when calibrated)."""
    max_bins = max((len(r.get("per_second_speed", [])) for r in results if "error" not in r), default=0)
    columns = []
    for second in range(max_bins):
        columns.append((f"second_{second}_{second + 1}_avg_speed_px_s", second, "avg_speed_px_s"))
        if used_calibration:
            columns.append((f"second_{second}_{second + 1}_avg_speed_mm_s", second, "avg_speed_mm_s"))
    return columns


def _build_csv(results: list[dict], used_calibration: bool) -> str:
    second_columns = _per_second_csv_columns(results, used_calibration)
    base_columns = CSV_BASE_COLUMNS + (CSV_MM_COLUMNS if used_calibration else [])
    columns = base_columns + [name for name, _, _ in second_columns]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for r in results:
        if "error" in r:
            # Failed video: record name, leave the numeric cells blank so the
            # CSV stays a clean rectangle that lines up with the header.
            row = [r.get("video", "")] + ["" for _ in columns[1:]]
        else:
            row = [r.get(col, "") for col in base_columns]
            per_second = r.get("per_second_speed", [])
            for _, second, key in second_columns:
                value = per_second[second].get(key) if second < len(per_second) else ""
                row.append("" if value is None else value)
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
        fallback = "" if _rf_mode() == "model" else f" (fallback {RF_FALLBACK_MODEL_ID})"
        # Light reachability check against the inference server root.
        try:
            httpx.get(base, timeout=5)
            return {
                "ready": True,
                "backend": backend,
                "detail": f"Roboflow {target}{fallback} @ {base}, confidence {RF_CONFIDENCE:.2f}",
            }
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
    pcutoff: float = Form(DEFAULT_PCUTOFF),
    px_per_mm: float | None = Form(None),
    overlap_tolerance: float = Form(DEFAULT_OVERLAP_TOLERANCE),
):
    tmp_path = _save_upload_to_temp(file)
    try:
        results = analyze_video_file(tmp_path, sample_fps, pcutoff, px_per_mm, overlap_tolerance=overlap_tolerance)
        # Preserve the original uploaded filename rather than the temp name.
        for result in results:
            result["video"] = file.filename or result["video"]
        return {"video": file.filename or os.path.basename(tmp_path), "worms": results}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        _safe_unlink(tmp_path)


@app.post("/api/analyze-batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    sample_fps: float = Form(10.0),
    pcutoff: float = Form(DEFAULT_PCUTOFF),
    px_per_mm: float | None = Form(None),
    overlap_tolerance: float = Form(DEFAULT_OVERLAP_TOLERANCE),
):
    results: list[dict] = []
    for upload in files:
        tmp_path = _save_upload_to_temp(upload)
        try:
            video_results = analyze_video_file(tmp_path, sample_fps, pcutoff, px_per_mm, overlap_tolerance=overlap_tolerance)
            for res in video_results:
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


def _json_line(event: dict) -> str:
    return json.dumps(event, separators=(",", ":")) + "\n"


@app.post("/api/analyze-stream")
async def analyze_stream(
    files: list[UploadFile] = File(...),
    sample_fps: float = Form(10.0),
    pcutoff: float = Form(DEFAULT_PCUTOFF),
    px_per_mm: float | None = Form(None),
    overlap_tolerance: float = Form(DEFAULT_OVERLAP_TOLERANCE),
):
    jobs = []
    for upload in files:
        jobs.append({
            "name": upload.filename or "unknown",
            "path": _save_upload_to_temp(upload),
        })

    def event_stream():
        events: queue.Queue = queue.Queue(maxsize=8)
        sentinel = object()

        def put(event: dict) -> None:
            events.put(event)

        def worker() -> None:
            results: list[dict] = []
            try:
                for video_i, job in enumerate(jobs, start=1):
                    name = job["name"]
                    path = job["path"]

                    def progress(event: dict, video_name=name, index=video_i) -> None:
                        payload = dict(event)
                        payload["video"] = video_name
                        payload["video_index"] = index
                        payload["video_total"] = len(jobs)
                        put(payload)

                    try:
                        video_results = analyze_video_file(
                            path,
                            sample_fps=sample_fps,
                            pcutoff=pcutoff,
                            px_per_mm=px_per_mm,
                            progress_callback=progress,
                            overlap_tolerance=overlap_tolerance,
                        )
                        for res in video_results:
                            res["video"] = name
                            results.append(res)
                    except HTTPException as exc:
                        put({"type": "fatal_error", "detail": exc.detail})
                        return
                    except Exception as exc:  # noqa: BLE001 - keep batch behavior
                        error = str(exc)
                        results.append({"video": name, "error": error})
                        put({
                            "type": "video_error",
                            "video": name,
                            "video_index": video_i,
                            "video_total": len(jobs),
                            "error": error,
                        })

                csv_text = _build_csv(results, used_calibration=bool(px_per_mm))
                token = secrets.token_hex(16)
                _CSV_STORE[token] = csv_text
                put({
                    "type": "done",
                    "results": results,
                    "csv_token": token,
                    "summary": _batch_summary(results, used_mm=bool(px_per_mm)),
                })
            finally:
                for job in jobs:
                    _safe_unlink(job["path"])
                put(sentinel)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            event = events.get()
            if event is sentinel:
                break
            yield _json_line(event)

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        # X-Accel-Buffering: no tells nginx-style reverse proxies (e.g. Hugging
        # Face Spaces) NOT to buffer the response. Without it the proxy holds the
        # whole NDJSON stream and releases it at the end, so the live preview
        # stays blank the entire run and only "pops" to results when finished.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
