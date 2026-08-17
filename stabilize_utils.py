"""
stabilize_utils.py — AI video stabilization for shaky microscope clips.

Uses the Hugging Face `magic-leap-community/superpoint` keypoint detector
(via `transformers`) to find robust, learned feature points in each frame,
matches them frame-to-frame, and uses the matched points to estimate and
smooth out the camera-shake trajectory — then re-renders the video with that
motion removed.

This is the classic "point feature matching" stabilization recipe (track
points across frames, integrate the motion into a trajectory, smooth the
trajectory, warp each frame by the difference), swapping the hand-crafted
feature detector (goodFeaturesToTrack / ORB) for a learned one pulled from
the Hugging Face Hub — which holds up far better on the low-contrast,
low-texture microscope footage this app processes than corner detectors do.

`transformers`/`torch` are imported lazily (inside `_get_stabilizer`), same
as `ultralytics` in app.py, so the server still starts cleanly without them
and only pays the import/download cost if a caller actually asks to
stabilize a video.
"""

import os
import tempfile

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

HF_MODEL_ID = os.environ.get("WORM_STABILIZE_MODEL_ID", "magic-leap-community/superpoint").strip()


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    return int(_float_env(name, default))


# Keypoint score threshold. Microscope frames are low-contrast/low-texture,
# so this defaults lower than SuperPoint's own default (0.005) to keep enough
# points to match on. Lower further with WORM_STABILIZE_KEYPOINT_THRESHOLD if
# a clip still isn't finding enough matches to stabilize.
KEYPOINT_THRESHOLD = _float_env("WORM_STABILIZE_KEYPOINT_THRESHOLD", 0.003)
MAX_KEYPOINTS = _int_env("WORM_STABILIZE_MAX_KEYPOINTS", 1024)
# Lowe's ratio test cutoff for descriptor matching between consecutive
# frames. Higher = more permissive = more (noisier) matches kept; kept
# permissive by default for the same low-texture-footage reason as above.
MATCH_RATIO = _float_env("WORM_STABILIZE_MATCH_RATIO", 0.85)
# Frames are resized to this max dimension before keypoint detection, purely
# for speed — the estimated transform is scaled back up to full resolution.
DETECT_MAX_DIMENSION = _int_env("WORM_STABILIZE_MAX_DIMENSION", 480)
# How many frames of trajectory to average over when smoothing out the
# shake. Bigger = smoother but can start to eat real camera pans.
SMOOTHING_RADIUS = _int_env("WORM_STABILIZE_SMOOTHING_RADIUS", 15)
# Below this many good matches between two frames, trust nothing and treat
# that frame pair as motionless rather than warp on a handful of possibly-bad
# correspondences.
MIN_MATCHES = _int_env("WORM_STABILIZE_MIN_MATCHES", 4)
# Crop the stabilized frame in slightly to hide the black/replicated borders
# that appear at the edges once shake is removed, then rescale back up to the
# original frame size.
BORDER_CROP_RATIO = min(0.2, max(0.0, _float_env("WORM_STABILIZE_BORDER_CROP", 0.04)))

# Lazily-populated model/processor handles and the last load error, mirroring
# app.py's get_model() pattern.
_PROCESSOR = None
_MODEL = None
_LOAD_ERROR: str | None = None


class StabilizationUnavailableError(RuntimeError):
    """Raised when the AI stabilizer's dependencies or model can't be loaded."""


def _get_stabilizer():
    """Lazily load the Hugging Face SuperPoint keypoint model."""
    global _PROCESSOR, _MODEL, _LOAD_ERROR

    if _MODEL is not None:
        return _PROCESSOR, _MODEL
    if _LOAD_ERROR is not None:
        raise StabilizationUnavailableError(_LOAD_ERROR)

    try:
        from transformers import AutoImageProcessor, SuperPointForKeypointDetection
    except ImportError as exc:
        _LOAD_ERROR = (
            "AI video stabilization needs `torch` and `transformers` "
            "(pip install torch transformers). Install them, or leave the "
            "stabilize option off."
        )
        raise StabilizationUnavailableError(_LOAD_ERROR) from exc

    try:
        processor = AutoImageProcessor.from_pretrained(HF_MODEL_ID)
        model = SuperPointForKeypointDetection.from_pretrained(HF_MODEL_ID)
        model.eval()
    except Exception as exc:  # noqa: BLE001 - surface any download/load failure
        _LOAD_ERROR = (
            f"Failed to load the Hugging Face stabilization model "
            f"'{HF_MODEL_ID}': {exc}. Check network access to "
            f"huggingface.co (first run downloads the weights), or set "
            f"WORM_STABILIZE_MODEL_ID to a reachable alternative."
        )
        raise StabilizationUnavailableError(_LOAD_ERROR) from exc

    _PROCESSOR, _MODEL = processor, model
    return _PROCESSOR, _MODEL


# --------------------------------------------------------------------------- #
# Keypoint detection / matching
# --------------------------------------------------------------------------- #

def _resize_for_detection(frame_rgb: np.ndarray, max_dim: int):
    h, w = frame_rgb.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale >= 1.0:
        return frame_rgb, 1.0
    small = cv2.resize(
        frame_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )
    return small, scale


def _detect_keypoints(processor, model, frame_rgb: np.ndarray):
    """Return (keypoints Nx2 float32 in pixel coords, descriptors NxD float32)."""
    import torch

    inputs = processor(images=frame_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    sizes = [(frame_rgb.shape[0], frame_rgb.shape[1])]
    processed = processor.post_process_keypoint_detection(outputs, sizes)[0]
    keypoints = processed["keypoints"].cpu().numpy().astype(np.float32)
    scores = processed["scores"].cpu().numpy()
    descriptors = processed["descriptors"].cpu().numpy().astype(np.float32)

    keep = scores >= KEYPOINT_THRESHOLD
    keypoints, scores, descriptors = keypoints[keep], scores[keep], descriptors[keep]

    if len(scores) > MAX_KEYPOINTS:
        top = np.argsort(scores)[::-1][:MAX_KEYPOINTS]
        keypoints, descriptors = keypoints[top], descriptors[top]

    return keypoints, descriptors


def _match_descriptors(desc_a: np.ndarray, desc_b: np.ndarray):
    """Nearest-neighbour matching with Lowe's ratio test. Plain float32
    descriptors, so cv2's generic brute-force matcher (no OpenCV-specific
    feature type needed) does the work."""
    if len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    idx_a, idx_b = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < MATCH_RATIO * n.distance:
            idx_a.append(m.queryIdx)
            idx_b.append(m.trainIdx)
    return np.array(idx_a, dtype=np.int64), np.array(idx_b, dtype=np.int64)


def _estimate_motion(kp_prev, desc_prev, kp_cur, desc_cur, scale: float):
    """Estimate the (dx, dy, da) rigid motion from prev -> cur frame, scaled
    up to full-resolution pixel units. Returns None if there aren't enough
    trustworthy matches."""
    idx_prev, idx_cur = _match_descriptors(desc_prev, desc_cur)
    if len(idx_prev) < MIN_MATCHES:
        return None

    pts_prev = kp_prev[idx_prev]
    pts_cur = kp_cur[idx_cur]

    transform, inliers = cv2.estimateAffinePartial2D(
        pts_prev, pts_cur, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if transform is None or inliers is None or int(inliers.sum()) < MIN_MATCHES:
        return None

    dx = float(transform[0, 2]) / scale
    dy = float(transform[1, 2]) / scale
    da = float(np.arctan2(transform[1, 0], transform[0, 0]))
    return dx, dy, da


def _moving_average(curve: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return curve
    window = 2 * radius + 1
    kernel = np.ones(window) / window
    padded = np.pad(curve, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def stabilize_video(input_path: str, output_path: str | None = None) -> str:
    """
    Re-render `input_path` with camera shake removed, using a Hugging Face
    keypoint model to track features across frames.

    Returns the path to the stabilized video (a new temp file if
    `output_path` isn't given — the caller is responsible for deleting it).
    Raises StabilizationUnavailableError if the AI dependencies/model can't
    be loaded, or ValueError if the input isn't a readable video.
    """
    processor, model = _get_stabilizer()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open file as a video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        raise ValueError("Video is too short to stabilize (need at least 2 frames).")

    # Pass 1 — detect AI keypoints once per frame (cached, so each frame is
    # only run through the model once even though it's compared to both its
    # previous and next neighbor).
    keypoints_by_frame = []
    descriptors_by_frame = []
    scale = 1.0
    for frame in frames:
        small, scale = _resize_for_detection(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), DETECT_MAX_DIMENSION)
        try:
            kp, desc = _detect_keypoints(processor, model, small)
        except Exception:  # noqa: BLE001 - one bad frame must not abort the video
            kp, desc = np.empty((0, 2), dtype=np.float32), np.empty((0, 256), dtype=np.float32)
        keypoints_by_frame.append(kp)
        descriptors_by_frame.append(desc)

    # Pass 2 — estimate frame-to-frame motion from the cached keypoints.
    deltas = [(0.0, 0.0, 0.0)]
    for i in range(1, len(frames)):
        try:
            motion = _estimate_motion(
                keypoints_by_frame[i - 1], descriptors_by_frame[i - 1],
                keypoints_by_frame[i], descriptors_by_frame[i],
                scale,
            )
        except Exception:  # noqa: BLE001 - fall back to "no motion" for this pair
            motion = None
        deltas.append(motion if motion is not None else (0.0, 0.0, 0.0))

    # Pass 3 — integrate into a trajectory, then smooth it. The correction to
    # apply to frame i is the difference between the smooth trajectory and
    # the camera's actual (shaky) trajectory at that frame.
    trajectory = np.cumsum(np.array(deltas), axis=0)  # columns: x, y, angle
    smoothed = np.column_stack([
        _moving_average(trajectory[:, i], SMOOTHING_RADIUS) for i in range(3)
    ])
    correction = smoothed - trajectory

    # Pass 4 — warp each frame by its correction and crop in slightly to hide
    # the resulting border, then write the result out at the original size.
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Could not open '{output_path}' for writing the stabilized video.")

    crop_x = int(width * BORDER_CROP_RATIO)
    crop_y = int(height * BORDER_CROP_RATIO)

    try:
        for i, frame in enumerate(frames):
            dx, dy, da = correction[i]
            cos_a, sin_a = np.cos(da), np.sin(da)
            transform = np.array([
                [cos_a, -sin_a, dx],
                [sin_a, cos_a, dy],
            ], dtype=np.float32)
            warped = cv2.warpAffine(frame, transform, (width, height), borderMode=cv2.BORDER_REPLICATE)
            if crop_x or crop_y:
                cropped = warped[crop_y:height - crop_y, crop_x:width - crop_x]
                warped = cv2.resize(cropped, (width, height))
            writer.write(warped)
    finally:
        writer.release()

    return output_path
