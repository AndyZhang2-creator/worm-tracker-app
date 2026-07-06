"""
speed_utils.py — the core speed-from-keypoints algorithm.

This module is deliberately kept isolated from FastAPI / OpenCV / ultralytics so
it can be unit-tested in isolation. It contains the ONE piece of logic in this
app that is correctness-critical: turning a per-frame keypoint track into speed
numbers while correctly handling low-confidence (untracked) frames.

The algorithm here has already been validated against synthetic ground-truth
data (known displacement per frame, recovered exactly). Do NOT redesign it.
"""

import numpy as np


def compute_track_speed(xs, ys, confs, fps, pcutoff=0.5, px_per_mm=None):
    """
    xs, ys, confs: 1D arrays, one value per analyzed frame, for ONE point
        (here: the worm's center).
    fps: the EFFECTIVE sampling fps actually used between consecutive analyzed
        frames (native_video_fps / frame_step), not necessarily the video's
        native fps. Speed is reported per SECOND, so this converts per-frame
        displacement into a rate.
    pcutoff: frames below this confidence become NaN (untracked), not
        zero displacement. This is the single most important correctness
        rule in this whole app: a low-confidence frame must NEVER be
        treated as "the worm didn't move." It must create a gap that's
        excluded from the average, not a fake zero or a fake jump.

    Why NaN-gaps matter:
        If we instead zero-filled or carried-forward a missed frame, a single
        bad detection would either (a) read as the worm teleporting back to its
        last known spot next frame, producing a fake huge speed spike, or
        (b) read as zero velocity, dragging the average down. Both corrupt the
        data. Skipping the interval entirely is the only correct behavior.

        Verified with this test case (frame 4 deliberately low-confidence, worm
        moving at a constant 5 px/frame, 10fps):
            distances:        [5, 5, 5, nan, nan, 5, 5, 5, 5]
            speeds (px/s):    [50, 50, 50, nan, nan, 50, 50, 50, 50]
            mean speed ignoring gaps: 50.0  <- exactly correct, unaffected by
                                              the bad frame.

        np.diff across a NaN naturally produces NaN on BOTH adjacent intervals
        (the interval entering the bad frame and the interval leaving it), and
        np.nanmean / np.nansum then exclude those intervals from the result.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    confs = np.asarray(confs, dtype=float)
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be positive to compute speed in px/s")

    valid = confs >= pcutoff
    # Untracked frames become NaN positions. Crucially NOT zeroed and NOT
    # forward-filled — see the docstring for why that distinction is the whole
    # point of this function.
    x = np.where(valid, xs, np.nan)
    y = np.where(valid, ys, np.nan)

    # Euclidean per-interval displacement in pixels. A NaN at either end of an
    # interval makes that interval's distance NaN, so it is excluded below.
    dist_px = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    speed_px_s = dist_px * fps

    out = {
        "frames_tracked_pct": float(100 * np.mean(valid)) if len(valid) else 0.0,
        "avg_speed_px_s": _safe_nanmean(speed_px_s),
        "max_speed_px_s": _safe_nanmax(speed_px_s),
        "total_distance_px": float(np.nansum(dist_px)) if len(dist_px) else 0.0,
        "speed_series_px_s": [None if np.isnan(v) else round(float(v), 3) for v in speed_px_s],
    }
    if px_per_mm:
        out["avg_speed_mm_s"] = out["avg_speed_px_s"] / px_per_mm if out["avg_speed_px_s"] is not None else None
        out["max_speed_mm_s"] = out["max_speed_px_s"] / px_per_mm if out["max_speed_px_s"] is not None else None
        out["total_distance_mm"] = out["total_distance_px"] / px_per_mm
    return out


def compute_displacement(xs, ys, confs, pcutoff=0.5, px_per_mm=None):
    """
    Displacement metrics for ONE keypoint track (here: the tail).

    farthest_displacement: the maximum straight-line distance the tail ever
        reaches from its FIRST tracked position — i.e. how far the tail got from
        where it started.
    net_displacement: straight-line distance from the first tracked position to
        the LAST tracked position (start -> endpoint), regardless of the path.

    Low-confidence frames are excluded (same NaN rule as compute_track_speed): a
    bad detection must never count as a real position. Displacement is measured
    relative to the first *tracked* frame, not frame 0, so a blurry opening
    frame doesn't anchor the whole measurement to a garbage point.

    Returns px values (and *_mm if px_per_mm given), plus the frame index of the
    farthest point so the UI can mark when it happened.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    confs = np.asarray(confs, dtype=float)

    valid = confs >= pcutoff
    x = np.where(valid, xs, np.nan)
    y = np.where(valid, ys, np.nan)
    idx = np.where(valid)[0]

    out = {
        "farthest_displacement_px": None,
        "net_displacement_px": None,
        "farthest_displacement_frame": None,
    }
    if len(idx) == 1:
        out["farthest_displacement_px"] = 0.0
        out["net_displacement_px"] = 0.0
        out["farthest_displacement_frame"] = int(idx[0])
    elif len(idx) >= 2:
        x0, y0 = x[idx[0]], y[idx[0]]
        dist_from_start = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)  # NaN where invalid
        far_i = int(np.nanargmax(dist_from_start))
        out["farthest_displacement_px"] = float(dist_from_start[far_i])
        out["farthest_displacement_frame"] = far_i
        xl, yl = x[idx[-1]], y[idx[-1]]
        out["net_displacement_px"] = float(np.sqrt((xl - x0) ** 2 + (yl - y0) ** 2))

    if px_per_mm:
        for key in ("farthest_displacement", "net_displacement"):
            v = out[key + "_px"]
            out[key + "_mm"] = (v / px_per_mm) if v is not None else None
    return out


def _safe_nanmean(arr):
    return None if len(arr) == 0 or np.all(np.isnan(arr)) else float(np.nanmean(arr))


def _safe_nanmax(arr):
    return None if len(arr) == 0 or np.all(np.isnan(arr)) else float(np.nanmax(arr))
