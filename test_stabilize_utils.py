"""
test_stabilize_utils.py — offline test of the stabilization math.

Exercises the pieces of stabilize_utils.py that don't depend on downloading
the Hugging Face keypoint model itself (that lazy-loaded model is out of
scope here, same as test_pipeline.py stubs out the YOLO model): descriptor
matching, RANSAC motion estimation from matched points, and trajectory
smoothing.

Run: python3 test_stabilize_utils.py
"""

import numpy as np

import stabilize_utils as su


def test_moving_average_smooths_a_spike():
    curve = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    smoothed = su._moving_average(curve, radius=2)
    assert len(smoothed) == len(curve)
    # A lone spike should be spread out and reduced, not eliminated or grown.
    assert smoothed[3] < curve[3]
    assert smoothed[3] > 0
    assert smoothed[0] >= 0
    print("OK: moving average dampens a single-frame trajectory spike")


def test_moving_average_radius_zero_is_identity():
    curve = np.array([1.0, 5.0, -3.0, 2.0])
    assert np.array_equal(su._moving_average(curve, radius=0), curve)
    print("OK: radius=0 moving average is a no-op")


def _identity_descriptors(n):
    """n distinct one-hot descriptors, so nearest-neighbor matching between
    two such sets is unambiguous and identity-preserving."""
    return np.eye(n, dtype=np.float32)


def test_match_descriptors_pairs_by_identity():
    desc_a = _identity_descriptors(6)
    # Same identities, shuffled order in frame b - matching should still pair
    # each row in a with the row in b carrying the same identity.
    order = [3, 0, 4, 1, 5, 2]  # desc_b[k] == desc_a[order[k]]
    desc_b = desc_a[order]
    idx_a, idx_b = su._match_descriptors(desc_a, desc_b)
    assert len(idx_a) == 6, (idx_a, idx_b)
    for a, b in zip(idx_a, idx_b):
        assert order[b] == a, (a, b, order)
    print("OK: descriptor matching recovers identity regardless of frame order")


def test_match_descriptors_rejects_ambiguous_points():
    # The last row of desc_a is roughly equidistant from two candidates in
    # desc_b (ambiguous) - the ratio test should decline to guess for it,
    # while the clearly unambiguous first row still matches.
    desc_a = np.array([[1, 0], [0, 1], [0.71, 0.71]], dtype=np.float32)
    desc_b = np.array([[1, 0], [0.70, 0.70], [0.72, 0.72]], dtype=np.float32)
    idx_a, idx_b = su._match_descriptors(desc_a, desc_b)
    assert 0 in idx_a and idx_b[list(idx_a).index(0)] == 0
    assert len(idx_a) < 3
    print("OK: ratio test rejects ambiguous descriptor matches")


def test_estimate_motion_recovers_known_translation():
    rng = np.random.default_rng(0)
    kp_prev = rng.uniform(0, 400, size=(30, 2)).astype(np.float32)
    dx_true, dy_true = 12.0, -7.0
    kp_cur = kp_prev + np.array([dx_true, dy_true], dtype=np.float32)
    desc = _identity_descriptors(30)  # same descriptors both frames => 1:1 matches

    motion = su._estimate_motion(kp_prev, desc, kp_cur, desc, scale=1.0)
    assert motion is not None
    dx, dy, da = motion
    assert abs(dx - dx_true) < 0.5, motion
    assert abs(dy - dy_true) < 0.5, motion
    assert abs(da) < 0.01, motion
    print("OK: motion estimation recovers a known pure translation")


def test_estimate_motion_scales_translation_by_detection_scale():
    rng = np.random.default_rng(1)
    kp_prev = rng.uniform(0, 200, size=(20, 2)).astype(np.float32)
    dx_small, dy_small = 5.0, 5.0
    kp_cur = kp_prev + np.array([dx_small, dy_small], dtype=np.float32)
    desc = _identity_descriptors(20)

    scale = 0.5  # keypoints were detected on a half-resolution frame
    motion = su._estimate_motion(kp_prev, desc, kp_cur, desc, scale=scale)
    assert motion is not None
    dx, dy, _ = motion
    # Full-resolution motion should be the small-frame motion divided by scale.
    assert abs(dx - dx_small / scale) < 0.5, motion
    assert abs(dy - dy_small / scale) < 0.5, motion
    print("OK: motion is scaled back up to full-resolution pixel units")


def test_estimate_motion_returns_none_below_min_matches():
    kp_prev = np.array([[10.0, 10.0]], dtype=np.float32)
    kp_cur = np.array([[12.0, 11.0]], dtype=np.float32)
    desc = np.array([[1.0, 0.0]], dtype=np.float32)
    assert su._estimate_motion(kp_prev, desc, kp_cur, desc, scale=1.0) is None
    print("OK: too few matches yields no motion estimate rather than a guess")


def test_resize_for_detection_no_op_under_max_dim():
    frame = np.zeros((100, 150, 3), dtype=np.uint8)
    resized, scale = su._resize_for_detection(frame, max_dim=480)
    assert resized.shape == frame.shape
    assert scale == 1.0
    print("OK: frames already under the detection size are left alone")


def test_resize_for_detection_scales_down_large_frames():
    frame = np.zeros((480, 960, 3), dtype=np.uint8)
    resized, scale = su._resize_for_detection(frame, max_dim=480)
    assert max(resized.shape[:2]) == 480
    assert 0 < scale < 1.0
    print("OK: oversized frames are scaled down for faster keypoint detection")


def main():
    test_moving_average_smooths_a_spike()
    test_moving_average_radius_zero_is_identity()
    test_match_descriptors_pairs_by_identity()
    test_match_descriptors_rejects_ambiguous_points()
    test_estimate_motion_recovers_known_translation()
    test_estimate_motion_scales_translation_by_detection_scale()
    test_estimate_motion_returns_none_below_min_matches()
    test_resize_for_detection_no_op_under_max_dim()
    test_resize_for_detection_scales_down_large_frames()
    print("\nAll stabilize_utils tests passed.")


if __name__ == "__main__":
    main()
