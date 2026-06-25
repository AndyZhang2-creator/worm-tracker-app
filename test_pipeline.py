"""
test_pipeline.py — offline integration test of everything EXCEPT the real model.

We can't ship the trained YOLO weights, so this stubs `app.get_model` with a
fake detector that finds the brightest blob's centroid and reports it as the
tail keypoint. That lets us exercise the parts that don't depend on the model's
brains: OpenCV decode + sampling, per-video native-fps handling, the speed
algorithm end-to-end, batch error isolation, CSV building, and mm calibration.

Run:  ./.venv/Scripts/python.exe test_pipeline.py
"""

import base64
import os
import tempfile

os.environ["WORM_BACKEND"] = "ultralytics"

import cv2
import numpy as np

import app


# --------------------------------------------------------------------------- #
# Fake model that mimics the Ultralytics Results contract (spec §2)
# --------------------------------------------------------------------------- #

class _Tensorish:
    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=float)
    def cpu(self):
        return self
    def numpy(self):
        return self._arr
    def __len__(self):
        return len(self._arr)
    def __getitem__(self, i):
        return _Tensorish(self._arr[i])


class _Keypoints:
    def __init__(self, xy, conf):
        self.xy = _Tensorish(xy)      # shape (n, 2, 2): [head, tail] per det
        self.conf = _Tensorish(conf)  # shape (n, 2)


class _Boxes:
    def __init__(self, conf):
        self.conf = _Tensorish(conf)


class _Result:
    def __init__(self, keypoints, boxes):
        self.keypoints = keypoints
        self.boxes = boxes


class FakeModel:
    """Reports the brightest pixel as the tail; head is a fixed offset."""
    def __call__(self, frame, verbose=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, maxval, _, maxloc = cv2.minMaxLoc(gray)
        if maxval < 40:  # nothing bright -> "worm not detected"
            kp = _Keypoints(np.zeros((0, 2, 2)), np.zeros((0, 2)))
            return [_Result(kp, _Boxes(np.zeros((0,))))]
        tx, ty = float(maxloc[0]), float(maxloc[1])
        xy = np.array([[[tx - 5, ty], [tx, ty]]])   # [head, tail]
        conf = np.array([[0.9, 0.9]])
        return [_Result(_Keypoints(xy, conf), _Boxes(np.array([0.95])))]


def make_video(path, native_fps, blob_step_px, n_frames=40, size=240):
    """A white blob moving blob_step_px per frame, written at native_fps."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, native_fps, (size, size))
    for i in range(n_frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        x = 20 + i * blob_step_px
        cv2.circle(frame, (int(x), 120), 6, (255, 255, 255), -1)
        vw.write(frame)
    vw.release()


def main():
    app._MODEL = FakeModel()  # stub the lazy loader's result

    tmp = tempfile.mkdtemp()
    # Two clips, DIFFERENT native fps, SAME px/frame motion (§10 fps check).
    v30 = os.path.join(tmp, "clip30fps.mp4")
    v15 = os.path.join(tmp, "clip15fps.mp4")
    make_video(v30, native_fps=30.0, blob_step_px=4)
    make_video(v15, native_fps=15.0, blob_step_px=4)

    r30_rows = app.analyze_video_file(v30, sample_fps=10.0, pcutoff=0.5, px_per_mm=None)
    r15_rows = app.analyze_video_file(v15, sample_fps=10.0, pcutoff=0.5, px_per_mm=None)
    assert len(r30_rows) == 1, r30_rows
    assert len(r15_rows) == 1, r15_rows
    r30 = r30_rows[0]
    r15 = r15_rows[0]

    print(f"30fps clip: native={r30['native_fps']} eff={r30['effective_fps']} "
          f"avg={r30['avg_speed_px_s']} px/s  tracked={r30['frames_tracked_pct']}%")
    print(f"15fps clip: native={r15['native_fps']} eff={r15['effective_fps']} "
          f"avg={r15['avg_speed_px_s']} px/s  tracked={r15['frames_tracked_pct']}%")

    # Per-video fps must NOT be assumed global. 30fps -> step 3 -> eff 10fps,
    # blob moves 4px/frame so 12px per sampled interval -> ~120 px/s.
    # 15fps -> step round(15/10)=2 wait... round(1.5)=2 -> eff 7.5fps,
    # 8px per sampled interval * 7.5 = 60 px/s. Different effective fps, and
    # the speeds reflect each clip's own fps rather than one shared value.
    assert abs(r30["effective_fps"] - 10.0) < 1e-6
    assert abs(r15["effective_fps"] - 7.5) < 1e-6
    assert abs(r30["avg_speed_px_s"] - 120.0) < 2.0, r30["avg_speed_px_s"]
    assert abs(r15["avg_speed_px_s"] - 60.0) < 2.0, r15["avg_speed_px_s"]
    print("OK: per-video native fps handled correctly (not skewed by a global fps)")

    # Displacement metrics. The blob moves in a straight horizontal line, so the
    # farthest reach from start == net (start->endpoint) == total path (~156px).
    assert abs(r30["farthest_displacement_px"] - 156.0) < 6.0, r30["farthest_displacement_px"]
    assert abs(r30["net_displacement_px"] - 156.0) < 6.0, r30["net_displacement_px"]
    assert r30["farthest_displacement_time_s"] is not None
    assert r30["per_second_speed"][0]["avg_speed_px_s"] == 120.0, r30["per_second_speed"]
    assert r30["per_second_speed"][1]["avg_speed_px_s"] == 120.0, r30["per_second_speed"]
    assert r30["per_second_speed"][1]["tracked_duration_s"] == 0.3, r30["per_second_speed"]
    assert len(r15["per_second_speed"]) == 3, r15["per_second_speed"]
    assert r15["per_second_speed"][2]["avg_speed_px_s"] == 60.0, r15["per_second_speed"]
    assert r30["tracking_frame_image"].startswith("data:image/jpeg;base64,"), r30.keys()
    assert r30["tracking_frame_index"] == 0
    assert r30["tracking_frame_analyzed_index"] == 0
    assert r30["tracking_frame_time_s"] == 0.0
    preview_bytes = base64.b64decode(r30["tracking_frame_image"].split(",", 1)[1])
    assert len(preview_bytes) > 1000
    print("OK: labeled tracking-frame preview returned with worm result")

    progress_events = []
    streamed_rows = app.analyze_video_file(
        v30,
        sample_fps=10.0,
        pcutoff=0.5,
        px_per_mm=None,
        progress_callback=progress_events.append,
    )
    event_types = [event["type"] for event in progress_events]
    assert event_types[0] == "video_start", event_types
    assert "frame" in event_types, event_types
    assert event_types[-1] == "video_done", event_types
    frame_event = next(event for event in progress_events if event["type"] == "frame")
    assert frame_event["image"].startswith("data:image/jpeg;base64,"), frame_event.keys()
    assert frame_event["detections"] == 1, frame_event
    assert frame_event["worm_ids"] == [1], frame_event
    assert streamed_rows[0]["avg_speed_px_s"] == r30["avg_speed_px_s"]
    print("OK: live progress events include annotated model-detection frames")
    # Both endpoints are reported for reference, but the tracked point is the
    # CENTER (midpoint of the two endpoints), and we measure how far it travels.
    assert len(r30["endpoints"]) == 2, r30["endpoints"]
    assert r30["tracked_point"] == "center", r30.get("tracked_point")
    print(f"OK: center-midpoint displacement — farthest {r30['farthest_displacement_px']}px @ "
          f"{r30['farthest_displacement_time_s']}s, net {r30['net_displacement_px']}px "
          f"(tracking {r30['tracked_endpoint_name']})")

    # Calibration switches units.
    rc = app.analyze_video_file(v30, sample_fps=10.0, pcutoff=0.5, px_per_mm=10.0)[0]
    assert abs(rc["avg_speed_mm_s"] - 12.0) < 0.3, rc["avg_speed_mm_s"]
    assert abs(rc["farthest_displacement_mm"] - 15.6) < 0.6, rc["farthest_displacement_mm"]
    print(f"OK: calibration px_per_mm=10 -> avg {rc['avg_speed_mm_s']} mm/s, "
          f"farthest {rc['farthest_displacement_mm']} mm")

    # Batch with a corrupt file in the middle: others survive, bad one errors.
    bad = os.path.join(tmp, "broken.mp4")
    with open(bad, "wb") as f:
        f.write(b"not a real video")
    results = []
    for p in (v30, bad, v15):
        try:
            results.extend(app.analyze_video_file(p, 10.0, 0.5, None))
        except Exception as exc:  # noqa: BLE001
            results.append({"video": os.path.basename(p), "error": str(exc)})
    assert "error" not in results[0] and "error" in results[1] and "error" not in results[2]
    print(f"OK: batch error isolation — bad file reported: {results[1]['error']!r}")

    # Batch summary — the headline answers the user asked for.
    summary = app._batch_summary(results, used_mm=False)
    assert summary["videos_ok"] == 2 and summary["videos_failed"] == 1
    # mean of the two per-video averages (~120 and ~60) -> ~90 px/s.
    assert abs(summary["avg_speed"] - 90.0) < 3.0, summary["avg_speed"]
    assert summary["farthest_displacement_video"] in ("clip30fps.mp4", "clip15fps.mp4")
    assert summary["farthest_displacement_worm_id"] == 1
    print(f"OK: batch summary — {summary['videos_ok']} worms, avg "
          f"{summary['avg_speed']} px/s, farthest {summary['farthest_displacement']}px "
          f"in {summary['farthest_displacement_video']}")

    # CSV building (pixels + mm variants), now including displacement columns.
    csv_px = app._build_csv(results, used_calibration=False)
    csv_mm = app._build_csv([rc], used_calibration=True)
    header_px = csv_px.splitlines()[0]
    header_mm = csv_mm.splitlines()[0]
    assert header_px.startswith(",".join(app.CSV_BASE_COLUMNS))
    assert "farthest_displacement_px" in csv_px.splitlines()[0]
    assert "second_0_1_avg_speed_px_s" in header_px
    assert "second_2_3_avg_speed_px_s" in header_px
    assert "broken.mp4" in csv_px
    assert "second_0_1_avg_speed_mm_s" in header_mm
    print("OK: CSV header + rows correct (px and mm variants, incl. displacement + per-second speeds)")
    print("\nCSV preview (pixels):")
    print(csv_px.strip())

    # ---- Roboflow workflow parser (offline, mocked response) ---------------- #
    test_roboflow_parser()
    test_overlap_dedupe()

    print("\nALL OFFLINE PIPELINE TESTS PASSED")


def test_overlap_dedupe():
    """Overlapping boxes collapse to the highest-confidence one; tol=1 disables."""
    c_hi = {"x": 100, "y": 100, "width": 40, "height": 40, "confidence": 0.9,
            "keypoints": [(90, 100, 0.9, "End-point1"), (110, 100, 0.9, "End-Point-2")]}
    c_dup = {"x": 102, "y": 101, "width": 40, "height": 40, "confidence": 0.5,
             "keypoints": [(92, 101, 0.5, "End-point1"), (112, 101, 0.5, "End-Point-2")]}
    c_far = {"x": 400, "y": 400, "width": 40, "height": 40, "confidence": 0.7,
             "keypoints": [(390, 400, 0.7, "End-point1"), (410, 400, 0.7, "End-Point-2")]}
    kept = app._dedupe_overlapping([c_hi, c_dup, c_far], 0.9)
    assert [c["confidence"] for c in kept] == [0.9, 0.7], kept
    assert len(app._dedupe_overlapping([c_hi, c_dup, c_far], 1.0)) == 3  # disabled
    print("OK: overlap dedup — keeps highest-confidence of overlapping boxes, tol=1 off")


def test_roboflow_parser():
    """Validate endpoint extraction against the model's REAL response schema."""
    # Real shape from c-elegan-detection-5haae/1: two endpoints named
    # "End-point1" (id 0) and "End-Point-2" (id 1), NOT head/tail. Two
    # detections; the more confident one wins. With default config (no "tail"
    # name, RF_TAIL_INDEX=1) we track index 1 = "End-Point-2".
    resp = {
        "inference_id": "x", "time": 0.1,
        "image": {"width": 1080, "height": 1920},
        "predictions": [
            {
                "x": 100, "y": 100, "confidence": 0.40, "class": "C Elegan", "class_id": 0,
                "keypoints": [
                    {"x": 90, "y": 100, "confidence": 0.9, "class_id": 0, "class": "End-point1"},
                    {"x": 110, "y": 100, "confidence": 0.8, "class_id": 1, "class": "End-Point-2"},
                ],
            },
            {
                "x": 300, "y": 200, "confidence": 0.95, "class": "C Elegan", "class_id": 0,
                "keypoints": [
                    {"x": 290, "y": 200, "confidence": 0.92, "class_id": 0, "class": "End-point1"},
                    {"x": 315, "y": 205, "confidence": 0.88, "class_id": 1, "class": "End-Point-2"},
                ],
            },
        ],
    }
    kps = app._parse_roboflow_keypoints(resp)
    assert kps is not None, "parser returned None"
    # ALL keypoints of the highest-confidence (0.95) detection, in order.
    assert kps == [
        (290.0, 200.0, 0.92, "End-point1"),
        (315.0, 205.0, 0.88, "End-Point-2"),
    ], kps

    # No detections -> None (becomes a tracking gap, never a fake point).
    assert app._parse_roboflow_keypoints({"predictions": []}) is None
    print("OK: Roboflow parser — returns all endpoints of the best detection")

    def cand(cx, conf, label):
        return {
            "x": cx,
            "y": 100,
            "confidence": conf,
            "class": label,
            "keypoints": [
                (cx - 5, 100.0, conf, "End-point1"),
                (cx + 5, 100.0, conf, "End-Point-2"),
            ],
        }

    same_worm_track = app._select_same_worm_track([
        [cand(100, 0.90, "seed-worm"), cand(300, 0.60, "other-worm")],
        [cand(112, 0.35, "seed-worm"), cand(300, 0.99, "other-worm")],
    ])
    assert same_worm_track[1][0][0] == 107.0, same_worm_track
    print("OK: same-worm tracking — nearest center beats per-frame confidence jump")

    live_tracker = app._LiveWormTracker(max_tracks=2)
    live_seed = live_tracker.update([cand(100, 0.90, "worm-a"), cand(200, 0.80, "worm-b")])
    live_later = live_tracker.update([cand(204, 0.99, "worm-b"), cand(104, 0.20, "worm-a")])
    assert [(round(app._candidate_center(c)[0]), worm_id) for c, worm_id in live_seed] == [(100, 1), (200, 2)]
    assert [(round(app._candidate_center(c)[0]), worm_id) for c, worm_id in live_later] == [(104, 1), (204, 2)]
    print("OK: live preview labels keep stable worm ids when confidence rank changes")

    seeded = [cand(100 + i * 40, 0.90 - i * 0.05, f"seed-{i}") for i in range(5)]
    lower_seed = cand(340, 0.10, "not-seeded")
    later = [cand(104 + i * 40, 0.40, f"seed-{i}") for i in range(5)]
    tempting_intruder = cand(340, 0.99, "not-seeded")
    five_tracks = app._select_same_worm_tracks([
        seeded + [lower_seed],
        later + [tempting_intruder],
    ], max_tracks=5)
    assert len(five_tracks) == 5, five_tracks
    assert [t["worm_id"] for t in five_tracks] == [1, 2, 3, 4, 5]
    assert [t["seed_confidence"] for t in five_tracks] == [0.9, 0.85, 0.8, 0.75, 0.7]
    assert [t["frames"][1][0][0] for t in five_tracks] == [99.0, 139.0, 179.0, 219.0, 259.0]
    no_jump_tracks = app._select_same_worm_tracks([
        [cand(100, 0.90, "seed")],
        [cand(400, 0.99, "far-intruder")],
    ], max_tracks=1)
    assert no_jump_tracks[0]["frames"][1] is None, no_jump_tracks
    print("OK: top-five worm tracking — seeded five identities ignore later intruder")

    # Endpoint selection: default picks the greatest-displacement endpoint;
    # an untracked (None) endpoint never wins over a tracked one.
    pe = [
        {"index": 0, "name": "End-point1", "farthest_displacement_px": 12.0, "net_displacement_px": 5.0},
        {"index": 1, "name": "End-Point-2", "farthest_displacement_px": 40.0, "net_displacement_px": 30.0},
    ]
    assert app._choose_endpoint(pe) == 1, app._choose_endpoint(pe)
    pe2 = [
        {"index": 0, "name": "a", "farthest_displacement_px": None, "net_displacement_px": None},
        {"index": 1, "name": "b", "farthest_displacement_px": 3.0, "net_displacement_px": 3.0},
    ]
    assert app._choose_endpoint(pe2) == 1
    print("OK: endpoint selection — greatest displacement wins, None never beats tracked")


if __name__ == "__main__":
    main()
