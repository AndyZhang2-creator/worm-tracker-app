"""
test_pipeline.py — offline integration test of everything EXCEPT the real model.

We can't ship the trained YOLO weights, so this stubs `app.get_model` with a
fake detector that finds the brightest blob's centroid and reports it as the
tail keypoint. That lets us exercise the parts that don't depend on the model's
brains: OpenCV decode + sampling, per-video native-fps handling, the speed
algorithm end-to-end, batch error isolation, CSV building, and mm calibration.

Run:  ./.venv/Scripts/python.exe test_pipeline.py
"""

import os
import tempfile

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

    r30 = app.analyze_video_file(v30, sample_fps=10.0, pcutoff=0.5, px_per_mm=None)
    r15 = app.analyze_video_file(v15, sample_fps=10.0, pcutoff=0.5, px_per_mm=None)

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
    print(f"OK: tail displacement — farthest {r30['farthest_displacement_px']}px @ "
          f"{r30['farthest_displacement_time_s']}s, net {r30['net_displacement_px']}px")

    # Calibration switches units.
    rc = app.analyze_video_file(v30, sample_fps=10.0, pcutoff=0.5, px_per_mm=10.0)
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
            res = app.analyze_video_file(p, 10.0, 0.5, None)
            results.append(res)
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
    print(f"OK: batch summary — {summary['videos_ok']} worms, avg "
          f"{summary['avg_speed']} px/s, farthest {summary['farthest_displacement']}px "
          f"in {summary['farthest_displacement_video']}")

    # CSV building (pixels + mm variants), now including displacement columns.
    csv_px = app._build_csv(results, used_calibration=False)
    csv_mm = app._build_csv([rc], used_calibration=True)
    assert csv_px.splitlines()[0] == ",".join(app.CSV_BASE_COLUMNS)
    assert "farthest_displacement_px" in csv_px.splitlines()[0]
    assert "broken.mp4" in csv_px
    assert csv_mm.splitlines()[0].endswith("net_displacement_mm")
    print("OK: CSV header + rows correct (px and mm variants, incl. displacement)")
    print("\nCSV preview (pixels):")
    print(csv_px.strip())

    # ---- Roboflow workflow parser (offline, mocked response) ---------------- #
    test_roboflow_parser()

    print("\nALL OFFLINE PIPELINE TESTS PASSED")


def test_roboflow_parser():
    """Validate tail extraction from a realistic Roboflow workflow response."""
    # Two detections; the more confident one's tail keypoint should win. Nested
    # under an arbitrary output key to exercise the recursive finder.
    resp = [
        {
            "model_predictions": {
                "image": {"width": 640, "height": 480},
                "predictions": [
                    {
                        "x": 100, "y": 100, "width": 50, "height": 20,
                        "confidence": 0.40, "class": "worm",
                        "keypoints": [
                            {"x": 90, "y": 100, "confidence": 0.9, "class_name": "head"},
                            {"x": 110, "y": 100, "confidence": 0.8, "class_name": "tail"},
                        ],
                    },
                    {
                        "x": 300, "y": 200, "width": 50, "height": 20,
                        "confidence": 0.95, "class": "worm",
                        "keypoints": [
                            {"x": 290, "y": 200, "confidence": 0.92, "class_name": "head"},
                            {"x": 315, "y": 205, "confidence": 0.88, "class_name": "tail"},
                        ],
                    },
                ],
            }
        }
    ]
    tail = app._parse_roboflow_tail(resp)
    assert tail is not None, "parser returned None"
    x, y, conf = tail
    # Should pick the 0.95-confidence detection's tail keypoint (315, 205).
    assert (x, y) == (315.0, 205.0), tail
    assert abs(conf - 0.88) < 1e-6, tail

    # Fallback: no class_name -> use index 1 (tail) of the keypoints list.
    resp2 = {"out": {"predictions": [
        {"x": 1, "y": 1, "confidence": 0.7,
         "keypoints": [{"x": 5, "y": 5}, {"x": 9, "y": 7}]},
    ]}}
    assert app._parse_roboflow_tail(resp2) == (9.0, 7.0, 1.0)

    # No detections -> None (becomes a tracking gap, never a fake point).
    assert app._parse_roboflow_tail([{"out": {"predictions": []}}]) is None
    print("OK: Roboflow parser — picks highest-confidence tail, name + index fallback, empty->None")


if __name__ == "__main__":
    main()
