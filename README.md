---
title: Worm Tail Tracker
emoji: 🪱
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Worm Tail Tracker

A local web app that takes *C. elegans* microscope videos (~20 s each, up to
~100 of them), runs a trained YOLO-pose model frame-by-frame to find the worm's
**head** and **tail**, and turns that into speed numbers — per video and
aggregated across a batch.

The pose model is **not** built here. It is trained separately (Roboflow
Keypoint Detection, class `worm`, 2 keypoints: `head` then `tail`) and exported
as standard Ultralytics weights (`best.pt`). This app is built around that
contract and runs fine before the weights exist — you'll just see a "model not
ready" banner until you drop them in.

## Install

```bash
pip install -r requirements.txt
```

## Detection backends

The app gets a per-frame tail `(x, y, confidence)` from one of two
interchangeable backends, chosen by the `WORM_BACKEND` env var (auto-detected if
unset):

| Backend | When used | Notes |
|---------|-----------|-------|
| `ultralytics` | default; or `WORM_BACKEND=ultralytics` | Local YOLO-pose weights at `model/best.pt`. The permanent backend. |
| `roboflow` | auto when `ROBOFLOW_API_KEY` is set; or `WORM_BACKEND=roboflow` | **Temporary** Roboflow Inference *workflow* over HTTP. Each frame is JPEG+base64-posted to `{api_url}/infer/workflows/{workspace}/{workflow_id}`. |

Roboflow backend env vars (the API key is **never** committed to the repo):

```bash
export ROBOFLOW_API_KEY=...                      # required
export ROBOFLOW_API_URL=http://localhost:9001    # default; or a hosted endpoint
export ROBOFLOW_WORKSPACE=andy-zhang-ud8qm        # default
export ROBOFLOW_WORKFLOW_ID=c-elegan-detection-v1-logic
export WORM_BACKEND=roboflow                       # force it explicitly
```

To run a local Roboflow inference server: `pip install inference && inference
server start` (listens on `:9001`). One HTTP round-trip per sampled frame, so it
is slower than local weights — fine for short clips, and it's only temporary.

> The hosted (serverless) version of the workflow currently fails to compile
> (`InnerWorkflowParameterBindingsUnknownInputError` on `model_id`). Until that's
> fixed in the Roboflow workflow editor, use a local inference server. Use
> `probe_roboflow.py` to dump the raw workflow response and confirm the keypoint
> schema the parser expects.

## Add your trained model

Export your trained Ultralytics YOLO-pose weights and place them at:

```
model/best.pt
```

Or point the app elsewhere with an environment variable:

```bash
export WORM_MODEL_PATH=/path/to/best.pt      # macOS / Linux
$env:WORM_MODEL_PATH = "C:\path\to\best.pt"  # Windows PowerShell
```

> **Keypoint order matters.** The app assumes **index 0 = head, index 1 = tail**,
> which must match the skeleton order you defined in the Roboflow project. If
> that order is reversed, head and tail will be swapped in every result.

## Run

```bash
uvicorn app:app --reload
```

Then open <http://localhost:8000>.

## How to use

1. Drag-and-drop (or click to browse) one or more video files.
2. Set options:
   - **sample_fps** (default `10`) — how many frames per second to actually run
     inference on. Higher = more accurate but slower; 10/sec is a reasonable
     default for a 20-second clip. The video's native fps is read per-video, so
     mixing clips with different frame rates in one batch is fine.
   - **pcutoff** (default `0.5`) — keypoint confidence below this is treated as
     *untracked* (a gap), not as zero movement (see "Why gaps" below).
   - **px_per_mm** (optional) — leave blank to stay in pixels; set it to convert
     every speed/distance to millimetres.
3. Click **Analyze**. A summary banner shows the headline numbers, results
   appear in a table, and clicking any row draws that video's speed-over-time
   chart (with an amber marker at the moment the tail was farthest from start).
4. After a batch run, **Download CSV** exports the summary table.

### Metrics reported

Per video (and as CSV columns):

- **avg / max speed** — the average and peak tail speed (px/s, or mm/s if
  calibrated). The average is the headline "how fast the worm is moving".
- **farthest tail** (`farthest_displacement`) — the maximum straight-line
  distance the tail ever reached from its **first tracked position**: how far
  the worm's tail got from where it started.
- **endpoint** (`net_displacement`) — straight-line distance from the first to
  the last tracked tail position (start → endpoint), regardless of path.
- **distance** (`total_distance`) — total path length the tail travelled.

Batch summary banner:

- **average worm speed** — mean of the per-video average speeds.
- **farthest tail moved** — the largest `farthest_displacement` across the
  batch, and which video it came from.

All displacement metrics use the same NaN-gap rule as speed: low-confidence
frames are excluded and measured relative to the first *tracked* frame, so a
blurry opening frame never anchors the measurement to a garbage point.

## Why low-confidence frames become gaps, not zeros

This is the single most important correctness rule in the app. A frame where
the model isn't confident about the tail is excluded from the speed average —
it is **not** treated as "the worm didn't move":

- Zero-filling a missed frame would drag the average speed down toward zero.
- Carrying the last position forward would make the *next* good frame look like
  the worm teleported, producing a fake speed spike.

Both corrupt the data. The only correct behavior is to skip the interval. See
`speed_utils.py` for the validated implementation and its synthetic test case.

## API

| Method | Path                   | Purpose                                    |
|--------|------------------------|--------------------------------------------|
| POST   | `/api/analyze`         | Analyze one video.                         |
| POST   | `/api/analyze-batch`   | Analyze many videos; returns a `csv_token`.|
| GET    | `/api/download/{token}`| Download the batch summary CSV.            |
| GET    | `/api/model-status`    | `{ "ready": true }` or a reason it isn't.  |

A request that needs the model but can't find/load it returns **503** with an
actionable message (where to put `best.pt`). One bad video in a batch returns an
`"error"` entry for that video and does not abort the rest of the batch.

## Deploy as a website (Hugging Face Spaces)

This repo is Spaces-ready: a `Dockerfile` plus the YAML header at the top of
this README (`sdk: docker`, `app_port: 7860`) are all a Docker Space needs.

1. Create a new Space at <https://huggingface.co/new-space> → SDK **Docker** →
   Blank. (CPU Basic, free, is fine; the site runs before the model exists.)
2. Push this repo to the Space's git remote:
   ```bash
   pip install -U huggingface_hub
   huggingface-cli login                       # paste a write token
   git remote add space https://huggingface.co/spaces/<user>/worm-tracker-app
   git push space master:main
   ```
   The Space builds the Docker image and serves the app at
   `https://<user>-worm-tracker-app.hf.space`.
3. Until you add the model, the site loads and shows the amber **"model not
   ready"** pill (every analyze call returns a clean 503). To enable inference,
   add the trained weights one of two ways:
   - **Commit them:** drop `best.pt` into `model/`, `git push space` again
     (use Git LFS for large files), or
   - **Persistent storage:** attach Spaces persistent storage, upload `best.pt`
     to `/data/`, and set the Space secret/variable
     `WORM_MODEL_PATH=/data/best.pt`.

Any other container host works too — `docker build -t worm-tracker . &&
docker run -p 7860:7860 worm-tracker` serves it locally on port 7860.

## Project layout

```
worm-tracker-app/
  app.py            # FastAPI app, all endpoints
  speed_utils.py    # the speed algorithm, isolated and testable
  requirements.txt
  model/
    best.pt         # NOT included — add after training
  static/
    index.html      # the entire frontend (inline <style>/<script>)
  README.md
```
