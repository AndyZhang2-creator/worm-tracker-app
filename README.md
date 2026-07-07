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

The app gets per-frame worm candidates from one of two interchangeable backends,
chosen by the `WORM_BACKEND` env var (auto-detected if unset):

| Backend | When used | Notes |
|---------|-----------|-------|
| `ultralytics` | default; or `WORM_BACKEND=ultralytics` | Local YOLO-pose weights at `model/best.pt`. The permanent backend. |
| `roboflow` | auto when `ROBOFLOW_API_KEY` is set; or `WORM_BACKEND=roboflow` | Hosted Roboflow workflow `andy-zhang-ud8qm/c-elegan-detection-v6-logic`. The app uses `InferenceHTTPClient.run_workflow` when the optional SDK is available, with an HTTP fallback for Python versions the SDK does not support. If Roboflow returns the known workflow `model_id` compile error or times out, the app falls back to direct model `c-elegan-detection-5haae/6`. |

To use the Roboflow workflow, set an API key in `.env` locally or as a Space
secret in production:

```bash
ROBOFLOW_API_KEY=...
WORM_BACKEND=roboflow
```

Other Roboflow knobs (sensible defaults already point at this project's model):

```bash
ROBOFLOW_WORKSPACE=andy-zhang-ud8qm
ROBOFLOW_WORKFLOW_ID=c-elegan-detection-v6-logic
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_CONFIDENCE=0.30      # minimum detection/keypoint confidence
ROBOFLOW_WORKFLOW_TIMEOUT_S=15
ROBOFLOW_MODEL_TIMEOUT_S=120
ROBOFLOW_FALLBACK_MODEL_ID=c-elegan-detection-5haae/6
WORM_PCUTOFF=0.30             # low-confidence keypoints become tracking gaps
WORM_MAX_TRACKS=5             # seed and track up to five worms per video
WORM_MAX_MATCH_DISTANCE_PX=150  # far detections become gaps instead of swaps
WORM_TRACKING_FRAME_MAX_WIDTH=900  # max width of the labeled preview image
WORM_LIVE_FRAME_MAX_WIDTH=720      # max width of the live detection preview
```

Notes:

- **Workflow first, direct model only on known compile error.** The app is
  configured for the requested v6 workflow. If Roboflow serverless returns the
  `InnerWorkflowParameterBindingsUnknownInputError` seen in older workflow
  versions, or if serverless times out, the app falls back to the underlying v6
  direct model with the same 30% confidence threshold so analysis still works.
- **Same worms across frames.** Roboflow can return multiple worm detections in
  one frame. The app seeds the five most confident worms from the first frame
  with detections, then associates later detections one-to-one by nearest center
  so each reported row follows the same worm identity. Distant unmatched
  detections become gaps instead of replacing one of the seeded worms.
- **Endpoints, not head/tail.** This model emits two keypoints named
  `End-point1` (id 0) and `End-Point-2` (id 1), not head/tail. The app reports
  each endpoint's displacement for reference, then tracks the worm center
  midpoint for speed and headline displacement. `probe_roboflow.py` dumps the
  raw response if you want to re-confirm the schema.
- One HTTP round-trip per sampled frame, so hosted workflow inference is slower
  than local weights.

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
   - **pcutoff** (default `0.3`) — keypoint confidence below this is treated as
     *untracked* (a gap), not as zero movement (see "Why gaps" below).
   - **px_per_mm** (optional) — leave blank to stay in pixels; set it to convert
     every speed/distance to millimetres.
3. Click **Analyze**. While the run is in progress, the page shows each sampled
   frame as it comes back from the model with stable `worm 1`, `worm 2`, etc.
   labels. Those labels are seeded once and then matched by nearest center, so a
   worm does not get renamed just because confidence order changes. When the run
   finishes, a summary banner shows the headline numbers, results appear in a
   table, and clicking any row shows the labeled tracking frame for that worm,
   draws that video's speed-over-time chart, and adds an amber marker at the
   moment the worm center was farthest from start.
4. After a batch run, **Download CSV** exports one row per worm/video.

### Metrics reported

Per tracked worm (and as CSV columns):

- **avg / max speed** — the average and peak center speed (px/s, or mm/s if
  calibrated). The average is the headline "how fast the worm is moving".
- **worm_id** — identity number seeded by confidence, up to five per video.
- **tracking frame** (`tracking_frame_image`) - a JPEG data URL of the seed
  frame with the tracked worms labeled. The selected row's worm is highlighted
  in the browser so the picture and the speed numbers refer to the same seeded
  identity.
- **farthest center** (`farthest_displacement`) — the maximum straight-line
  distance the tracked center ever reached from its **first tracked position**.
- **endpoint** (`net_displacement`) — straight-line distance from the first to
  the last tracked center position (start -> endpoint), regardless of path.
- **distance** (`total_distance`) — total path length the center travelled.

Batch summary banner:

- **average worm speed** — mean of the per-video average speeds.
- **farthest center moved** — the largest `farthest_displacement` across the
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
| POST   | `/api/analyze`         | Analyze one video; returns a `worms` array.|
| POST   | `/api/analyze-batch`   | Analyze many videos; returns flattened worm rows and a `csv_token`.|
| POST   | `/api/analyze-stream`  | Analyze many videos as NDJSON progress events, including live detection frames, then final results.|
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

### Auto-deploy from GitHub (recommended)

Instead of pushing to the Space by hand, let GitHub mirror `master` to the
Space on every change. The workflow at
`.github/workflows/sync-to-huggingface.yml` does this — merging a PR to
`master` redeploys the site automatically. One-time setup in the GitHub repo
(**Settings → Secrets and variables → Actions**):

- Secret **`HF_TOKEN`** — a Hugging Face **write** token
  (<https://huggingface.co/settings/tokens>).
- Variable **`HF_SPACE_ID`** — your Space id, e.g. `your-user/worm-tracker-app`.

Then any push to `master` (or a manual run from the **Actions** tab) force-pushes
the repo to the Space's `main` branch and triggers a rebuild. The token lives
only in GitHub's encrypted secrets — never in the repo.
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
