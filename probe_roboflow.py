"""
Probe the Roboflow workflow output schema, so we can lock the parser in
app.py (_parse_roboflow_tail) to the real response shape.

Usage (key comes from the environment, never hard-coded):
    ROBOFLOW_API_KEY=xxxx ./.venv/Scripts/python.exe probe_roboflow.py
    # against your local inference server instead of the hosted endpoint:
    ROBOFLOW_API_KEY=xxxx ROBOFLOW_API_URL=http://localhost:9001 python probe_roboflow.py
"""
import base64, json, os
from pathlib import Path
import cv2, numpy as np, httpx


def _load_local_env(path):
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env(Path(__file__).resolve().parent / ".env")

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
WORKSPACE = os.environ.get("ROBOFLOW_WORKSPACE", "andy-zhang-ud8qm")
WORKFLOW = os.environ.get("ROBOFLOW_WORKFLOW_ID", "c-elegan-detection-v6-logic")
FALLBACK_MODEL = os.environ.get("ROBOFLOW_FALLBACK_MODEL_ID", "c-elegan-detection-5haae/6")
CONFIDENCE = float(os.environ.get("ROBOFLOW_CONFIDENCE", "0.30"))
LOCAL_URL = os.environ.get("ROBOFLOW_API_URL")  # set to probe a local server

if not API_KEY:
    raise SystemExit("Set ROBOFLOW_API_KEY in the environment first.")

# Synthetic test frame (worm-ish bright squiggle) just to learn the envelope.
img = np.zeros((480, 480, 3), dtype=np.uint8)
cv2.line(img, (120, 240), (360, 250), (200, 200, 200), 6)
cv2.circle(img, (120, 240), 8, (255, 255, 255), -1)
ok, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf).decode("utf-8")

body = {
    "api_key": API_KEY,
    "inputs": {
        "image": {"type": "base64", "value": b64},
        "confidence": CONFIDENCE,
    },
    "use_cache": True,
}

bases = [LOCAL_URL] if LOCAL_URL else [
    "https://serverless.roboflow.com",
    "https://detect.roboflow.com",
]
for base in bases:
    base = base.rstrip("/")
    url = f"{base}/infer/workflows/{WORKSPACE}/{WORKFLOW}"
    try:
        r = httpx.post(url, json=body, timeout=60)
        print(f"\n=== {base}  HTTP {r.status_code} ===")
        if r.status_code != 200:
            print(r.text[:600])
            if "InnerWorkflowParameterBindingsUnknownInputError" not in r.text:
                continue
            fallback_url = f"{base}/{FALLBACK_MODEL}"
            rf = httpx.post(
                fallback_url,
                params={"api_key": API_KEY, "confidence": int(CONFIDENCE * 100)},
                content=b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=60,
            )
            print(f"\n=== fallback {fallback_url}  HTTP {rf.status_code} ===")
            print(rf.text[:3000])
            break
        data = r.json()
        print("TOP-LEVEL TYPE:", type(data).__name__)
        print(json.dumps(data, indent=2)[:3000])
        break
    except Exception as e:
        print(f"\n=== {base} ERROR: {type(e).__name__}: {e} ===")
