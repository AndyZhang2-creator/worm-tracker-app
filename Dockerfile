# Dockerfile for Hugging Face Spaces (Docker SDK) and any container host.
# Spaces serves the container on port 7860.

FROM python:3.11-slim

# System libs: ffmpeg for robust video decoding, libglib for OpenCV runtime.
# (opencv-python-headless avoids the libGL/X11 stack, so we don't install it.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch FIRST so ultralytics/transformers don't drag in the
# multi-GB CUDA build. This keeps the Space image lean and the build within
# free limits. Both the YOLO detector and the Hugging Face AI stabilizer run
# on this same CPU torch install.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Where the app looks for trained weights. On Spaces you can either commit
# best.pt into model/ or attach persistent storage at /data and override this.
ENV WORM_MODEL_PATH=/app/model/best.pt

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
