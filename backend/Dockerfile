FROM python:3.11-slim

# Link container image to the GitHub repository (required for GHCR)
LABEL org.opencontainers.image.source=https://github.com/YashMittal1503/RAG-PROJECT
# Install system dependencies: libmagic for file detection, libgl1 & libglib2.0-0 for OpenCV/RapidOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Performance & resource configuration for 1 GB RAM / 2 vCPU instances ──
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    ORT_MAX_THREADS=2 \
    ENABLE_RERANKER=true \
    ENABLE_ONNX_ARENA=true \
    TOKENIZERS_PARALLELISM=false

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render provides $PORT dynamically; default to 10000 (Render's default)
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
