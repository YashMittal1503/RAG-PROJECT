FROM python:3.11-slim

# Install libmagic for python-magic file type detection
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Memory optimizations for 512 MB Render free-tier ─────────────────────
# OMP_NUM_THREADS=1: ONNX Runtime defaults to N threads per core; cap to 1
#   saves ~24 MB of thread stack memory on multi-core instances.
# TOKENIZERS_PARALLELISM=false: Prevents HuggingFace tokenizers from forking.
# MALLOC_TRIM_THRESHOLD_: Aggressively return freed memory to OS.
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV MALLOC_TRIM_THRESHOLD_=65536

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render provides $PORT dynamically; default to 10000 (Render's default)
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
