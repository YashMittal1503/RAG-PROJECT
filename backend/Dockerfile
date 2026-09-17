FROM python:3.11-slim

# Install libmagic for python-magic file type detection
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Memory optimizations for 512 MB Render free-tier ─────────────────────
# Keep common numerical/native backends single-threaded. Multiple independent
# thread pools can otherwise add avoidable stack and working-memory overhead.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    MALLOC_TRIM_THRESHOLD_=65536

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render provides $PORT dynamically; default to 10000 (Render's default)
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
