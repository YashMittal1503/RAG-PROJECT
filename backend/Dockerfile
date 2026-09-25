FROM python:3.11-slim

# Install libmagic for python-magic file type detection
RUN apt-get update && apt-get install -y --no-install-recommends libmagic1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Performance & resource configuration for 1 GB RAM / 2 vCPU instances ──
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    ORT_MAX_THREADS=2 \
    ENABLE_RERANKER=true \
    TOKENIZERS_PARALLELISM=false \
    MALLOC_TRIM_THRESHOLD_=131072

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Render provides $PORT dynamically; default to 10000 (Render's default)
CMD alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
