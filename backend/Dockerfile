# DocTalk backend image. Built by .github/workflows/deploy.yml and run on EC2 by deploy/server/doctalk-deploy.sh.
# Same Python as CI (3.12), so the tests check what actually ships.
FROM python:3.12-slim

# Link the image to the GitHub repository (shown on GHCR)
LABEL org.opencontainers.image.source=https://github.com/YashMittal1503/RAG-PROJECT

# libmagic: file type detection. libgl1 + libglib2.0-0: OpenCV / RapidOCR. curl: the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Performance & resource configuration for 1 GB RAM / 2 vCPU instances ──
ENV OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    ORT_MAX_THREADS=2 \
    ENABLE_RERANKER=true \
    ENABLE_ONNX_ARENA=false \
    EMBED_BATCH_SIZE=8 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Models live in the image, not in /tmp: a restart doesn't re-download them from Hugging Face
    FASTEMBED_CACHE_PATH=/opt/models/fastembed \
    FLASHRANK_CACHE_DIR=/opt/models/flashrank \
    PORT=10000

# Install dependencies first (layer caching: code changes don't reinstall them)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download the embedding, sparse (BM25) and reranker models at build time.
# Keep the names in sync with app/config.py (embedding_model), embedding.py and reranker.py.
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
from flashrank import Ranker; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding('Qdrant/bm25'); \
Ranker(model_name='ms-marco-TinyBERT-L-2-v2', cache_dir='/opt/models/flashrank')"

# Copy application code (see .dockerignore for what is left out)
COPY . .

# Run as an unprivileged user (uid 1000). /app/data is a mounted volume on the server; the deploy
# script makes it writable for this uid.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /app/data \
    && chown -R app:app /app/data /opt/models
USER app

# Which commit this image was built from; reported by /api/health so the pipeline can confirm a deploy.
ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA}

# Docker marks the container unhealthy if the API stops answering. start-period covers model loading.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# Database migrations are NOT run here: the deploy script runs them once, before switching containers,
# so a failed migration aborts the deploy instead of crash-looping the live container.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
