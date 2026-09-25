"""
Embedding service using FastEmbed (Dense & BM25 Sparse).

1. Dense: BAAI/bge-small-en-v1.5 (384 dimensions) for semantic search.
2. Sparse: Qdrant/bm25 (~10 MB) for exact keyword & technical term matching.

ONNX Runtime keeps models lightweight (~80 MB total) with low memory footprint,
running CPU-efficiently within free-tier resource bounds.
"""

import logging
import threading
from typing import Optional

import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.models import SparseVector

from app.config import settings
from app.utils.memory import release_memory

# ── FastEmbed ONNX Runtime Session Tuning ─────────────────────────────────
# On 1GB+ RAM instances, enable_cpu_mem_arena=True re-uses tensor allocation buffers,
# yielding a 3x-5x speedup over dynamic OS malloc/free calls.
from fastembed.common.onnx_model import OnnxModel

_orig_load_onnx_model = OnnxModel.load_onnx_model


def _bounded_load_onnx_model(
    self,
    model_dir,
    model_file,
    threads=None,
    providers=None,
):
    import onnxruntime as ort

    model_path = model_dir / model_file
    onnx_providers = ["CPUExecutionProvider"] if providers is None else list(providers)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_cpu_mem_arena = settings.enable_onnx_arena
    so.intra_op_num_threads = settings.embedding_threads
    so.inter_op_num_threads = settings.embedding_threads

    self.model = ort.InferenceSession(
        str(model_path), providers=onnx_providers, sess_options=so
    )


OnnxModel.load_onnx_model = _bounded_load_onnx_model

logger = logging.getLogger(__name__)

# Module-level model references — initialized lazily on first use
_model: Optional[TextEmbedding] = None
_sparse_model: Optional[SparseTextEmbedding] = None
_model_lock = threading.Lock()
_sparse_lock = threading.Lock()

SPARSE_MODEL_NAME = "Qdrant/bm25"


# ── Dense Embeddings (BGE-small) ──────────────────────────────────────────

def init_model() -> TextEmbedding:
    """
    Load the FastEmbed dense model using configured thread settings.
    """
    global _model
    logger.info(f"Loading dense embedding model: {settings.embedding_model} (threads={settings.embedding_threads})")
    _model = TextEmbedding(model_name=settings.embedding_model, threads=settings.embedding_threads)
    logger.info("Dense embedding model loaded successfully")
    return _model


def get_model() -> TextEmbedding:
    """Get the loaded dense model, initializing lazily on first use (thread-safe)."""
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            init_model()
    return _model


def is_model_loaded() -> bool:
    """Check if the dense embedding model has been loaded."""
    return _model is not None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts into dense vectors with configured batch size.
    """
    if not texts:
        return []
    model = get_model()
    embeddings = list(model.embed(texts, batch_size=settings.embed_batch_size, parallel=None))
    return [emb.tolist() for emb in embeddings]


def embed_query(text: str) -> list[float]:
    """
    Embed a single query text into a dense vector with batch_size=1.
    """
    model = get_model()
    embeddings = list(model.query_embed(text, batch_size=1))
    return embeddings[0].tolist()


# ── Sparse Embeddings (BM25) ──────────────────────────────────────────────

def init_sparse_model() -> SparseTextEmbedding:
    """
    Load the FastEmbed BM25 sparse model using configured thread settings.
    """
    global _sparse_model
    logger.info(f"Loading sparse BM25 embedding model: {SPARSE_MODEL_NAME} (threads={settings.embedding_threads})")
    _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME, threads=settings.embedding_threads)
    logger.info("Sparse BM25 embedding model loaded successfully")
    return _sparse_model


def get_sparse_model() -> SparseTextEmbedding:
    """Get the loaded sparse model, initializing lazily on first use (thread-safe)."""
    if _sparse_model is not None:
        return _sparse_model
    with _sparse_lock:
        if _sparse_model is None:
            init_sparse_model()
    return _sparse_model


def is_sparse_model_loaded() -> bool:
    """Check if the sparse embedding model has been loaded."""
    return _sparse_model is not None


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    """
    Embed document texts into Qdrant SparseVector objects using BM25 with configured batch size.
    """
    if not texts:
        return []
    model = get_sparse_model()
    embeddings = list(model.embed(texts, batch_size=settings.embed_batch_size, parallel=None))
    return [
        SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
        for emb in embeddings
    ]


def embed_sparse_query(text: str) -> SparseVector:
    """
    Embed a query text into a Qdrant SparseVector using BM25 with batch_size=1.
    """
    model = get_sparse_model()
    embeddings = list(model.query_embed(text, batch_size=1))
    emb = embeddings[0]
    return SparseVector(
        indices=emb.indices.tolist(),
        values=emb.values.tolist(),
    )

