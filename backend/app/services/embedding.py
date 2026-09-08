"""
Embedding service using FastEmbed (Dense & BM25 Sparse).

1. Dense: BAAI/bge-small-en-v1.5 (384 dimensions) for semantic search.
2. Sparse: Qdrant/bm25 (~10 MB) for exact keyword & technical term matching.

ONNX Runtime keeps models lightweight (~80 MB total) with low memory footprint,
running CPU-efficiently within free-tier resource bounds.
"""

import logging
from typing import Optional

import numpy as np
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.models import SparseVector

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level model references — initialized once during lifespan startup
_model: Optional[TextEmbedding] = None
_sparse_model: Optional[SparseTextEmbedding] = None

SPARSE_MODEL_NAME = "Qdrant/bm25"


# ── Dense Embeddings (BGE-small) ──────────────────────────────────────────

def init_model() -> TextEmbedding:
    """
    Load the FastEmbed dense model. Called once during FastAPI lifespan startup.

    The model is downloaded on first run and cached locally.
    Subsequent loads are fast (~1-2 seconds).
    """
    global _model
    logger.info(f"Loading dense embedding model: {settings.embedding_model}")
    _model = TextEmbedding(model_name=settings.embedding_model)
    logger.info("Dense embedding model loaded successfully")
    return _model


def get_model() -> TextEmbedding:
    """Get the loaded dense model, raising if not initialized."""
    if _model is None:
        return init_model()
    return _model


def is_model_loaded() -> bool:
    """Check if the dense embedding model has been loaded."""
    return _model is not None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts into dense vectors.
    """
    if not texts:
        return []
    model = get_model()
    embeddings = list(model.embed(texts))
    return [emb.tolist() for emb in embeddings]


def embed_query(text: str) -> list[float]:
    """
    Embed a single query text into a dense vector.
    """
    model = get_model()
    embeddings = list(model.query_embed(text))
    return embeddings[0].tolist()


# ── Sparse Embeddings (BM25) ──────────────────────────────────────────────

def init_sparse_model() -> SparseTextEmbedding:
    """
    Load the FastEmbed BM25 sparse model. Called once during lifespan startup.
    """
    global _sparse_model
    logger.info(f"Loading sparse BM25 embedding model: {SPARSE_MODEL_NAME}")
    _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    logger.info("Sparse BM25 embedding model loaded successfully")
    return _sparse_model


def get_sparse_model() -> SparseTextEmbedding:
    """Get the loaded sparse model, initializing if needed."""
    global _sparse_model
    if _sparse_model is None:
        return init_sparse_model()
    return _sparse_model


def is_sparse_model_loaded() -> bool:
    """Check if the sparse embedding model has been loaded."""
    return _sparse_model is not None


def embed_sparse_texts(texts: list[str]) -> list[SparseVector]:
    """
    Embed document texts into Qdrant SparseVector objects using BM25.
    """
    if not texts:
        return []
    model = get_sparse_model()
    embeddings = list(model.embed(texts))
    return [
        SparseVector(
            indices=emb.indices.tolist(),
            values=emb.values.tolist(),
        )
        for emb in embeddings
    ]


def embed_sparse_query(text: str) -> SparseVector:
    """
    Embed a query text into a Qdrant SparseVector using BM25.
    """
    model = get_sparse_model()
    embeddings = list(model.query_embed(text))
    emb = embeddings[0]
    return SparseVector(
        indices=emb.indices.tolist(),
        values=emb.values.tolist(),
    )
