"""
Embedding service using FastEmbed (ONNX-based).

Uses BAAI/bge-small-en-v1.5 (384 dimensions) via Qdrant's FastEmbed library.
ONNX Runtime keeps the model lightweight (~70 MB) with ~100 MB RAM at runtime,
fitting comfortably within Render's 512 MB free tier.
"""

import logging
from typing import Optional

import numpy as np
from fastembed import TextEmbedding

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level model reference — initialized once via init_model()
_model: Optional[TextEmbedding] = None


def init_model() -> TextEmbedding:
    """
    Load the FastEmbed model. Called once during FastAPI lifespan startup.

    The model is downloaded on first run and cached locally.
    Subsequent loads are fast (~1-2 seconds).
    """
    global _model
    logger.info(f"Loading embedding model: {settings.embedding_model}")
    _model = TextEmbedding(model_name=settings.embedding_model)
    logger.info("Embedding model loaded successfully")
    return _model


def get_model() -> TextEmbedding:
    """Get the loaded model, raising if not initialized."""
    if _model is None:
        raise RuntimeError(
            "Embedding model not initialized. "
            "Call init_model() during app startup."
        )
    return _model


def is_model_loaded() -> bool:
    """Check if the embedding model has been loaded."""
    return _model is not None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of document texts.

    FastEmbed's embed() returns a generator of numpy arrays.
    We convert them to plain Python lists for JSON serialization
    and Qdrant compatibility.

    FastEmbed handles the BGE instruction prefix internally,
    so we don't need to prepend anything manually.
    """
    model = get_model()
    # embed() returns a generator — we must consume it into a list
    embeddings = list(model.embed(texts))
    return [emb.tolist() for emb in embeddings]


def embed_query(text: str) -> list[float]:
    """
    Embed a single query text.

    Uses query_embed() which applies the appropriate BGE query prefix
    internally ("Represent this sentence for searching relevant passages:").
    """
    model = get_model()
    # query_embed() also returns a generator
    embeddings = list(model.query_embed(text))
    return embeddings[0].tolist()
