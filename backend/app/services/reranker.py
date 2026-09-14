"""
Reranking service using FlashRank (ONNX-based Cross-Encoder).

Uses ms-marco-TinyBERT-L-2-v2 (~3.3 MB model) via FlashRank for high-speed,
low-memory cross-encoder reranking. Re-evaluates top candidate chunks from vector
search to significantly improve Context Precision.
"""

import logging
import time
from typing import Optional
from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

_ranker: Optional[Ranker] = None


def init_ranker(model_name: str = "ms-marco-TinyBERT-L-2-v2") -> Ranker:
    """Initialize and cache the FlashRank model."""
    global _ranker
    if _ranker is None:
        logger.info(f"Loading FlashRank reranker model: {model_name}")
        t0 = time.time()
        _ranker = Ranker(model_name=model_name)
        logger.info(f"FlashRank reranker loaded in {(time.time() - t0)*1000:.1f}ms")
    return _ranker


def get_ranker() -> Ranker:
    """Get the loaded ranker, initializing if necessary."""
    global _ranker
    if _ranker is None:
        return init_ranker()
    return _ranker


def is_ranker_loaded() -> bool:
    """Check if the ranker model is loaded."""
    return _ranker is not None


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
    min_score: float = 0.0001,
) -> list[dict]:
    """
    Rerank a list of retrieved chunks using FlashRank cross-encoder.

    Each chunk dict must have 'content' or 'text'.
    Returns the top_k reranked chunks with updated 'rerank_score' and re-sorted.
    If reranking fails or no chunks provided, falls back gracefully to original chunks.
    """
    if not chunks or len(chunks) <= 1:
        return chunks

    try:
        ranker = get_ranker()

        passages = []
        chunk_map = {}
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id") or str(i)
            text = chunk.get("content") or chunk.get("text") or ""
            passages.append({"id": str(chunk_id), "text": text})
            chunk_map[str(chunk_id)] = chunk

        t0 = time.time()
        rerank_req = RerankRequest(query=query, passages=passages)
        reranked_results = ranker.rerank(rerank_req)
        elapsed_ms = (time.time() - t0) * 1000

        # Map back to original chunk objects with updated rerank_score
        reranked_chunks = []
        for item in reranked_results:
            cid = str(item["id"])
            if cid in chunk_map:
                original_chunk = dict(chunk_map[cid])
                original_chunk["rerank_score"] = float(item["score"])
                original_chunk["vector_score"] = original_chunk.get("score", 0.0)
                # Keep score updated to rerank_score for downstream consistency
                original_chunk["score"] = float(item["score"])
                # Preserve candidates up to top_k so exploratory narrative queries are not starved
                if item["score"] >= min_score or len(reranked_chunks) < top_k:
                    reranked_chunks.append(original_chunk)

        result = reranked_chunks[:top_k]
        top_orig = result[0].get("vector_score", 0.0) if result else 0.0
        top_new = result[0]["score"] if result else 0.0
        logger.info(
            f"Reranked {len(chunks)} -> {len(result)} chunks in {elapsed_ms:.1f}ms "
            f"(top score: {top_new:.4f} vs vector: {top_orig:.4f})"
        )
        return result

    except Exception as e:
        logger.error(f"FlashRank reranking failed: {e}. Falling back to vector search order.")
        return chunks[:top_k]
