"""
Reranking service using FlashRank (ONNX-based Cross-Encoder).

Uses ms-marco-TinyBERT-L-2-v2 (~3.3 MB model) via FlashRank for high-speed,
low-memory cross-encoder reranking. Re-evaluates top candidate chunks from vector
search to significantly improve Context Precision.
"""

import logging
import threading
import time
from typing import Optional
from flashrank import Ranker, RerankRequest

from app.config import settings
from app.utils.memory import release_memory

logger = logging.getLogger(__name__)

_ranker: Optional[Ranker] = None
_ranker_lock = threading.Lock()


def init_ranker(model_name: str = "ms-marco-TinyBERT-L-2-v2") -> Ranker:
    """
    Initialize and cache the FlashRank model with tuned ONNX session options.
    """
    global _ranker
    if _ranker is None:
        logger.info(f"Loading FlashRank reranker model: {model_name}")
        t0 = time.time()
        ranker = Ranker(model_name=model_name)

        # Optimize the underlying ONNX Runtime session
        try:
            import onnxruntime as ort
            from flashrank.Config import model_file_map

            if model_name in model_file_map and hasattr(ranker, "model_dir"):
                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                so.enable_cpu_mem_arena = settings.enable_onnx_arena
                so.intra_op_num_threads = settings.embedding_threads
                so.inter_op_num_threads = settings.embedding_threads
                model_path = str(ranker.model_dir / model_file_map[model_name])
                ranker.session = ort.InferenceSession(
                    model_path,
                    sess_options=so,
                    providers=["CPUExecutionProvider"],
                )
                logger.info(f"FlashRank ONNX session tuned (arena={settings.enable_onnx_arena}, threads={settings.embedding_threads})")
        except Exception as opt_err:
            logger.warning(f"Could not apply ONNX session tuning to FlashRank: {opt_err}")

        _ranker = ranker
        logger.info(f"FlashRank reranker loaded in {(time.time() - t0)*1000:.1f}ms")
    return _ranker


def get_ranker() -> Ranker:
    """Get the loaded ranker, initializing lazily on first use (thread-safe)."""
    if _ranker is not None:
        return _ranker
    with _ranker_lock:
        if _ranker is None:
            init_ranker()
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

    When settings.enable_reranker is False (default for 512MB memory environments),
    gracefully passes through Qdrant Cloud's server-side Hybrid RRF ranking,
    consuming 0 MB of container RAM.

    Each chunk dict must have 'content' or 'text'.
    Returns the top_k reranked chunks with updated 'rerank_score' and re-sorted.
    If reranking fails or no chunks provided, falls back gracefully to original chunks.
    """
    if not chunks or len(chunks) <= 1:
        return chunks

    # Zero-memory path for 512MB RAM containers: rely directly on Qdrant's Hybrid RRF
    if not settings.enable_reranker:
        logger.debug(
            f"Reranking skipped (settings.enable_reranker=False). "
            f"Using Qdrant Cloud Hybrid RRF ranking ({len(chunks[:top_k])} chunks)."
        )
        return chunks[:top_k]

    try:
        ranker = get_ranker()

        # Bound candidates to top_k * 2 (max 6) to avoid multi-chunk BERT cross-encoder spikes
        eval_chunks = chunks[: min(len(chunks), top_k * 2, 6)]

        passages = []
        chunk_map = {}
        for i, chunk in enumerate(eval_chunks):
            chunk_id = chunk.get("id") or str(i)
            # Truncate text to 350 chars (~70 words). Drastically minimizes BERT quadratic
            # attention memory overhead while preserving the core topical relevance.
            text = (chunk.get("content") or chunk.get("text") or "")[:350]
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
                original_chunk["score"] = float(item["score"])
                if item["score"] >= min_score or len(reranked_chunks) < top_k:
                    reranked_chunks.append(original_chunk)

        result = reranked_chunks[:top_k]
        top_orig = result[0].get("vector_score", 0.0) if result else 0.0
        top_new = result[0]["score"] if result else 0.0
        logger.info(
            f"Reranked {len(eval_chunks)} -> {len(result)} chunks in {elapsed_ms:.1f}ms "
            f"(top score: {top_new:.4f} vs vector: {top_orig:.4f})"
        )
        return result

    except Exception as e:
        logger.error(f"FlashRank reranking failed: {e}. Falling back to vector search order.")
        return chunks[:top_k]
    finally:
        release_memory()
