"""
SVD-RAG: Efficient Tree-Organized Retrieval-Augmented Generation via Singular Value Decomposition.

Reference:
    "SVD-RAG: Efficient Tree-Organized Retrieval-Augmented Generation via Singular Value Decomposition"
    Zhihui Sun (arXiv:2607.10316, July 2026)

Key Principles:
1. Deterministic Extractive Summarization via Economy SVD on dense sentence/chunk embedding matrices:
   M = U * Σ * V^T
2. Adaptive Energy Thresholding (tau = 0.95):
   E(k) = sum_{j=1}^k (sigma_j^2) / sum_{j=1}^r (sigma_j^2) >= tau
   Automatically adapts the number of sentences to semantic complexity.
3. Sentence Energy Scoring:
   e_i = sum_{j=1}^k (sigma_j * U_{i,j})^2
4. Chronological reconstruction: preserves document narrative flow.
5. Recursive multi-level tree construction converging to a single Root summary node.
"""

import logging
import uuid
from typing import Any, Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from app.services.chunking import _split_into_sentences, count_tokens
from app.services import embedding

logger = logging.getLogger(__name__)


def compute_svd_extractive_summary(
    sentences: list[str],
    sentence_embeddings: np.ndarray,
    tau: float = 0.95,
    min_sentences: int = 2,
    max_sentences: int = 6,
) -> tuple[str, list[int], float]:
    """
    Compute an extractive summary from a list of sentences and their dense embeddings using SVD.

    Args:
        sentences: Raw sentence strings in chronological order.
        sentence_embeddings: 2D numpy array M in R^{N x D} of sentence embeddings.
        tau: Cumulative energy threshold (default: 0.95 = 95% variance preserved).
        min_sentences: Minimum sentences to extract.
        max_sentences: Maximum sentences to extract.

    Returns:
        (summary_text, selected_indices, retained_energy_ratio)
    """
    n_sentences = len(sentences)
    if n_sentences == 0:
        return "", [], 0.0

    if n_sentences <= min_sentences:
        return " ".join(s.strip() for s in sentences), list(range(n_sentences)), 1.0

    # Ensure float32 matrix
    M = np.asarray(sentence_embeddings, dtype=np.float32)
    if M.ndim != 2 or M.shape[0] != n_sentences:
        raise ValueError(f"Expected embedding matrix shape ({n_sentences}, D), got {M.shape}")

    # Economy SVD: M = U * diag(s) * Vt
    try:
        U, s, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError as e:
        logger.warning(f"SVD decomposition failed ({e}), falling back to lead sentences")
        selected = list(range(min(min_sentences, n_sentences)))
        return " ".join(sentences[i].strip() for i in selected), selected, 1.0

    r = len(s)
    s_squared = s ** 2
    total_energy = float(np.sum(s_squared))

    if total_energy <= 1e-9:
        selected = list(range(min(min_sentences, n_sentences)))
        return " ".join(sentences[i].strip() for i in selected), selected, 1.0

    # Cumulative energy ratio E(k)
    cumulative_energy = np.cumsum(s_squared) / total_energy

    # Find smallest k such that E(k) >= tau
    k_candidates = np.where(cumulative_energy >= tau)[0]
    if len(k_candidates) > 0:
        k = int(k_candidates[0]) + 1
    else:
        k = r

    # Clamp k between min_sentences and max_sentences
    k = max(min_sentences, min(k, max_sentences, n_sentences))

    # Sentence energy projection: e_i = sum_{j=1}^k (sigma_j * U_{i, j})^2
    scaled_U = U[:, :k] * s[:k]
    sentence_energies = np.sum(scaled_U ** 2, axis=1)

    # Top-k sentences with highest energy contribution
    top_indices = np.argsort(sentence_energies)[::-1][:k]

    # Re-sort chronologically to preserve narrative coherence
    chronological_indices = sorted(top_indices.tolist())
    retained_ratio = float(cumulative_energy[min(k - 1, r - 1)])

    summary_text = " ".join(sentences[idx].strip() for idx in chronological_indices)
    return summary_text, chronological_indices, retained_ratio


def cluster_embeddings(
    embeddings: np.ndarray,
    target_cluster_size: int = 6,
) -> list[list[int]]:
    """
    Cluster embedding vectors using hierarchical agglomerative clustering with cosine distance.

    Args:
        embeddings: 2D numpy array in R^{N x D}.
        target_cluster_size: Desired average number of items per cluster.

    Returns:
        List of clusters, where each cluster is a list of original item indices.
    """
    n_items = len(embeddings)
    if n_items <= target_cluster_size:
        return [list(range(n_items))]

    # Target number of clusters
    k_clusters = max(2, int(np.ceil(n_items / target_cluster_size)))

    # Normalize vectors for cosine metric
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_embeddings = embeddings / norms

    # Pairwise cosine distance
    dists = pdist(norm_embeddings, metric="cosine")
    dists = np.nan_to_num(dists, nan=0.0)
    dists = np.clip(dists, 0.0, 2.0)

    # Hierarchical linkage (average linkage over cosine distance)
    Z = linkage(dists, method="average")
    labels = fcluster(Z, t=k_clusters, criterion="maxclust")

    # Group original indices by cluster label
    cluster_dict: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        cluster_dict.setdefault(label, []).append(idx)

    # Sort clusters chronologically by their earliest member index to keep structure stable
    clusters = list(cluster_dict.values())
    clusters.sort(key=lambda c: min(c))
    return clusters


def summarize_cluster_nodes(
    cluster_nodes: list[dict],
    tau: float = 0.95,
    min_sentences: int = 2,
    max_sentences: int = 6,
) -> tuple[str, float]:
    """
    Extract sentences across all nodes in a cluster, embed them, and compute an SVD summary.
    """
    all_sentences: list[str] = []
    for node in cluster_nodes:
        content = node.get("content", "")
        s_list = _split_into_sentences(content)
        all_sentences.extend(s_list)

    if not all_sentences:
        return "", 0.0

    if len(all_sentences) <= min_sentences:
        return " ".join(s.strip() for s in all_sentences), 1.0

    # Embed cluster sentences with dense model
    sent_vectors = embedding.embed_texts(all_sentences)
    sent_mat = np.array(sent_vectors, dtype=np.float32)

    summary_text, _, energy_ratio = compute_svd_extractive_summary(
        sentences=all_sentences,
        sentence_embeddings=sent_mat,
        tau=tau,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    return summary_text, energy_ratio


def build_recursive_svd_tree(
    leaf_chunks: list[dict],
    leaf_embeddings: list[list[float]],
    cluster_size: int = 6,
    tau: float = 0.95,
    doc_id: str | None = None,
    filename: str = "",
) -> tuple[list[dict], dict]:
    """
    Recursively constructs an SVD hierarchical tree from Level 0 leaves up to a single Root node.

    Args:
        leaf_chunks: List of leaf chunk dicts (Level 0).
        leaf_embeddings: Corresponding dense vectors in R^{N x D}.
        cluster_size: Number of chunks per cluster.
        tau: Energy ratio threshold for SVD summarization.
        doc_id: Associated document UUID.
        filename: Document filename.

    Returns:
        (all_summary_nodes, tree_metadata)
        - all_summary_nodes: List of generated summary node dicts ready for Qdrant/Postgres storage.
        - tree_metadata: Diagnostic stats (depth, total_summaries, root_id, energy_ratios).
    """
    n_leaves = len(leaf_chunks)
    if n_leaves < 2:
        # Single chunk documents do not need hierarchical trees
        return [], {"depth": 0, "total_summaries": 0, "root_id": None}

    all_summary_nodes: list[dict] = []
    current_level_nodes = leaf_chunks
    current_level_embeddings = np.array(leaf_embeddings, dtype=np.float32)
    level = 1

    energy_ratios: list[float] = []

    while True:
        n_current = len(current_level_nodes)
        if n_current <= 1:
            # Single node reached, nothing more to summarize
            break

        # If current level has <= cluster_size, all of them form 1 final root cluster
        if n_current <= cluster_size:
            clusters = [list(range(n_current))]
        else:
            clusters = cluster_embeddings(current_level_embeddings, target_cluster_size=cluster_size)

        next_level_nodes: list[dict] = []
        is_final_level = (len(clusters) == 1)

        for cluster_idx, member_indices in enumerate(clusters):
            cluster_members = [current_level_nodes[i] for i in member_indices]
            child_ids = [str(m["id"]) for m in cluster_members]

            # Collect representative page number
            pages = [
                m.get("page_number")
                for m in cluster_members
                if m.get("page_number") is not None
            ]
            rep_page = min(pages) if pages else None

            # SVD summarization over cluster sentences
            summary_text, energy_ratio = summarize_cluster_nodes(
                cluster_members,
                tau=tau,
                min_sentences=2,
                max_sentences=6 if not is_final_level else 8,
            )
            energy_ratios.append(energy_ratio)

            summary_id = str(uuid.uuid4())
            is_root_node = bool(is_final_level)

            node_dict = {
                "id": summary_id,
                "content": summary_text,
                "chunk_type": "summary",
                "tree_level": level,
                "is_root": is_root_node,
                "child_chunk_ids": child_ids,
                "page_number": rep_page,
                "row_range_start": None,
                "row_range_end": None,
                "token_count": count_tokens(summary_text),
                "doc_id": doc_id,
                "filename": filename,
            }

            next_level_nodes.append(node_dict)
            all_summary_nodes.append(node_dict)

        if is_final_level:
            # Reached the single root node!
            break

        # Generate dense embeddings for the next level nodes
        next_texts = [n["content"] for n in next_level_nodes]
        next_embeddings = embedding.embed_texts(next_texts)
        current_level_nodes = next_level_nodes
        current_level_embeddings = np.array(next_embeddings, dtype=np.float32)
        level += 1

    # Finally, compute dense and sparse BM25 vectors for ALL generated summary nodes
    if all_summary_nodes:
        summary_texts = [n["content"] for n in all_summary_nodes]
        dense_vecs = embedding.embed_texts(summary_texts)
        sparse_vecs = embedding.embed_sparse_texts(summary_texts)

        for node, d_vec, s_vec in zip(all_summary_nodes, dense_vecs, sparse_vecs):
            node["vector"] = d_vec
            node["sparse_vector"] = s_vec

    root_node = next((n for n in all_summary_nodes if n.get("is_root")), None)
    tree_metadata = {
        "depth": level,
        "total_summaries": len(all_summary_nodes),
        "root_id": root_node["id"] if root_node else None,
        "root_preview": (root_node["content"][:150] + "...") if root_node else "",
        "avg_energy_ratio": float(np.mean(energy_ratios)) if energy_ratios else 1.0,
    }

    logger.info(
        f"SVD Tree built for {filename}: {len(leaf_chunks)} leaves -> "
        f"{len(all_summary_nodes)} summaries across {level} level(s), root={tree_metadata['root_id']}"
    )

    return all_summary_nodes, tree_metadata
