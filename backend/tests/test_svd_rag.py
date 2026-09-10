"""
Unit tests for SVD-RAG (Singular Value Decomposition Tree-Organized RAG).

Verifies:
1. Economy SVD decomposition and energy thresholding (tau).
2. Sentence energy scoring and chronological sentence reconstruction.
3. Hierarchical agglomerative clustering on embedding matrices.
4. Multi-level recursive tree construction converging to a single Root summary node.
5. Flattened retrieval formatting and citation validation for multi-level summaries.
"""

import uuid
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.services.svd_summarizer import (
    compute_svd_extractive_summary,
    cluster_embeddings,
    build_recursive_svd_tree,
    summarize_cluster_nodes,
)
from app.services.query import _build_context, validate_citations


def test_svd_extractive_summary_basic():
    """Verify SVD decomposition extracts top sentences and preserves chronological order."""
    sentences = [
        "In 2024, artificial intelligence transformed retrieval-augmented generation systems.",
        "Traditional vector search often retrieves fragmented snippets lacking holistic context.",
        "Tree-organized architectures build hierarchical representations of long documents.",
        "RAPTOR uses large language models at internal tree nodes, causing high token costs.",
        "SVD-RAG replaces generative summaries with deterministic singular value decomposition.",
        "Singular value decomposition captures the principal semantic variance of sentence matrices.",
        "Benchmarking demonstrates significant latency reductions while retaining high recall.",
    ]

    # Create synthetic deterministic embeddings (7 sentences, 384 dimensions)
    rng = np.random.default_rng(42)
    # Give sentences 0, 4, 5 higher magnitude along principal axes
    embeddings = rng.normal(0, 0.1, size=(len(sentences), 384)).astype(np.float32)
    embeddings[0, :50] += 1.5
    embeddings[4, :50] += 2.0
    embeddings[5, :50] += 1.8

    summary, indices, ratio = compute_svd_extractive_summary(
        sentences=sentences,
        sentence_embeddings=embeddings,
        tau=0.95,
        min_sentences=2,
        max_sentences=4,
    )

    assert len(indices) >= 2
    assert len(indices) <= 4
    # Check chronological ordering: indices must be strictly increasing
    assert indices == sorted(indices)
    assert ratio > 0.0
    assert len(summary) > 0
    # The summary should be a joined string of the selected sentences
    for idx in indices:
        assert sentences[idx] in summary


def test_svd_tau_energy_threshold_adaptation():
    """Verify that higher tau preserves higher cumulative energy variance."""
    sentences = [f"This is test sentence number {i} with detailed information." for i in range(10)]
    rng = np.random.default_rng(123)
    embeddings = rng.normal(0, 1.0, size=(10, 384)).astype(np.float32)

    _, indices_low_tau, ratio_low = compute_svd_extractive_summary(
        sentences, embeddings, tau=0.50, min_sentences=1, max_sentences=8
    )
    _, indices_high_tau, ratio_high = compute_svd_extractive_summary(
        sentences, embeddings, tau=0.95, min_sentences=1, max_sentences=8
    )

    assert len(indices_high_tau) >= len(indices_low_tau)
    assert ratio_high >= ratio_low


def test_svd_edge_cases():
    """Verify edge cases: empty sentences, single sentence, identical zero vectors."""
    # Empty
    summary, idxs, r = compute_svd_extractive_summary([], np.zeros((0, 384), dtype=np.float32))
    assert summary == ""
    assert idxs == []
    assert r == 0.0

    # Below min_sentences
    s_single = ["Just one sentence."]
    e_single = np.ones((1, 384), dtype=np.float32)
    summary, idxs, r = compute_svd_extractive_summary(s_single, e_single, min_sentences=2)
    assert summary == "Just one sentence."
    assert idxs == [0]

    # Zero energy matrix
    s_zeros = ["Sentence A.", "Sentence B.", "Sentence C."]
    e_zeros = np.zeros((3, 384), dtype=np.float32)
    summary, idxs, r = compute_svd_extractive_summary(s_zeros, e_zeros, min_sentences=2)
    assert len(idxs) == 2


def test_cluster_embeddings_partitioning():
    """Verify hierarchical clustering cleanly groups embeddings with target size."""
    rng = np.random.default_rng(99)
    # 18 items with 3 distinct clusters in 384-dim space
    c1 = rng.normal(0.5, 0.05, size=(6, 384))
    c2 = rng.normal(-0.5, 0.05, size=(6, 384))
    c3 = rng.normal(0.0, 0.05, size=(6, 384))
    embeddings = np.vstack([c1, c2, c3]).astype(np.float32)

    clusters = cluster_embeddings(embeddings, target_cluster_size=6)

    # All items 0..17 must appear in exactly one cluster
    all_assigned = [item for cl in clusters for item in cl]
    assert sorted(all_assigned) == list(range(18))
    assert len(clusters) >= 2


def test_build_recursive_svd_tree_to_root():
    """Verify recursive tree construction builds multi-level tree converging to a single Root node."""
    n_leaves = 18
    fake_leaves = [
        {
            "id": str(uuid.uuid4()),
            "content": f"Document chapter {i // 3 + 1}. Detail paragraph {i}. Here is factual content regarding topic {i % 4}.",
            "chunk_index": i,
            "page_number": (i // 3) + 1,
            "row_range_start": None,
            "row_range_end": None,
            "token_count": 30,
        }
        for i in range(n_leaves)
    ]

    rng = np.random.default_rng(42)
    fake_embeddings = rng.normal(0, 0.5, size=(n_leaves, 384)).tolist()

    # Mock embedding calls for newly created summary nodes
    def mock_embed_texts(texts):
        return [rng.normal(0, 0.5, size=384).tolist() for _ in texts]

    def mock_embed_sparse(texts):
        return [{"indices": [1, 2], "values": [0.5, 0.8]} for _ in texts]

    with patch("app.services.embedding.embed_texts", side_effect=mock_embed_texts), \
         patch("app.services.embedding.embed_sparse_texts", side_effect=mock_embed_sparse):

        summary_nodes, tree_metadata = build_recursive_svd_tree(
            leaf_chunks=fake_leaves,
            leaf_embeddings=fake_embeddings,
            cluster_size=6,
            tau=0.95,
            doc_id="test-doc-123",
            filename="sample_paper.pdf",
        )

    assert len(summary_nodes) > 0
    assert tree_metadata["depth"] >= 2
    assert tree_metadata["root_id"] is not None

    # Check that exactly ONE node is marked is_root=True
    root_nodes = [n for n in summary_nodes if n.get("is_root")]
    assert len(root_nodes) == 1
    root = root_nodes[0]
    assert root["tree_level"] == tree_metadata["depth"]
    assert root["id"] == tree_metadata["root_id"]

    # Verify all summary nodes have valid vectors and metadata
    for node in summary_nodes:
        assert node["chunk_type"] == "summary"
        assert "vector" in node
        assert "sparse_vector" in node
        assert len(node["child_chunk_ids"]) > 0
        assert node["filename"] == "sample_paper.pdf"


def test_build_context_labels_root_and_level_summaries():
    """Verify _build_context creates [Root Summary] and [Summary LX] citation headers."""
    chunks = [
        {
            "id": "leaf-1",
            "chunk_type": "text",
            "page_number": 3,
            "tree_level": 0,
            "is_root": False,
            "filename": "doc.pdf",
            "content": "Leaf chunk content on page 3.",
        },
        {
            "id": "sum-l1",
            "chunk_type": "summary",
            "page_number": 2,
            "tree_level": 1,
            "is_root": False,
            "filename": "doc.pdf",
            "content": "Cluster summary content covering pages 1-4.",
        },
        {
            "id": "sum-root",
            "chunk_type": "summary",
            "page_number": 1,
            "tree_level": 2,
            "is_root": True,
            "filename": "doc.pdf",
            "content": "Overall document root summary synthesizing all themes.",
        },
    ]

    context = _build_context(chunks, query="Tell me about this document", enable_compression=False)

    assert "[Page 3]" in context
    assert "[Summary L1]" in context
    assert "[Root Summary]" in context


def test_validate_citations_matches_root_and_summaries():
    """Verify validate_citations properly detects [Root Summary] and [Summary]."""
    context_chunks = [
        {
            "id": "c1",
            "page_number": 5,
            "chunk_type": "text",
            "tree_level": 0,
            "is_root": False,
            "filename": "doc.pdf",
            "content": "Specific detail on page 5.",
        },
        {
            "id": "c2",
            "page_number": None,
            "chunk_type": "summary",
            "tree_level": 2,
            "is_root": True,
            "filename": "doc.pdf",
            "content": "The overall document discusses advanced retrieval algorithms.",
        },
    ]

    response_text = (
        "According to the **[Root Summary]**, the document discusses advanced retrieval algorithms. "
        "Furthermore, a key experiment is detailed in **[Page 5]**."
    )

    citations = validate_citations(response_text, context_chunks)
    assert len(citations) == 2
    cited_ids = {c["chunk_id"] for c in citations}
    assert "c1" in cited_ids
    assert "c2" in cited_ids

    # Check root metadata is preserved in citation payload
    root_cit = next(c for c in citations if c["chunk_id"] == "c2")
    assert root_cit["is_root"] is True
    assert root_cit["tree_level"] == 2
