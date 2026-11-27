"""
Unit tests for reranker candidate retention on overview-style queries.

Verifies that rerank_chunks preserves up to top_k candidates without context starvation.
(Overview detection itself is now the LLM query classifier: see test_query_classification.py.)
"""

from unittest.mock import MagicMock, patch
import pytest

from app.services.reranker import rerank_chunks


class TestRerankerCandidateRetention:
    """Test that reranker preserves candidates up to top_k and does not starve context."""

    def test_rerank_preserves_top_k_candidates(self):
        # 10 mock candidate chunks
        candidates = [
            {"id": f"chunk-{i}", "content": f"Passage text number {i}", "score": 0.5 - (i * 0.02)}
            for i in range(10)
        ]

        mock_ranker = MagicMock()
        # Mock cross-encoder returning tiny scores (< 0.001) as often happens for prose
        # With the cap removed, all 10 candidates are now evaluated by the reranker
        mock_ranker.rerank.return_value = [
            {"id": f"chunk-{i}", "score": 0.0005 - (i * 0.00004)}
            for i in range(10)
        ]

        with patch("app.services.reranker.settings.enable_reranker", True), patch(
            "app.services.reranker.get_ranker", return_value=mock_ranker
        ):
            results = rerank_chunks(
                query="Tell me about the novel",
                chunks=candidates,
                top_k=5,
                min_score=0.0001,
            )

            # Must preserve up to top_k (5) candidates, not decimate to 2
            assert len(results) == 5
            assert results[0]["id"] == "chunk-0"
            assert results[4]["id"] == "chunk-4"

    def test_rerank_disabled_passthrough(self):
        """Test that disabling reranker passes Qdrant Hybrid RRF candidates through directly."""
        candidates = [
            {"id": f"chunk-{i}", "content": f"Passage text {i}", "score": 0.9 - (i * 0.05)}
            for i in range(10)
        ]

        with patch("app.services.reranker.settings.enable_reranker", False):
            results = rerank_chunks(
                query="Tell me about the novel",
                chunks=candidates,
                top_k=5,
            )
            assert len(results) == 5
            assert results[0]["id"] == "chunk-0"
            assert results[4]["id"] == "chunk-4"
