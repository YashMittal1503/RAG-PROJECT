"""
Unit tests for Overview / Synopsis Query Intent and Lead-Chunk Anchoring.

Verifies:
1. is_overview_query detects overview inquiries ('tell me about', 'summary', 'synopsis', 'overview', etc.).
2. rerank_chunks preserves up to top_k candidates without context starvation.
"""

from unittest.mock import MagicMock, patch
import pytest

from app.services.query import is_overview_query
from app.services.reranker import rerank_chunks


class TestOverviewQueryDetection:
    """Test detection of broad document overview/synopsis questions."""

    def test_overview_phrases(self):
        assert is_overview_query("Tell me about it ends with us")
        assert is_overview_query("What is this book about?")
        assert is_overview_query("Can you give me an overview of the document?")
        assert is_overview_query("Summarize the novel")
        assert is_overview_query("Who wrote this paper and what is the synopsis?")
        assert is_overview_query("What is the plot summary?")

    def test_specific_questions_not_overview(self):
        assert not is_overview_query("What is Lily's phone number?")
        assert not is_overview_query("How much did the revenue increase in Q2?")
        assert not is_overview_query("Where did Ryle go after the restaurant?")


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
        mock_ranker.rerank.return_value = [
            {"id": f"chunk-{i}", "score": 0.0005 - (i * 0.00004)}
            for i in range(10)
        ]

        with patch("app.services.reranker.get_ranker", return_value=mock_ranker):
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
