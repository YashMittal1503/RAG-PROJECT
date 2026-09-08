"""
Unit tests for BM25 Sparse Embedding and Hybrid Search.

Verifies:
1. SparseTextEmbedding (Qdrant/bm25) generates valid Qdrant SparseVector instances.
2. Query sparse embedding produces proper term weights and indices.
3. Vector store search handles dual dense+sparse queries with RRF fusion.
4. Fallback from hybrid to dense search on error/legacy collections.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

from app.services import embedding, vector_store


class TestBM25SparseEmbedding:
    """Test BM25 sparse vector generation using FastEmbed."""

    def test_init_sparse_model(self):
        model = embedding.init_sparse_model()
        assert model is not None
        assert embedding.is_sparse_model_loaded()

    def test_embed_sparse_texts_empty(self):
        vectors = embedding.embed_sparse_texts([])
        assert vectors == []

    def test_embed_sparse_texts_content(self):
        texts = [
            "Revenue for Q3 reached $1.2 million with 25% growth.",
            "Customer churn rate decreased significantly in the European region.",
        ]
        vectors = embedding.embed_sparse_texts(texts)
        assert len(vectors) == 2
        for vec in vectors:
            assert isinstance(vec, SparseVector)
            assert len(vec.indices) > 0
            assert len(vec.values) > 0
            assert len(vec.indices) == len(vec.values)
            assert all(isinstance(idx, int) for idx in vec.indices)
            assert all(isinstance(val, float) for val in vec.values)

    def test_embed_sparse_query(self):
        query = "quarterly revenue growth"
        sparse_vec = embedding.embed_sparse_query(query)
        assert isinstance(sparse_vec, SparseVector)
        assert len(sparse_vec.indices) > 0
        assert len(sparse_vec.values) > 0
        assert len(sparse_vec.indices) == len(sparse_vec.values)


class TestHybridVectorStoreSearch:
    """Test hybrid search behavior with RRF fusion and fallback."""

    @pytest.mark.anyio
    async def test_hybrid_search_executes_rrf_query_points(self):
        mock_client = AsyncMock()

        # Mock collection check
        coll_mock = MagicMock()
        coll_mock.collections = [MagicMock(name="docs_user_123")]
        coll_mock.collections[0].name = "docs_user_123"
        mock_client.get_collections.return_value = coll_mock

        # Mock query_points response
        point_hit = MagicMock()
        point_hit.id = "chunk-1"
        point_hit.score = 0.95
        point_hit.payload = {"content": "Sample text", "filename": "doc.pdf"}

        query_response = MagicMock()
        query_response.points = [point_hit]
        mock_client.query_points.return_value = query_response

        with patch("app.services.vector_store.get_client", return_value=mock_client):
            dense_vec = [0.1] * 384
            sparse_vec = SparseVector(indices=[101, 202], values=[1.5, 2.0])

            results = await vector_store.search(
                user_id="user-123",
                query_vector=dense_vec,
                query_sparse_vector=sparse_vec,
                limit=5,
            )

            assert len(results) == 1
            assert results[0]["id"] == "chunk-1"
            assert results[0]["content"] == "Sample text"

            # Verify query_points was called with prefetch containing both vectors and RRF fusion
            mock_client.query_points.assert_called_once()
            call_kwargs = mock_client.query_points.call_args.kwargs
            assert call_kwargs["collection_name"] == "docs_user_123"
            assert len(call_kwargs["prefetch"]) == 2
            assert call_kwargs["query"].fusion == Fusion.RRF

    @pytest.mark.anyio
    async def test_hybrid_search_fallback_to_dense_on_error(self):
        mock_client = AsyncMock()

        # Mock collection check
        coll_mock = MagicMock()
        coll_mock.collections = [MagicMock(name="docs_user_123")]
        coll_mock.collections[0].name = "docs_user_123"
        mock_client.get_collections.return_value = coll_mock

        # Make query_points fail (e.g. legacy collection without sparse vectors)
        mock_client.query_points.side_effect = Exception("Collection has no sparse vectors configured")

        # Mock dense search fallback
        dense_hit = MagicMock()
        dense_hit.id = "fallback-1"
        dense_hit.score = 0.88
        dense_hit.payload = {"content": "Dense fallback text", "filename": "doc.pdf"}
        mock_client.search.return_value = [dense_hit]

        with patch("app.services.vector_store.get_client", return_value=mock_client):
            dense_vec = [0.1] * 384
            sparse_vec = SparseVector(indices=[101, 202], values=[1.5, 2.0])

            results = await vector_store.search(
                user_id="user-123",
                query_vector=dense_vec,
                query_sparse_vector=sparse_vec,
                limit=5,
            )

            # Successfully fell back to dense search
            assert len(results) == 1
            assert results[0]["id"] == "fallback-1"
            assert results[0]["content"] == "Dense fallback text"
            mock_client.search.assert_called_once()
