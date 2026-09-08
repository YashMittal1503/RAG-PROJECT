"""
Qdrant vector store wrapper with Hybrid Search (Dense + BM25 Sparse).

Manages per-user collections in Qdrant Cloud for data isolation.
Collection naming: docs_{user_id}

Each point stores:
- vector:
  - "": 384-dim dense embedding from bge-small-en-v1.5
  - "bm25": sparse embedding from Qdrant/bm25 for exact keyword matching
- payload: metadata for filtering and citation display
"""

import logging
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings

logger = logging.getLogger(__name__)

# Async Qdrant client — created once, reused across requests
_client: AsyncQdrantClient | None = None


async def get_client() -> AsyncQdrantClient:
    """Get or create the async Qdrant client."""
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )
    return _client


def _collection_name(user_id: str) -> str:
    """
    Generate the Qdrant collection name for a user.
    Per-user collections provide true data isolation at the vector DB level.
    """
    # Replace hyphens in UUID to make a valid collection name
    return f"docs_{user_id.replace('-', '_')}"


async def ensure_collection(user_id: str) -> None:
    """
    Create the user's collection if it doesn't already exist,
    or upgrade an existing collection to support BM25 sparse vectors.
    """
    client = await get_client()
    name = _collection_name(user_id)

    # Check if collection already exists
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]

    if name not in existing_names:
        logger.info(f"Creating Qdrant collection with Dense + BM25 sparse vectors: {name}")
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,  # 384 for bge-small-en-v1.5
                distance=Distance.COSINE,
            ),
            sparse_vectors_config={
                "bm25": SparseVectorParams(),
            },
        )
        # Create payload indices required for filtering
        await client.create_payload_index(
            collection_name=name,
            field_name="chunk_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=name,
            field_name="doc_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    else:
        # Check if existing collection already has sparse_vectors_config
        try:
            coll_info = await client.get_collection(name)
            has_sparse = (
                coll_info.config.params.sparse_vectors
                and "bm25" in coll_info.config.params.sparse_vectors
            )
            if not has_sparse:
                logger.info(f"Upgrading existing collection {name} with BM25 sparse vector index...")
                await client.update_collection(
                    collection_name=name,
                    sparse_vectors_config={
                        "bm25": SparseVectorParams(),
                    },
                )
        except Exception as e:
            logger.warning(f"Could not verify/upgrade sparse config on collection {name}: {e}")


async def upsert_chunks(
    user_id: str,
    chunks: list[dict],
) -> None:
    """
    Upsert chunk vectors and metadata into the user's Qdrant collection.

    Each item in `chunks` should have:
    - id: str (UUID)
    - vector: list[float] (dense vector)
    - sparse_vector: SparseVector | None (optional BM25 sparse vector)
    - payload: dict with doc_id, chunk_type, filename, page_number, etc.
    """
    client = await get_client()
    name = _collection_name(user_id)

    points = []
    for chunk in chunks:
        # Support dual dense+bm25 vectors and legacy dense-only vectors
        if "sparse_vector" in chunk and chunk["sparse_vector"] is not None:
            vector = {
                "": chunk["vector"],
                "bm25": chunk["sparse_vector"],
            }
        else:
            vector = chunk["vector"]

        points.append(
            PointStruct(
                id=chunk["id"],
                vector=vector,
                payload=chunk["payload"],
            )
        )

    # Upsert in batches of 100 to avoid oversized requests
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        await client.upsert(
            collection_name=name,
            points=batch,
        )

    logger.info(f"Upserted {len(points)} chunks to collection {name}")


async def search(
    user_id: str,
    query_vector: list[float],
    query_sparse_vector: SparseVector | None = None,
    limit: int = 8,
    chunk_type_filter: str | None = None,
) -> list[dict]:
    """
    Search for similar chunks in the user's collection using Hybrid Search (Dense + BM25 RRF).

    If query_sparse_vector is provided, executes native server-side Reciprocal Rank Fusion (RRF)
    between dense cosine similarity and BM25 keyword matching.
    Otherwise, gracefully falls back to dense vector search.
    """
    client = await get_client()
    name = _collection_name(user_id)

    # Check if collection exists
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if name not in existing_names:
        return []

    # Build optional filter
    query_filter = None
    if chunk_type_filter:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value=chunk_type_filter),
                )
            ]
        )

    # If sparse BM25 query vector is provided, execute Hybrid Search with RRF
    if query_sparse_vector is not None:
        try:
            query_res = await client.query_points(
                collection_name=name,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using="",  # default unnamed dense vector
                        limit=limit * 2,
                        filter=query_filter,
                    ),
                    Prefetch(
                        query=query_sparse_vector,
                        using="bm25",  # named sparse vector
                        limit=limit * 2,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
            )
            return [
                {
                    "id": str(hit.id),
                    "score": hit.score,
                    **hit.payload,
                }
                for hit in query_res.points
            ]
        except Exception as e:
            logger.warning(
                f"Hybrid search failed on collection {name} ({e}). "
                f"Falling back to dense vector search..."
            )

    # Fallback to standard dense vector search
    results = await client.search(
        collection_name=name,
        query_vector=query_vector,
        limit=limit,
        query_filter=query_filter,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            **hit.payload,
        }
        for hit in results
    ]


async def delete_by_document(user_id: str, doc_id: str) -> None:
    """
    Delete all vectors belonging to a specific document from the user's collection.
    """
    client = await get_client()
    name = _collection_name(user_id)

    # Check if collection exists before trying to delete
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if name not in existing_names:
        return

    await client.delete(
        collection_name=name,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id),
                )
            ]
        ),
    )

    logger.info(f"Deleted vectors for doc {doc_id} from collection {name}")


async def get_summary_chunks(user_id: str) -> list[dict]:
    """
    Retrieve ALL summary-type chunks from the user's collection.
    Used when an aggregation query is detected, so summary chunks
    are included in context regardless of vector similarity.
    """
    client = await get_client()
    name = _collection_name(user_id)

    # Check if collection exists
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]
    if name not in existing_names:
        return []

    # Scroll through all points with chunk_type="summary"
    results, _ = await client.scroll(
        collection_name=name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="chunk_type",
                    match=MatchValue(value="summary"),
                )
            ]
        ),
        limit=100,
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "id": str(point.id),
            "score": 1.0,
            **point.payload,
        }
        for point in results
    ]
