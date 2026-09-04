"""
Qdrant vector store wrapper.

Manages per-user collections in Qdrant Cloud for data isolation.
Collection naming: docs_{user_id}

Each point stores:
- vector: 384-dim embedding from bge-small-en-v1.5
- payload: metadata for filtering and citation display
"""

import logging
from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
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
    Create the user's collection if it doesn't already exist.
    Called during the first document upload for a user.
    """
    client = await get_client()
    name = _collection_name(user_id)

    # Check if collection already exists
    collections = await client.get_collections()
    existing_names = [c.name for c in collections.collections]

    if name not in existing_names:
        logger.info(f"Creating Qdrant collection: {name}")
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.embedding_dim,  # 384 for bge-small-en-v1.5
                distance=Distance.COSINE,
            ),
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


async def upsert_chunks(
    user_id: str,
    chunks: list[dict],
) -> None:
    """
    Upsert chunk vectors and metadata into the user's Qdrant collection.

    Each item in `chunks` should have:
    - id: str (UUID)
    - vector: list[float]
    - payload: dict with doc_id, chunk_type, filename, page_number, etc.
    """
    client = await get_client()
    name = _collection_name(user_id)

    points = [
        PointStruct(
            id=chunk["id"],
            vector=chunk["vector"],
            payload=chunk["payload"],
        )
        for chunk in chunks
    ]

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
    limit: int = 8,
    chunk_type_filter: str | None = None,
) -> list[dict]:
    """
    Search for similar chunks in the user's collection.

    Returns a list of dicts with: id, score, and all payload fields.
    Optionally filters by chunk_type (e.g., "summary").
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
        limit=100,  # Plenty for Phase 1 volumes
        with_payload=True,
        with_vectors=False,
    )

    return [
        {
            "id": str(point.id),
            "score": 1.0,  # Summary chunks get max relevance when explicitly fetched
            **point.payload,
        }
        for point in results
    ]
