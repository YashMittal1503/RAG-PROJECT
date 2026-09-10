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


async def collection_supports_sparse(client: AsyncQdrantClient, name: str) -> bool:
    """Check whether a collection has the 'bm25' sparse vector configured."""
    try:
        coll_info = await client.get_collection(name)
        return bool(
            coll_info.config.params.sparse_vectors
            and "bm25" in coll_info.config.params.sparse_vectors
        )
    except Exception:
        return False


async def ensure_collection(user_id: str, force_recreate: bool = False) -> None:
    """
    Create the user's collection if it doesn't already exist,
    or recreate it if force_recreate is True,
    and ensure payload indices are present.
    """
    client = await get_client()
    name = _collection_name(user_id)

    exists = await client.collection_exists(name)
    if exists and force_recreate:
        logger.info(f"Force recreating collection {name} with dense + sparse vector config...")
        await client.delete_collection(name)
        exists = False

    if not exists:
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
    
    # Ensure payload indices required for fast filtering exist (safe idempotent calls)
    for field_name in ("chunk_type", "doc_id", "filename"):
        try:
            await client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass  # Index already exists or cannot be altered


async def recreate_collection(user_id: str) -> None:
    """Delete and recreate the user's collection with dual dense + sparse BM25 vector config."""
    await ensure_collection(user_id, force_recreate=True)


async def upsert_chunks(
    user_id: str,
    chunks: list[dict],
) -> None:
    """
    Upsert chunk vectors and metadata into the user's Qdrant collection.
    Automatically adapts to collection capabilities:
    - If the collection supports 'bm25' sparse vectors, upserts dual dense+bm25 vectors.
    - If the collection is legacy dense-only, upserts dense vectors.
    - If a vector name error occurs, automatically catches it and retries with dense-only vectors.
    """
    client = await get_client()
    name = _collection_name(user_id)

    has_sparse = await collection_supports_sparse(client, name)

    points = []
    for chunk in chunks:
        if has_sparse and "sparse_vector" in chunk and chunk["sparse_vector"] is not None:
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

    batch_size = 100
    try:
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await client.upsert(
                collection_name=name,
                points=batch,
            )
        logger.info(f"Upserted {len(points)} chunks to collection {name} (sparse={has_sparse})")
    except Exception as e:
        # If Qdrant failed because bm25 vector is missing from schema, fallback to dense-only vectors
        err_str = str(e).lower()
        if "vector name error" in err_str or "bm25" in err_str or "not existing" in err_str:
            logger.warning(
                f"Upsert with sparse vectors rejected by Qdrant schema ({e}). "
                f"Falling back to dense-only upsert for collection {name}..."
            )
            dense_points = [
                PointStruct(
                    id=p.id,
                    vector=p.vector[""] if isinstance(p.vector, dict) else p.vector,
                    payload=p.payload,
                )
                for p in points
            ]
            for i in range(0, len(dense_points), batch_size):
                batch = dense_points[i : i + batch_size]
                await client.upsert(
                    collection_name=name,
                    points=batch,
                )
            logger.info(f"Successfully upserted {len(dense_points)} dense-only chunks to collection {name}")
        else:
            raise


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
