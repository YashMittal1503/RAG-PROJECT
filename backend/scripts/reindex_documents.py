"""
Reindex Documents Script

Recreates Qdrant collections with native dual Dense + BM25 sparse vector schemas
and re-ingests all PDF and text documents through the production pipeline.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models import Document, DocumentStatus
from app.services import vector_store, embedding
from app.services.ingestion import ingest_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex_documents")


async def main():
    logger.info("Starting document re-indexing migration...")

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.file_type.in_(["pdf", "txt"]))
        )
        docs = result.scalars().all()

    if not docs:
        logger.info("No PDF or TXT documents found to re-index.")
        return

    # Group documents by user_id
    user_docs: dict[str, list[Document]] = {}
    for doc in docs:
        uid = str(doc.user_id)
        user_docs.setdefault(uid, []).append(doc)

    logger.info(f"Found {len(docs)} documents across {len(user_docs)} user(s).")

    client = await vector_store.get_client()

    for user_id, doc_list in user_docs.items():
        coll_name = vector_store._collection_name(user_id)
        logger.info(f"=== Recreating collection {coll_name} for user {user_id} ===")

        await vector_store.recreate_collection(user_id)
        supports_sparse = await vector_store.collection_supports_sparse(client, coll_name)
        logger.info(f"Collection {coll_name} created. Sparse BM25 supported: {supports_sparse}")

        if not supports_sparse:
            logger.error(f"Failed to enable sparse vectors on {coll_name}. Aborting for user {user_id}.")
            continue

        for idx, doc in enumerate(doc_list, 1):
            logger.info(
                f"[{idx}/{len(doc_list)}] Processing '{doc.filename}' "
                f"(id: {doc.id}, storage_path: {doc.storage_path})..."
            )
            try:
                await ingest_document(
                    doc_id=doc.id,
                    user_id=str(doc.user_id),
                    filename=doc.filename,
                    file_type=doc.file_type,
                    storage_path=doc.storage_path,
                )
                logger.info(f"Successfully finished ingestion for '{doc.filename}'.")
            except Exception as e:
                logger.error(f"Error during ingestion of '{doc.filename}': {e}", exc_info=True)

        # Inspect final collection state
        info = await client.get_collection(coll_name)
        logger.info(f"Collection {coll_name} final points_count: {info.points_count}")

        # Verify points have both dense and sparse vectors
        scroll_res, _ = await client.scroll(
            collection_name=coll_name,
            limit=1,
            with_vectors=True,
            with_payload=True,
        )
        if scroll_res:
            sample_pt = scroll_res[0]
            vector_keys = list(sample_pt.vector.keys()) if isinstance(sample_pt.vector, dict) else [type(sample_pt.vector).__name__]
            logger.info(f"Sample point ID: {sample_pt.id}, vector keys: {vector_keys}")
            has_bm25 = isinstance(sample_pt.vector, dict) and "bm25" in sample_pt.vector
            logger.info(f"Point contains BM25 sparse vector: {has_bm25}")

        # Test Native Hybrid Search with RRF
        test_query = "Python programming variables functions Colleen Hoover Verity"
        q_dense = embedding.embed_query(test_query)
        q_sparse = embedding.embed_sparse_query(test_query)
        logger.info(f"Testing Hybrid Search on {coll_name} with query: '{test_query}'...")
        hits = await vector_store.search(
            user_id=user_id,
            query_vector=q_dense,
            query_sparse_vector=q_sparse,
            limit=3,
        )
        logger.info(f"Hybrid Search returned {len(hits)} hits:")
        for h in hits:
            logger.info(f"  - [{h.get('filename')}] score: {h.get('score'):.4f} snippet: {h.get('content', '')[:60]}...")

    logger.info("=== Migration complete! All documents re-indexed with native Hybrid vectors. ===")


if __name__ == "__main__":
    asyncio.run(main())
