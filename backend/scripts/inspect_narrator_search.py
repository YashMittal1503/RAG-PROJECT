import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import AsyncSessionLocal
from app.models import Document
from app.services import embedding, reranker, vector_store
from sqlalchemy import select


async def check():
    async with AsyncSessionLocal() as db:
        d = (
            await db.execute(select(Document).where(Document.filename.ilike("%verity%")))
        ).scalar_one()

    q = "What is the name of the narrator in the book Verity?"
    q_dense = embedding.embed_query(q)
    q_sparse = embedding.embed_sparse_query(q)

    results = await vector_store.search(
        user_id=str(d.user_id),
        query_vector=q_dense,
        query_sparse_vector=q_sparse,
        limit=25,
        doc_id_filter=str(d.id),
    )
    print(f"Top {len(results)} search hits for: {q}")
    for i, r in enumerate(results):
        has_lowen = "lowen" in r.get("content", "").lower()
        print(f"  {i+1}. [Page {r.get('page_number')}] score={r.get('score'):.4f} HasLowen={has_lowen}")

    # Now let's see FlashRank on top 25 vs top 10
    print("\n--- FlashRank on top 10 ---")
    reranked_10 = reranker.rerank_chunks(q, results[:10], top_k=5)
    for i, r in enumerate(reranked_10):
        print(f"  {i+1}. [Page {r.get('page_number')}] score={r.get('score'):.4f} HasLowen={'lowen' in r.get('content', '').lower()}")

    print("\n--- FlashRank on top 25 ---")
    reranked_25 = reranker.rerank_chunks(q, results, top_k=5)
    for i, r in enumerate(reranked_25):
        print(f"  {i+1}. [Page {r.get('page_number')}] score={r.get('score'):.4f} HasLowen={'lowen' in r.get('content', '').lower()}")


if __name__ == "__main__":
    asyncio.run(check())
