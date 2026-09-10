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


async def test_expanded():
    async with AsyncSessionLocal() as db:
        d = (
            await db.execute(select(Document).where(Document.filename.ilike("%verity%")))
        ).scalar_one()

    queries = [
        "What is the name of the narrator in the book Verity?",
        "Who is the narrator, protagonist, or main character in Verity and what is her name?",
        "What is the main character and narrator's name in Verity?",
    ]
    for q in queries:
        q_dense = embedding.embed_query(q)
        q_sparse = embedding.embed_sparse_query(q)
        res = await vector_store.search(
            str(d.user_id), q_dense, q_sparse, limit=30, doc_id_filter=str(d.id)
        )
        reranked = reranker.rerank_chunks(q, res, top_k=5)
        print("\n=== Query:", q)
        for i, r in enumerate(reranked):
            has_lowen = "lowen" in r.get("content", "").lower()
            text = r.get("content", "")[:120].replace("\n", " ")
            print(f"  {i+1}. [Page {r.get('page_number')}] score={r.get('score'):.4f} HasLowen={has_lowen} | {text}...")


if __name__ == "__main__":
    asyncio.run(test_expanded())
