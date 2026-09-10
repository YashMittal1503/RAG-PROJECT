import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath("."))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.database import AsyncSessionLocal
from app.models import Document, DocumentStatus
from app.services.query import retrieve_chunks, _build_context, generate_answer_stream
from sqlalchemy import select


async def test():
    async with AsyncSessionLocal() as db:
        res_doc = await db.execute(
            select(Document).where(
                Document.filename.ilike("%verity%"),
                Document.status == DocumentStatus.READY,
            )
        )
        doc = res_doc.scalars().first()

    if not doc:
        print("No Verity document found!")
        return

    print("Doc ID:", doc.id, "User ID:", doc.user_id, "Filename:", doc.filename)
    q = "What is the name of the narrator in the book Verity?"
    chunks = await retrieve_chunks(
        user_id=str(doc.user_id),
        question=q,
        is_aggregation=False,
        doc_id_filter=str(doc.id),
        filename_filter=doc.filename,
    )
    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        p = c.get("page_number")
        score = c.get("score")
        text = c.get("content", "")[:180].replace("\n", " ")
        print(f"  {i+1}. [Page {p}] score={score}: {text}...")

    print("\n--- Context built for LLM ---")
    ctx = _build_context(chunks, query=q)
    print(ctx[:800])

    print("\n--- Generating Answer ---")
    ans = ""
    async for token in generate_answer_stream(q, chunks):
        ans += token
    print("ANSWER:\n", ans)


if __name__ == "__main__":
    asyncio.run(test())
