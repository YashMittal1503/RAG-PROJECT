"""
Document ingestion orchestrator.

Runs as a background asyncio task (not a Celery/RQ worker).
Coordinates the full pipeline: parse → chunk → embed → store.

Status transitions: queued → parsing → chunking → embedding → ready
Any failure sets status to "failed" with a human-readable reason.
"""

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Chunk, Document, DocumentStatus
from app.services import storage, embedding, vector_store
from app.services.parsing import parse_pdf, parse_txt, parse_spreadsheet
from app.services.chunking import (
    ChunkData,
    chunk_text,
    chunk_text_from_string,
    chunk_spreadsheet,
)

logger = logging.getLogger(__name__)


async def _update_status(
    doc_id: uuid.UUID,
    status: DocumentStatus,
    failure_reason: str | None = None,
    chunk_count: int | None = None,
) -> None:
    """Update a document's status in the database."""
    async with AsyncSessionLocal() as session:
        values = {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
        }
        if failure_reason is not None:
            values["failure_reason"] = failure_reason
        if chunk_count is not None:
            values["chunk_count"] = chunk_count

        await session.execute(
            update(Document)
            .where(Document.id == doc_id)
            .values(**values)
        )
        await session.commit()


async def ingest_document(
    doc_id: uuid.UUID,
    user_id: str,
    filename: str,
    file_type: str,
    storage_path: str,
) -> None:
    """
    Full ingestion pipeline for a single document.

    This function is designed to be called via asyncio.create_task()
    so it runs in the background without blocking the upload response.
    """
    try:
        # ── Step 1: Parsing ───────────────────────────────────────────
        await _update_status(doc_id, DocumentStatus.PARSING)
        logger.info(f"[{doc_id}] Parsing {filename} ({file_type})")

        file_bytes = await storage.download_file(storage_path)

        all_chunks: list[ChunkData] = []

        if file_type == "pdf":
            pages = parse_pdf(file_bytes)
            await _update_status(doc_id, DocumentStatus.CHUNKING)
            all_chunks = chunk_text(pages, filename=filename)

        elif file_type == "txt":
            text = parse_txt(file_bytes)
            await _update_status(doc_id, DocumentStatus.CHUNKING)
            all_chunks = chunk_text_from_string(text, filename=filename)

        elif file_type in ("xlsx", "csv"):
            sheets = parse_spreadsheet(file_bytes, file_type)
            await _update_status(doc_id, DocumentStatus.CHUNKING)
            for sheet in sheets:
                sheet_chunks = chunk_spreadsheet(sheet, filename=filename)
                all_chunks.extend(sheet_chunks)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if not all_chunks:
            raise ValueError("No content could be extracted from the file.")

        logger.info(f"[{doc_id}] Created {len(all_chunks)} chunks")

        # ── Step 2: Embedding ─────────────────────────────────────────
        await _update_status(doc_id, DocumentStatus.EMBEDDING)
        logger.info(f"[{doc_id}] Embedding {len(all_chunks)} chunks")

        # Batch embed all chunk texts
        chunk_texts = [c.content for c in all_chunks]
        vectors = embedding.embed_texts(chunk_texts)

        # ── Step 3: Store in Qdrant ───────────────────────────────────
        await vector_store.ensure_collection(user_id)

        qdrant_points = []
        for chunk, vector in zip(all_chunks, vectors):
            qdrant_points.append({
                "id": chunk.id,
                "vector": vector,
                "payload": {
                    "doc_id": str(doc_id),
                    "chunk_type": chunk.chunk_type,
                    "filename": filename,
                    "page_number": chunk.page_number,
                    "row_range_start": chunk.row_range_start,
                    "row_range_end": chunk.row_range_end,
                    "content": chunk.content,
                },
            })

        await vector_store.upsert_chunks(user_id, qdrant_points)

        # ── Step 4: Save chunk metadata to Postgres ───────────────────
        async with AsyncSessionLocal() as session:
            for chunk in all_chunks:
                db_chunk = Chunk(
                    id=uuid.UUID(chunk.id),
                    document_id=doc_id,
                    user_id=uuid.UUID(user_id),
                    chunk_index=chunk.chunk_index,
                    chunk_type=chunk.chunk_type,
                    content=chunk.content,
                    page_number=chunk.page_number,
                    row_range_start=chunk.row_range_start,
                    row_range_end=chunk.row_range_end,
                    token_count=chunk.token_count,
                )
                session.add(db_chunk)
            await session.commit()

        # ── Step 5: Mark as ready ─────────────────────────────────────
        await _update_status(
            doc_id,
            DocumentStatus.READY,
            chunk_count=len(all_chunks),
        )
        logger.info(f"[{doc_id}] Ingestion complete — {len(all_chunks)} chunks ready")

    except ValueError as e:
        # Expected errors (parsing failures, empty files, etc.)
        logger.warning(f"[{doc_id}] Ingestion failed: {e}")
        await _update_status(doc_id, DocumentStatus.FAILED, failure_reason=str(e))

    except Exception as e:
        # Unexpected errors
        logger.exception(f"[{doc_id}] Unexpected ingestion error")
        await _update_status(
            doc_id,
            DocumentStatus.FAILED,
            failure_reason=f"An unexpected error occurred during processing: {type(e).__name__}",
        )
