"""
Document ingestion orchestrator.

Runs as a background asyncio task (not a Celery/RQ worker).
Coordinates the full pipeline: parse → chunk → embed → store.

Status transitions: queued → parsing → chunking → embedding → ready
Any failure sets status to "failed" with a human-readable reason.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

import logfire
from sqlalchemy import update, delete
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
from app.services import tabular_store

logger = logging.getLogger(__name__)


async def _update_status(
    doc_id: uuid.UUID,
    status: DocumentStatus,
    failure_reason: str | None = None,
    chunk_count: int | None = None,
) -> None:
    """Update a document's status in the database with retry for resilience."""
    for attempt in range(3):
        try:
            async with AsyncSessionLocal() as session:
                values = {
                    "status": status,
                    "updated_at": datetime.now(timezone.utc),
                }
                if failure_reason is not None:
                    values["failure_reason"] = failure_reason
                elif status == DocumentStatus.READY:
                    values["failure_reason"] = None
                if chunk_count is not None:
                    values["chunk_count"] = chunk_count

                await session.execute(
                    update(Document)
                    .where(Document.id == doc_id)
                    .values(**values)
                )
                await session.commit()
                return
        except Exception as e:
            if attempt == 2:
                logger.error(f"Failed to update status for doc {doc_id} to {status}: {e}")
                raise
            logger.warning(f"Retrying status update for doc {doc_id} (attempt {attempt + 1}): {e}")
            await asyncio.sleep(0.5 * (attempt + 1))


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
    with logfire.span("📥 Ingestion Pipeline | {filename}", filename=filename, doc_id=str(doc_id), file_type=file_type, user_id=user_id) as pipe_span:
        try:
            # ── Step 1: Parsing ───────────────────────────────────────
            await _update_status(doc_id, DocumentStatus.PARSING)
            logger.info(f"[{doc_id}] Parsing {filename} ({file_type})")

            with logfire.span("💾 Download Storage File | {filename}", filename=filename, storage_path=storage_path):
                file_bytes = await storage.download_file(storage_path)

            all_chunks: list[ChunkData] = []

            if file_type == "pdf":
                with logfire.span("📄 Parse PDF (PyMuPDF + RapidOCR) | {filename}", filename=filename) as p_span:
                    pages = await asyncio.to_thread(parse_pdf, file_bytes)
                    p_span.set_attribute("page_count", len(pages))
                await _update_status(doc_id, DocumentStatus.CHUNKING)
                with logfire.span("🧩 Chunk PDF Content | {filename}", filename=filename) as c_span:
                    all_chunks = await asyncio.to_thread(chunk_text, pages, filename=filename)
                    c_span.set_attribute("chunk_count", len(all_chunks))

            elif file_type == "txt":
                with logfire.span("📄 Parse Text File | {filename}", filename=filename):
                    text = await asyncio.to_thread(parse_txt, file_bytes)
                await _update_status(doc_id, DocumentStatus.CHUNKING)
                with logfire.span("🧩 Chunk Text Content | {filename}", filename=filename) as c_span:
                    all_chunks = await asyncio.to_thread(chunk_text_from_string, text, filename=filename)
                    c_span.set_attribute("chunk_count", len(all_chunks))

            elif file_type in ("xlsx", "csv"):
                with logfire.span("📊 Parse Spreadsheet | {filename}", filename=filename) as s_span:
                    sheets = await asyncio.to_thread(parse_spreadsheet, file_bytes, file_type)
                    s_span.set_attribute("sheet_count", len(sheets))

                # ── Text-to-SQL path: store in DuckDB, skip chunking/embedding ──
                await _update_status(doc_id, DocumentStatus.STORING)
                logger.info(f"[{doc_id}] Storing {len(sheets)} sheet(s) in DuckDB")

                total_rows = 0
                with logfire.span("🦆 Store Tabular in DuckDB | {filename}", filename=filename) as duck_span:
                    for sheet in sheets:
                        result = await asyncio.to_thread(
                            tabular_store.store_dataframe,
                            user_id, doc_id, sheet.name, sheet.df,
                        )
                        total_rows += result["rows"]
                    duck_span.set_attribute("total_rows", total_rows)

                # Mark the document as tabular in the DB
                async with AsyncSessionLocal() as session:
                    await session.execute(
                        update(Document)
                        .where(Document.id == doc_id)
                        .values(is_tabular=True)
                    )
                    await session.commit()

                # Mark as ready — no chunking/embedding needed
                await _update_status(
                    doc_id,
                    DocumentStatus.READY,
                    chunk_count=total_rows,
                )
                pipe_span.set_attribute("status", "ready")
                pipe_span.set_attribute("rows_stored", total_rows)
                logfire.info("✅ Tabular ingestion complete for '{filename}': {total_rows} rows stored in DuckDB", filename=filename, total_rows=total_rows, doc_id=str(doc_id))
                logger.info(f"[{doc_id}] Tabular ingestion complete — {total_rows} rows stored")
                return  # Early return — skip the embedding pipeline below

            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            if not all_chunks:
                raise ValueError("No content could be extracted from the file.")

            pipe_span.set_attribute("chunk_count", len(all_chunks))
            logger.info(f"[{doc_id}] Created {len(all_chunks)} chunks")

            # ── Step 2: Embedding (Dense + BM25 Sparse) ───────────────
            await _update_status(doc_id, DocumentStatus.EMBEDDING)
            logger.info(f"[{doc_id}] Embedding {len(all_chunks)} chunks (Dense + BM25)")

            with logfire.span("⚡ Generate Embeddings | {count} chunks (Dense + BM25)", count=len(all_chunks)):
                chunk_texts = [c.content for c in all_chunks]
                batch_embed_size = 64
                vectors = []
                sparse_vectors = []
                for i in range(0, len(chunk_texts), batch_embed_size):
                    sub_texts = chunk_texts[i : i + batch_embed_size]
                    sub_vectors = await asyncio.to_thread(embedding.embed_texts, sub_texts)
                    sub_sparse = await asyncio.to_thread(embedding.embed_sparse_texts, sub_texts)
                    vectors.extend(sub_vectors)
                    sparse_vectors.extend(sub_sparse)

            # ── Step 3: Store in Qdrant ───────────────────────────────
            with logfire.span("🗄️ Upsert to Qdrant | {points_count} points", points_count=len(all_chunks)):
                await vector_store.ensure_collection(user_id)
                # Purge any previously stored vectors for this doc (safe for re-ingestion)
                try:
                    await vector_store.delete_by_document(user_id, str(doc_id))
                except Exception as del_err:
                    logger.warning(f"[{doc_id}] Could not clear previous vectors: {del_err}")

                qdrant_points = []
                for chunk, vector, sparse_vector in zip(all_chunks, vectors, sparse_vectors):
                    qdrant_points.append({
                        "id": chunk.id,
                        "vector": vector,
                        "sparse_vector": sparse_vector,
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

            # ── Step 4: Save chunk metadata to Postgres ───────────────
            with logfire.span("💾 Persist Chunks to Postgres | {chunk_count} chunks", chunk_count=len(all_chunks)):
                # Clean up existing chunks for this document to prevent duplicates on re-ingestion
                async with AsyncSessionLocal() as session:
                    await session.execute(
                        delete(Chunk).where(Chunk.document_id == doc_id)
                    )
                    await session.commit()

                db_chunks = [
                    Chunk(
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
                    for chunk in all_chunks
                ]

                batch_db_size = 100
                for i in range(0, len(db_chunks), batch_db_size):
                    sub_db = db_chunks[i : i + batch_db_size]
                    for attempt in range(3):
                        try:
                            async with AsyncSessionLocal() as session:
                                session.add_all(sub_db)
                                await session.commit()
                                break
                        except Exception as e:
                            if attempt == 2:
                                raise
                            logger.warning(f"[{doc_id}] Retrying chunk save (attempt {attempt + 1}): {e}")
                            await asyncio.sleep(0.5 * (attempt + 1))

            # ── Step 5: Mark as ready ─────────────────────────────────
            await _update_status(
                doc_id,
                DocumentStatus.READY,
                chunk_count=len(all_chunks),
            )
            pipe_span.set_attribute("status", "ready")
            logfire.info("✅ Ingestion complete for '{filename}': {chunk_count} chunks indexed", filename=filename, chunk_count=len(all_chunks), doc_id=str(doc_id))
            logger.info(f"[{doc_id}] Ingestion complete — {len(all_chunks)} chunks ready")

        except ValueError as e:
            # Expected errors (parsing failures, empty files, etc.)
            logger.warning(f"[{doc_id}] Ingestion failed: {e}")
            pipe_span.record_exception(e)
            await _update_status(doc_id, DocumentStatus.FAILED, failure_reason=str(e))

        except Exception as e:
            # Unexpected errors
            logger.exception(f"[{doc_id}] Unexpected ingestion error")
            pipe_span.record_exception(e)
            try:
                await _update_status(
                    doc_id,
                    DocumentStatus.FAILED,
                    failure_reason=f"An unexpected error occurred during processing: {type(e).__name__}",
                )
            except Exception:
                logger.exception(f"[{doc_id}] Failed to record failure status")
