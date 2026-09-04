"""
Document management endpoints: upload, list, status, delete.

All endpoints require authentication and scope data to the
authenticated user's ID for data isolation.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy import select, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.models import Chunk, Document, DocumentStatus
from app.schemas import DocumentResponse, DocumentStatusResponse, UploadResponse
from app.services import storage, vector_store
from app.services.ingestion import ingest_document
from app.utils.file_validation import validate_file_type

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload one or more documents for processing.

    Validates each file by magic bytes, uploads to Supabase Storage,
    creates a Document record with status=queued, and kicks off
    background ingestion.

    Returns immediately — the frontend polls for status updates.
    """
    # Check batch size limit
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {settings.max_batch_size} files per upload.",
        )

    documents = []
    errors = []

    for file in files:
        try:
            # Read file content
            file_bytes = await file.read()

            # Check file size
            if len(file_bytes) > settings.max_file_size_mb * 1024 * 1024:
                errors.append({
                    "filename": file.filename,
                    "error": f"File exceeds the {settings.max_file_size_mb}MB size limit.",
                })
                continue

            # Validate file type by magic bytes
            file_type, mime_type = validate_file_type(file_bytes, file.filename)

            # Generate document ID
            doc_id = uuid.uuid4()

            # Upload to Supabase Storage
            storage_path = await storage.upload_file(
                user_id=user_id,
                doc_id=doc_id,
                filename=file.filename,
                file_bytes=file_bytes,
                content_type=mime_type,
            )

            # Create document record
            doc = Document(
                id=doc_id,
                user_id=uuid.UUID(user_id),
                filename=file.filename,
                file_type=file_type,
                file_size=len(file_bytes),
                storage_path=storage_path,
                status=DocumentStatus.QUEUED,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(doc)
            await db.flush()  # Flush to get the ID before starting background task

            # Start background ingestion (non-blocking)
            asyncio.create_task(
                ingest_document(
                    doc_id=doc_id,
                    user_id=user_id,
                    filename=file.filename,
                    file_type=file_type,
                    storage_path=storage_path,
                )
            )

            documents.append(DocumentResponse.model_validate(doc))

        except ValueError as e:
            # File validation errors (wrong type, blocked format, etc.)
            errors.append({
                "filename": file.filename or "unknown",
                "error": str(e),
            })
        except Exception as e:
            logger.exception(f"Failed to process upload: {file.filename}")
            errors.append({
                "filename": file.filename or "unknown",
                "error": "An unexpected error occurred during upload.",
            })

    await db.commit()

    return UploadResponse(documents=documents, errors=errors)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all documents belonging to the authenticated user."""
    result = await db.execute(
        select(Document)
        .where(Document.user_id == uuid.UUID(user_id))
        .order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get("/{doc_id}/status", response_model=DocumentStatusResponse)
async def get_document_status(
    doc_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the processing status of a single document (for polling)."""
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.user_id == uuid.UUID(user_id),
        )
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return DocumentStatusResponse.model_validate(doc)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    doc_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a document and all its associated data:
    - Chunk records from Postgres (CASCADE)
    - Vectors from Qdrant
    - File from Supabase Storage
    """
    # Verify document exists and belongs to the user
    result = await db.execute(
        select(Document).where(
            Document.id == doc_id,
            Document.user_id == uuid.UUID(user_id),
        )
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    # Delete vectors from Qdrant
    try:
        await vector_store.delete_by_document(user_id, str(doc_id))
    except Exception as e:
        logger.warning(f"Failed to delete vectors for doc {doc_id}: {e}")

    # Delete file from Supabase Storage
    await storage.delete_file(doc.storage_path)

    # Delete document record (chunks cascade automatically)
    await db.execute(
        sa_delete(Document).where(Document.id == doc_id)
    )
    await db.commit()
