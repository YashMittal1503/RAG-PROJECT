"""
Pydantic schemas for API request/response validation and serialization.

Organized by domain: documents, chunks, chat, and health.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Document schemas ──────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    """Single document in API responses."""
    id: UUID
    filename: str
    file_type: str
    file_size: int
    status: str
    failure_reason: Optional[str] = None
    chunk_count: Optional[int] = None
    is_tabular: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentStatusResponse(BaseModel):
    """Lightweight status-only response for polling."""
    id: UUID
    status: str
    failure_reason: Optional[str] = None
    chunk_count: Optional[int] = None

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    """Response after uploading files."""
    documents: list[DocumentResponse]
    errors: list[dict] = Field(default_factory=list)  # [{filename, error}]


# ── Chunk schemas ─────────────────────────────────────────────────────────

class ChunkResponse(BaseModel):
    """Chunk metadata for citation display."""
    id: UUID
    document_id: UUID
    chunk_type: str
    content: str
    page_number: Optional[int] = None
    row_range_start: Optional[int] = None
    row_range_end: Optional[int] = None

    model_config = {"from_attributes": True}


# ── Chat schemas ──────────────────────────────────────────────────────────

class ChatSessionCreate(BaseModel):
    """Request to create a new chat session."""
    title: str = "New conversation"


class ChatSessionResponse(BaseModel):
    """Chat session in API responses."""
    id: UUID
    title: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatMessageResponse(BaseModel):
    """Chat message in API responses."""
    id: UUID
    session_id: UUID
    role: str
    content: str
    citations: Optional[list[dict]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryRequest(BaseModel):
    """Request to ask a question in a chat session."""
    question: str = Field(..., min_length=1, max_length=2000)


class CitationData(BaseModel):
    """A single citation reference."""
    chunk_id: str
    filename: str
    page_number: Optional[int] = None
    row_range_start: Optional[int] = None
    row_range_end: Optional[int] = None
    content_preview: str = ""  # First ~200 chars of chunk


# ── Health schemas ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
