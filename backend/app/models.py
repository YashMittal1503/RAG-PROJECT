"""
SQLAlchemy ORM models for the RAG chatbot.

Tables:
- documents:      uploaded files and their processing status
- chunks:         chunk metadata (text lives here; vectors live in Qdrant)
- chat_sessions:  user conversation threads
- chat_messages:  individual messages within a session
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ── Document status enum ──────────────────────────────────────────────────

import enum

class DocumentStatus(str, enum.Enum):
    """Processing pipeline states for an uploaded document."""
    QUEUED = "queued"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    READY = "ready"
    FAILED = "failed"


# ── Document ──────────────────────────────────────────────────────────────

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)          # pdf, txt, xlsx, csv
    file_size = Column(Integer, nullable=False)              # bytes
    storage_path = Column(String(500), nullable=False)       # path in Supabase Storage
    status = Column(
        SAEnum(
            DocumentStatus,
            name="document_status",
            create_constraint=False,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        default=DocumentStatus.QUEUED,
    )
    failure_reason = Column(Text, nullable=True)             # human-readable error
    chunk_count = Column(Integer, nullable=True)             # set after chunking
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationship: a document has many chunks
    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


# ── Chunk ─────────────────────────────────────────────────────────────────

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)            # order within document
    chunk_type = Column(String(20), nullable=False)          # "text" or "summary"
    content = Column(Text, nullable=False)                   # the chunk text
    page_number = Column(Integer, nullable=True)             # PDF only
    row_range_start = Column(Integer, nullable=True)         # spreadsheet only
    row_range_end = Column(Integer, nullable=True)           # spreadsheet only
    token_count = Column(Integer, nullable=False)

    # Relationship back to document
    document = relationship("Document", back_populates="chunks")


# ── ChatSession ───────────────────────────────────────────────────────────

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    title = Column(String(255), nullable=False, default="New conversation")
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationship: a session has many messages
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.created_at",
    )


# ── ChatMessage ───────────────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)                # "user" or "assistant"
    content = Column(Text, nullable=False)
    citations = Column(JSONB, nullable=True)                 # [{chunk_id, filename, ...}]
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationship back to session
    session = relationship("ChatSession", back_populates="messages")
