"""
Chat endpoints: session management and streaming Q&A.

The query endpoint uses SSE (Server-Sent Events) to stream
the LLM response token-by-token to the frontend.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
import logfire
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.auth import get_current_user
from app.database import AsyncSessionLocal, get_db
from app.models import ChatMessage, ChatSession, Document, DocumentStatus
from app.schemas import (
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatSessionUpdate,
    QueryRequest,
)
from app.services.query import (
    analyze_document_scope,
    check_tabular_data,
    classify_intent,
    detect_aggregation,
    execute_sql_query,
    format_sql_result,
    generate_answer_stream,
    generate_chat_title,
    generate_direct_response,
    generate_sql_answer_stream,
    generate_sql_query,
    get_schema_for_prompt,
    get_tabular_tables,
    retrieve_chunks,
    rewrite_query,
    validate_citations,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Session management ────────────────────────────────────────────────────

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(
    body: ChatSessionCreate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new chat session."""
    session = ChatSession(
        user_id=uuid.UUID(user_id),
        title=body.title,
        created_at=datetime.now(timezone.utc),
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return ChatSessionResponse.model_validate(session)


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all chat sessions for the authenticated user."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == uuid.UUID(user_id))
        .order_by(ChatSession.created_at.desc())
    )
    sessions = result.scalars().all()
    return [ChatSessionResponse.model_validate(s) for s in sessions]


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(
    session_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a single chat session by ID."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == uuid.UUID(user_id),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return ChatSessionResponse.model_validate(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: uuid.UUID,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a chat session and all its messages."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == uuid.UUID(user_id),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    await db.delete(session)
    await db.commit()
    return None


@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_session(
    session_id: uuid.UUID,
    body: ChatSessionUpdate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a chat session's title."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == uuid.UUID(user_id),
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    session.title = body.title.strip()
    await db.commit()
    await db.refresh(session)
    return ChatSessionResponse.model_validate(session)



@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_messages(
    session_id: uuid.UUID,
    response: Response,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all messages in a chat session in a single database roundtrip."""
    result = await db.execute(
        select(ChatSession)
        .options(joinedload(ChatSession.messages))
        .where(
            ChatSession.id == session_id,
            ChatSession.user_id == uuid.UUID(user_id),
        )
    )
    session = result.unique().scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    if session.title:
        response.headers["X-Session-Title"] = session.title

    return [ChatMessageResponse.model_validate(m) for m in session.messages]


# ── Streaming query endpoint ──────────────────────────────────────────────

@router.post("/sessions/{session_id}/query")
async def query(
    session_id: uuid.UUID,
    body: QueryRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Ask a question and get a streamed answer via SSE.

    SSE events:
    - event: token     → data: {"token": "..."}   (streamed tokens)
    - event: citations → data: {"citations": [...]} (validated citations)
    - event: done      → data: {}                   (stream complete)
    - event: error     → data: {"message": "..."}   (error occurred)
    """
    # Verify session and save user message in a scoped session so the DB connection
    # is immediately returned to the pool and not held open during the SSE stream.
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == uuid.UUID(user_id),
            )
        )
        session_obj = result.scalar_one_or_none()
        if not session_obj:
            raise HTTPException(status_code=404, detail="Chat session not found.")
        current_session_title = session_obj.title or "New conversation"

        # Save the user message
        user_msg = ChatMessage(
            session_id=session_id,
            role="user",
            content=body.question,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user_msg)
        await db.commit()

        # Get chat history for query rewrite and title synthesis
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        all_messages = result.scalars().all()
        chat_history = [
            {"role": m.role, "content": m.content}
            for m in all_messages[:-1]  # Exclude the just-added user message
        ]
        user_message_count = len([m for m in all_messages if m.role == "user"])

    async def _maybe_update_session_title(full_response: str) -> str | None:
        nonlocal current_session_title
        # Auto-name or refine title during the first 3 user messages
        if user_message_count <= 3:
            try:
                new_title = await generate_chat_title(
                    chat_history=chat_history,
                    current_question=body.question,
                    current_answer=full_response,
                )
                if (
                    new_title
                    and new_title != "New conversation"
                    and new_title.strip().lower() != current_session_title.strip().lower()
                ):
                    async with AsyncSessionLocal() as title_db:
                        res = await title_db.execute(
                            select(ChatSession).where(ChatSession.id == session_id)
                        )
                        s_to_update = res.scalar_one_or_none()
                        if s_to_update:
                            s_to_update.title = new_title
                            await title_db.commit()
                            current_session_title = new_title
                            logger.info(f"Updated session {session_id} title: '{new_title}'")
                            return new_title
            except Exception as e:
                logger.warning(f"Failed to auto-generate or save session title: {e}")
        return None

    async def event_stream():
        """SSE event generator."""
        with logfire.span(
            "rag.chat_flow",
            session_id=str(session_id),
            user_id=user_id,
            question=body.question,
        ) as chat_span:
            try:
                # Flush initial SSE comment ping to immediately open stream and disable proxy buffering
                yield ": ping\n\n"
                await asyncio.sleep(0)

                # Step 0: Classify intent — does this need document retrieval?
                intent = await classify_intent(body.question, chat_history)
                chat_span.set_attribute("intent", intent)
                logger.info(f"Query intent: {intent} for '{body.question[:80]}'")

                if intent == "chitchat":
                    chat_span.set_attribute("route", "chitchat")
                    # Direct response — no retrieval needed
                    full_response = ""
                    async for token in generate_direct_response(body.question, chat_history):
                        full_response += token
                        yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

                    # No citations for chitchat
                    yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"

                    # Save assistant message
                    async with AsyncSessionLocal() as save_db:
                        assistant_msg = ChatMessage(
                            session_id=session_id,
                            role="assistant",
                            content=full_response,
                            created_at=datetime.now(timezone.utc),
                        )
                        save_db.add(assistant_msg)
                        await save_db.commit()

                    # Check for title update
                    new_title = await _maybe_update_session_title(full_response)
                    if new_title:
                        yield f"event: title\ndata: {json.dumps({'title': new_title})}\n\n"

                    yield f"event: done\ndata: {{}}\n\n"
                    return

                # Step 1: Analyze Document Scope & Ambiguity across user's library
                async with AsyncSessionLocal() as session:
                    result = await session.execute(
                        select(Document).where(
                            Document.user_id == user_id,
                            Document.status == DocumentStatus.READY,
                        )
                    )
                    ready_docs = [
                        {
                            "id": str(d.id),
                            "filename": d.filename,
                            "file_type": d.file_type,
                            "is_tabular": d.is_tabular,
                            "chunk_count": d.chunk_count,
                        }
                        for d in result.scalars().all()
                    ]

                scope = await analyze_document_scope(
                    question=body.question,
                    chat_history=chat_history,
                    available_documents=ready_docs,
                )
                logger.info(f"Document scope analysis: status={scope.status}, target={scope.target_filename}")

                # 1a. User has no documents uploaded
                if scope.status == "no_documents":
                    chat_span.set_attribute("route", "no_documents")
                    no_docs_msg = scope.clarification_text or "You don't have any uploaded documents yet. Please upload a document to get started."
                    yield f"event: token\ndata: {json.dumps({'token': no_docs_msg})}\n\n"
                    yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
                    async with AsyncSessionLocal() as save_db:
                        assistant_msg = ChatMessage(
                            session_id=session_id,
                            role="assistant",
                            content=no_docs_msg,
                            created_at=datetime.now(timezone.utc),
                        )
                        save_db.add(assistant_msg)
                        await save_db.commit()
                    yield f"event: done\ndata: {{}}\n\n"
                    return

                # 1b. Query is ambiguous across multiple documents — ask for clarification
                if scope.status == "needs_clarification":
                    chat_span.set_attribute("route", "clarification")
                    clarification_msg = scope.clarification_text
                    yield f"event: token\ndata: {json.dumps({'token': clarification_msg})}\n\n"
                    yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"

                    async with AsyncSessionLocal() as save_db:
                        assistant_msg = ChatMessage(
                            session_id=session_id,
                            role="assistant",
                            content=clarification_msg,
                            created_at=datetime.now(timezone.utc),
                        )
                        save_db.add(assistant_msg)
                        await save_db.commit()

                    new_title = await _maybe_update_session_title(clarification_msg)
                    if new_title:
                        yield f"event: title\ndata: {json.dumps({'title': new_title})}\n\n"

                    yield f"event: done\ndata: {{}}\n\n"
                    return

                # Step 2: Rewrite query (using cleaned or effective question)
                effective_q = scope.cleaned_question if scope.cleaned_question else body.question
                rewritten = await rewrite_query(effective_q, chat_history)

                # Step 3: Check if user has tabular data for SQL pipeline
                has_tabular = await check_tabular_data(user_id)
                # Only attempt SQL if target document is tabular or scope is cross_doc
                should_attempt_sql = has_tabular and (scope.is_tabular or scope.status == "all_docs")

                if should_attempt_sql:
                    schema_info = await get_schema_for_prompt(user_id)

                    if schema_info:
                        sql = await generate_sql_query(rewritten, schema_info, chat_history)

                        if sql:
                            try:
                                sql_result = await execute_sql_query(user_id, sql)
                                sql_result_text = format_sql_result(sql_result)

                                # ONLY emit sql_query event after successful execution
                                chat_span.set_attribute("route", "sql")
                                yield f"event: sql_query\ndata: {json.dumps({'sql': sql})}\n\n"

                                # Stream the LLM interpretation of the results
                                full_response = ""
                                async for token in generate_sql_answer_stream(
                                    rewritten, sql, sql_result_text
                                ):
                                    full_response += token
                                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

                                # No vector citations for SQL answers
                                yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"

                                # Save assistant message
                                async with AsyncSessionLocal() as save_db:
                                    assistant_msg = ChatMessage(
                                        session_id=session_id,
                                        role="assistant",
                                        content=full_response,
                                        created_at=datetime.now(timezone.utc),
                                    )
                                    save_db.add(assistant_msg)
                                    await save_db.commit()

                                # Check for title update
                                new_title = await _maybe_update_session_title(full_response)
                                if new_title:
                                    yield f"event: title\ndata: {json.dumps({'title': new_title})}\n\n"

                                yield f"event: done\ndata: {{}}\n\n"
                                return

                            except Exception as sql_err:
                                logger.warning(f"SQL execution failed, falling through to RAG pipeline: {sql_err}")
                                # Fall through to RAG pipeline without emitting sql_query

                # Step 4: RAG path — detect aggregation and retrieve chunks
                chat_span.set_attribute("route", "rag")
                is_agg = detect_aggregation(rewritten)
                chat_span.set_attribute("is_aggregation", is_agg)

                # Scope retrieval to specific document if resolved
                target_doc_filter = scope.target_doc_id if scope.status == "resolved" else None
                chunks = await retrieve_chunks(
                    user_id=user_id,
                    question=rewritten,
                    is_aggregation=is_agg,
                    doc_id_filter=target_doc_filter,
                )

                if not chunks:
                    # No documents or no relevant chunks found
                    no_docs_msg = (
                        "I don't have any documents to search through yet. "
                        "Please upload some documents first, then ask your question again."
                    )
                    yield f"event: token\ndata: {json.dumps({'token': no_docs_msg})}\n\n"

                    # Save assistant message
                    async with AsyncSessionLocal() as save_db:
                        assistant_msg = ChatMessage(
                            session_id=session_id,
                            role="assistant",
                            content=no_docs_msg,
                            created_at=datetime.now(timezone.utc),
                        )
                        save_db.add(assistant_msg)
                        await save_db.commit()

                    yield f"event: done\ndata: {{}}\n\n"
                    return

                # Step 5: Stream the answer
                full_response = ""
                async for token in generate_answer_stream(rewritten, chunks):
                    full_response += token
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

                # Step 6: Validate citations
                citations = validate_citations(full_response, chunks)
                chat_span.set_attribute("citations_count", len(citations))
                yield f"event: citations\ndata: {json.dumps({'citations': citations})}\n\n"

                # Save assistant message with citations
                async with AsyncSessionLocal() as save_db:
                    assistant_msg = ChatMessage(
                        session_id=session_id,
                        role="assistant",
                        content=full_response,
                        citations=citations if citations else None,
                        created_at=datetime.now(timezone.utc),
                    )
                    save_db.add(assistant_msg)
                    await save_db.commit()

                # Step 7: Auto-name / refine chat title
                new_title = await _maybe_update_session_title(full_response)
                if new_title:
                    yield f"event: title\ndata: {json.dumps({'title': new_title})}\n\n"

                yield f"event: done\ndata: {{}}\n\n"

            except Exception as e:
                logger.exception("Error during query processing")
                chat_span.record_exception(e)
                error_msg = "Something went wrong generating a response. Please try again in a moment."
                yield f"event: error\ndata: {json.dumps({'message': error_msg})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
        },
    )
