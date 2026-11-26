"""
Chat endpoints: session management and streaming Q&A.

The query endpoint uses SSE (Server-Sent Events) to stream the LLM response token-by-token to the frontend.
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
    classify_query,
    generate_answer_stream,
    generate_chat_title,
    generate_direct_response,
    retrieve_chunks,
    rewrite_query,
    validate_citations,
)
from app.services.doc_profiles import attach_document_profiles
from app.services.tabular_qa import (
    attach_tabular_profiles,
    check_tabular_data,
    plan_tabular_answer,
    stream_tabular_answer,
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
    with logfire.span("💬 Create Chat Session | '{title}'", title=body.title, user_id=user_id) as span:
        session = ChatSession(
            user_id=uuid.UUID(user_id),
            title=body.title,
            created_at=datetime.now(timezone.utc),
        )
        db.add(session)
        await db.commit()
        await db.refresh(session)
        span.set_attribute("session_id", str(session.id))
        logfire.info("Created session {session_id} for user {user_id}", session_id=str(session.id), user_id=user_id)
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
    with logfire.span("🗑️ Delete Chat Session | {session_id}", session_id=str(session_id), user_id=user_id) as span:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == uuid.UUID(user_id),
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            span.set_attribute("found", False)
            raise HTTPException(status_code=404, detail="Chat session not found.")

        await db.delete(session)
        await db.commit()
        logfire.info("Deleted session {session_id} for user {user_id}", session_id=str(session_id), user_id=user_id)
        return None


@router.patch("/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_session(
    session_id: uuid.UUID,
    body: ChatSessionUpdate,
    user_id: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update a chat session's title."""
    with logfire.span("✏️ Update Session Title | {session_id}", session_id=str(session_id), title=body.title.strip(), user_id=user_id) as span:
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == session_id,
                ChatSession.user_id == uuid.UUID(user_id),
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            span.set_attribute("found", False)
            raise HTTPException(status_code=404, detail="Chat session not found.")

        session.title = body.title.strip()
        await db.commit()
        await db.refresh(session)
        logfire.info("Updated title for session {session_id} to '{title}'", session_id=str(session_id), title=session.title)
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
        return []

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
            session_obj = ChatSession(
                id=session_id,
                user_id=uuid.UUID(user_id),
                title="New conversation",
            )
            db.add(session_obj)
            await db.commit()
            logger.info(f"Auto-created session {session_id} for user {user_id[:8]}")
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

        # Get chat history for query rewrite, scope analysis, and title synthesis
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        all_messages = result.scalars().all()
        chat_history = [
            {"role": m.role, "content": m.content, "citations": m.citations}
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

    async def _save_assistant_message(
        content: str,
        citations: list[dict] | None = None,
    ) -> None:
        """Persist an assistant message and optional citations to the database."""
        async with AsyncSessionLocal() as save_db:
            assistant_msg = ChatMessage(
                session_id=session_id,
                role="assistant",
                content=content,
                citations=citations if citations else None,
                created_at=datetime.now(timezone.utc),
            )
            save_db.add(assistant_msg)
            await save_db.commit()

    async def event_stream():
        """SSE event generator."""
        with logfire.span(
            "💬 Chat Stream | Session {session_short} | '{question_short}'",
            session_short=str(session_id)[:8],
            question_short=body.question[:50],
            session_id=str(session_id),
            user_id=user_id,
            question=body.question,
        ) as chat_span:
            try:
                # Flush initial SSE comment ping to immediately open stream and disable proxy buffering
                yield ": ping\n\n"
                await asyncio.sleep(0)

                # Step 0: Classify the query — chitchat or retrieval, how much of the
                # document it needs (scope), and whether it aggregates over many records.
                query_class = await classify_query(body.question, chat_history)
                intent = query_class.intent
                chat_span.set_attribute("intent", intent)
                chat_span.set_attribute("query_scope", query_class.scope)
                chat_span.set_attribute("is_aggregation", query_class.aggregation)
                logfire.info("🎯 Query: {intent} / {scope} / aggregation={agg} for '{question}'", intent=intent,
                             scope=query_class.scope, agg=query_class.aggregation, question=body.question[:60],
                             session_id=str(session_id))

                if intent == "chitchat":
                    chat_span.set_attribute("route", "chitchat")
                    logfire.info("💬 Route: direct conversational response", session_id=str(session_id))
                    # Direct response — no retrieval needed
                    full_response = ""
                    async for token in generate_direct_response(body.question, chat_history):
                        full_response += token
                        yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                        await asyncio.sleep(0)

                    # No citations for chitchat
                    yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"

                    # Save assistant message
                    await _save_assistant_message(full_response)

                    # Check for title update
                    new_title = await _maybe_update_session_title(full_response)
                    if new_title:
                        logfire.info("🏷️ Session Title: '{title}'", title=new_title, session_id=str(session_id))
                        yield f"event: title\ndata: {json.dumps({'title': new_title})}\n\n"

                    logfire.info("✅ Finished chitchat stream ({length} chars)", length=len(full_response), session_id=str(session_id))
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
                await attach_tabular_profiles(user_id, ready_docs)
                await attach_document_profiles(user_id, ready_docs)

                scope = await analyze_document_scope(
                    question=body.question,
                    chat_history=chat_history,
                    available_documents=ready_docs,
                )
                logger.info(f"Document scope analysis: status={scope.status}, target={scope.target_filename}")

                # 1a. User has no documents uploaded
                if scope.status == "no_documents":
                    chat_span.set_attribute("route", "no_documents")
                    logfire.info("⚠️ Route: user has no ready documents", user_id=user_id, session_id=str(session_id))
                    no_docs_msg = scope.clarification_text or "You don't have any uploaded documents yet. Please upload a document to get started."
                    yield f"event: token\ndata: {json.dumps({'token': no_docs_msg})}\n\n"
                    yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
                    # Save assistant message
                    await _save_assistant_message(no_docs_msg)
                    yield f"event: done\ndata: {{}}\n\n"
                    return

                # 1b. Query is ambiguous across multiple documents — ask for clarification
                if scope.status == "needs_clarification":
                    chat_span.set_attribute("route", "clarification")
                    logfire.info("❓ Route: ambiguous query across multiple documents — asking for clarification", session_id=str(session_id))
                    clarification_msg = scope.clarification_text
                    yield f"event: token\ndata: {json.dumps({'token': clarification_msg})}\n\n"
                    yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"

                    # Save assistant message
                    await _save_assistant_message(clarification_msg)

                    new_title = await _maybe_update_session_title(clarification_msg)
                    if new_title:
                        logfire.info("🏷️ Session Title: '{title}'", title=new_title, session_id=str(session_id))
                        yield f"event: title\ndata: {json.dumps({'title': new_title})}\n\n"

                    yield f"event: done\ndata: {{}}\n\n"
                    return

                # Step 2: Rewrite query (using cleaned or effective question)
                effective_q = scope.cleaned_question if scope.cleaned_question else body.question
                rewritten = await rewrite_query(effective_q, chat_history)

                # Step 3: Text-to-SQL — ONLY when scope analysis resolved to a spreadsheet.
                # SQL is never attempted for PDFs or text files, even if the user also has
                # spreadsheets. A spreadsheet never falls through to vector retrieval.
                if scope.is_tabular and await check_tabular_data(user_id):
                    target_doc_uuid = uuid.UUID(scope.target_doc_id) if scope.target_doc_id else None
                    plan = await plan_tabular_answer(
                        user_id, rewritten, target_doc_uuid, query_class, chat_history
                    )
                    if plan is not None:
                        chat_span.set_attribute("route", plan.route)
                        chat_span.set_attribute("sql_attempts", plan.attempts)
                        if plan.sql_error:
                            chat_span.set_attribute("sql_error", plan.sql_error)
                        if plan.route == "sql":
                            logfire.info("🦆 Route: Executed DuckDB SQL: {sql}", sql=plan.sql, session_id=str(session_id))
                            yield f"event: sql_query\ndata: {json.dumps({'sql': plan.sql})}\n\n"
                        else:
                            logfire.info("📊 Route: Tabular overview for '{filename}'", filename=scope.target_filename or "Spreadsheet", session_id=str(session_id))

                        full_response = ""
                        async for token in stream_tabular_answer(plan, rewritten, scope.target_filename or ""):
                            full_response += token
                            yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                            await asyncio.sleep(0)

                        # No vector citations for spreadsheet answers
                        yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
                        await _save_assistant_message(full_response)

                        new_title = await _maybe_update_session_title(full_response)
                        if new_title:
                            logfire.info("🏷️ Session Title: '{title}'", title=new_title, session_id=str(session_id))
                            yield f"event: title\ndata: {json.dumps({'title': new_title})}\n\n"

                        logfire.info("✅ Spreadsheet answer completed ({length} chars)", length=len(full_response), session_id=str(session_id))
                        yield f"event: done\ndata: {{}}\n\n"
                        return

                # Step 4: RAG path — retrieve chunks (whole-document questions also get summary chunks)
                chat_span.set_attribute("route", "rag")

                # Scope retrieval to specific document if resolved
                target_doc_filter = scope.target_doc_id if scope.status == "resolved" else None
                target_filename = scope.target_filename if scope.status == "resolved" else None
                chunks = await retrieve_chunks(
                    user_id=user_id,
                    question=rewritten,
                    is_overview=query_class.is_overview,
                    doc_id_filter=target_doc_filter,
                    filename_filter=target_filename,
                )
                logfire.info("🔍 Route: RAG retrieval found {chunk_count} chunks", chunk_count=len(chunks), session_id=str(session_id))

                if not chunks:
                    # No relevant chunks found in the searched scope
                    if scope.target_filename:
                        no_docs_msg = (
                            f"I searched '{scope.target_filename}', but couldn't find any relevant sections "
                            "answering your question. Please try asking about a specific topic from the document or rephrasing."
                        )
                    elif ready_docs:
                        no_docs_msg = (
                            "I searched through your uploaded documents, but couldn't find any relevant sections "
                            "answering your question. Please try rephrasing your question or asking about a specific topic."
                        )
                    else:
                        no_docs_msg = (
                            "I don't have any documents to search through yet. "
                            "Please upload some documents first, then ask your question again."
                        )
                    yield f"event: token\ndata: {json.dumps({'token': no_docs_msg})}\n\n"
                    yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"

                    # Save assistant message
                    await _save_assistant_message(no_docs_msg)

                    yield f"event: done\ndata: {{}}\n\n"
                    return

                # Step 5: Stream the answer
                full_response = ""
                async for token in generate_answer_stream(rewritten, chunks):
                    full_response += token
                    yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0)

                # Step 6: Validate citations (suppress citations if answer is an explicit refusal)
                if "I don't have enough information in the uploaded documents" in full_response:
                    citations = []
                else:
                    citations = validate_citations(full_response, chunks)
                chat_span.set_attribute("citations_count", len(citations))
                yield f"event: citations\ndata: {json.dumps({'citations': citations})}\n\n"

                # Save assistant message with citations
                await _save_assistant_message(full_response, citations=citations)

                # Step 7: Auto-name / refine chat title
                new_title = await _maybe_update_session_title(full_response)
                if new_title:
                    logfire.info("🏷️ Session Title: '{title}'", title=new_title, session_id=str(session_id))
                    yield f"event: title\ndata: {json.dumps({'title': new_title})}\n\n"

                logfire.info("✅ RAG response stream completed ({length} chars, {citations} citations)", length=len(full_response), citations=len(citations), session_id=str(session_id))
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
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering
        },
    )
