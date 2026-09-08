"""
Chat endpoints: session management and streaming Q&A.

The query endpoint uses SSE (Server-Sent Events) to stream
the LLM response token-by-token to the frontend.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.auth import get_current_user
from app.database import AsyncSessionLocal, get_db
from app.models import ChatMessage, ChatSession
from app.schemas import (
    ChatMessageResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    QueryRequest,
)
from app.services.query import (
    check_tabular_data,
    classify_intent,
    detect_aggregation,
    execute_sql_query,
    format_sql_result,
    generate_answer_stream,
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
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Chat session not found.")

        # Save the user message
        user_msg = ChatMessage(
            session_id=session_id,
            role="user",
            content=body.question,
            created_at=datetime.now(timezone.utc),
        )
        db.add(user_msg)
        await db.commit()

        # Get chat history for query rewrite
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

    async def event_stream():
        """SSE event generator."""
        try:
            # Step 0: Classify intent — does this need document retrieval?
            intent = await classify_intent(body.question, chat_history)
            logger.info(f"Query intent: {intent} for '{body.question[:80]}'")

            if intent == "chitchat":
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

                yield f"event: done\ndata: {{}}\n\n"
                return

            # Step 1: Rewrite query (for retrieve intent)
            rewritten = await rewrite_query(body.question, chat_history)

            # Step 2: Check if user has tabular data for SQL pipeline
            has_tabular = await check_tabular_data(user_id)

            if has_tabular:
                # ── Text-to-SQL path ──────────────────────────────
                schema_info = await get_schema_for_prompt(user_id)

                if schema_info:
                    sql = await generate_sql_query(rewritten, schema_info, chat_history)

                    # If model didn't formulate custom SQL, provide an overview preview query
                    # so tabular datasets are always analyzed rather than falling back to unrelated PDFs
                    if not sql:
                        tables = await get_tabular_tables(user_id)
                        if tables:
                            sql = f"SELECT * FROM {tables[0]} LIMIT 15"
                            logger.info(f"Using tabular overview preview query: {sql}")

                    if sql:
                        # Send the SQL query to the frontend for transparency
                        yield f"event: sql_query\ndata: {json.dumps({'sql': sql})}\n\n"

                        try:
                            sql_result = await execute_sql_query(user_id, sql)
                            sql_result_text = format_sql_result(sql_result)

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

                            yield f"event: done\ndata: {{}}\n\n"
                            return

                        except ValueError as sql_err:
                            logger.warning(f"SQL execution failed, attempting fallback preview: {sql_err}")
                            tables = await get_tabular_tables(user_id)
                            preview_sql = f"SELECT * FROM {tables[0]} LIMIT 15" if tables else None
                            if preview_sql and sql != preview_sql:
                                try:
                                    fb_res = await execute_sql_query(user_id, preview_sql)
                                    fb_text = format_sql_result(fb_res)
                                    yield f"event: sql_query\ndata: {json.dumps({'sql': preview_sql})}\n\n"
                                    full_response = ""
                                    async for token in generate_sql_answer_stream(
                                        rewritten, preview_sql, fb_text
                                    ):
                                        full_response += token
                                        yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"

                                    yield f"event: citations\ndata: {json.dumps({'citations': []})}\n\n"
                                    async with AsyncSessionLocal() as save_db:
                                        assistant_msg = ChatMessage(
                                            session_id=session_id,
                                            role="assistant",
                                            content=full_response,
                                            created_at=datetime.now(timezone.utc),
                                        )
                                        save_db.add(assistant_msg)
                                        await save_db.commit()

                                    yield f"event: done\ndata: {{}}\n\n"
                                    return
                                except Exception as fb_err:
                                    logger.warning(f"Fallback preview failed: {fb_err}")
                            # Fall through to RAG pipeline below

            # Step 3: RAG path — detect aggregation
            is_agg = detect_aggregation(rewritten)

            # Step 4: Retrieve chunks
            chunks = await retrieve_chunks(user_id, rewritten, is_agg)

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

            yield f"event: done\ndata: {{}}\n\n"

        except Exception as e:
            logger.exception("Error during query processing")
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
