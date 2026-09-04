"""
Query pipeline: classify intent → rewrite → detect aggregation → retrieve → generate → validate citations.

This is the core RAG logic. The LLM first classifies the user's intent to decide
whether document retrieval is needed. Chitchat/greetings skip retrieval entirely.
Implemented as plain Python functions — no LangChain/LangGraph.
"""

import json
import logging
import re
from typing import AsyncGenerator
from uuid import UUID

from groq import AsyncGroq

from app.config import settings
from app.services import embedding, vector_store

logger = logging.getLogger(__name__)

# Groq async client
_groq_client: AsyncGroq | None = None


def _get_groq() -> AsyncGroq:
    """Get or create the async Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


# ── Intent classification ─────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a document Q&A assistant. 
Given the user's message, classify it into exactly one category.

Categories:
- "retrieve": The user is asking a question that requires searching through uploaded documents (e.g. factual questions, analysis requests, summarization, questions about content).
- "chitchat": The user is making casual conversation, greeting, saying thanks, or asking something that clearly does NOT require document lookup (e.g. "Hi", "Hello", "Thanks!", "How are you?", "What can you do?", "Who are you?").

Respond with ONLY the category name — one word, no quotes, no explanation."""


async def classify_intent(
    question: str,
    chat_history: list[dict] | None = None,
) -> str:
    """
    Use a fast LLM call to classify whether the user's message needs
    document retrieval or is just chitchat/greeting.

    Returns: "retrieve" or "chitchat"
    """
    # Fast heuristic for obvious greetings / conversational pleasantries
    cleaned = question.strip().lower()
    cleaned = re.sub(r"[^\w\s]", "", cleaned)
    quick_chitchat = {
        "hi", "hello", "hey", "hola", "howdy",
        "good morning", "good afternoon", "good evening",
        "how are you", "how are you doing", "whats up", "what's up",
        "who are you", "what can you do", "help",
        "thanks", "thank you", "thx", "bye", "goodbye",
    }
    if cleaned in quick_chitchat:
        logger.info(f"Intent classified (heuristic): '{question}' → chitchat")
        return "chitchat"

    groq = _get_groq()

    messages: list[dict] = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
    ]

    # Include last 2 messages for context (helps with follow-ups like "thanks")
    if chat_history:
        for m in chat_history[-2:]:
            messages.append({
                "role": m["role"],
                "content": m["content"][:150],
            })

    messages.append({"role": "user", "content": question})

    try:
        response = await groq.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            max_tokens=150,
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip().lower()
        reasoning = (getattr(response.choices[0].message, "reasoning", None) or "").strip().lower()

        # Check content first
        if "chitchat" in content:
            logger.info(f"Intent classified: '{question}' → chitchat")
            return "chitchat"
        if "retrieve" in content:
            logger.info(f"Intent classified: '{question}' → retrieve")
            return "retrieve"

        # Fallback to reasoning if content was in reasoning
        if "chitchat" in reasoning:
            logger.info(f"Intent classified (from reasoning): '{question}' → chitchat")
            return "chitchat"
        if "retrieve" in reasoning:
            logger.info(f"Intent classified (from reasoning): '{question}' → retrieve")
            return "retrieve"

        # If the LLM returns something unexpected, default to retrieve
        logger.warning(f"Unknown intent '{content}' for '{question}', defaulting to retrieve")
        return "retrieve"

    except Exception as e:
        logger.warning(f"Intent classification failed: {e}, defaulting to retrieve")
        return "retrieve"


# ── Direct response (no retrieval) ────────────────────────────────────────

CHITCHAT_SYSTEM_PROMPT = """You are DocuChat, a friendly AI document assistant. The user is having a casual conversation — they are NOT asking about their documents right now.

Respond naturally and warmly. Keep it brief (1-3 sentences). You can:
- Greet them back
- Explain what you can help with (searching uploaded documents, answering questions about their files, summarizing content)
- Thank them back
- Be helpful and conversational

Do NOT mention document chunks, pages, retrieval, or any technical details. Just be a friendly assistant."""


async def generate_direct_response(
    question: str,
    chat_history: list[dict] | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream a direct conversational response without document retrieval.
    Used for chitchat, greetings, and meta-questions about the assistant.
    """
    groq = _get_groq()

    messages: list[dict] = [
        {"role": "system", "content": CHITCHAT_SYSTEM_PROMPT},
    ]

    # Include recent history for natural conversation flow
    if chat_history:
        for m in chat_history[-4:]:
            messages.append({
                "role": m["role"],
                "content": m["content"][:300],
            })

    messages.append({"role": "user", "content": question})

    stream = await groq.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=200,
        temperature=0.7,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ── Aggregation keywords ──────────────────────────────────────────────────

AGGREGATION_KEYWORDS = {
    "total", "sum", "average", "mean", "how many", "count",
    "minimum", "maximum", "min", "max", "aggregate", "overall",
    "combined", "altogether", "in total", "grand total",
}


def detect_aggregation(question: str) -> bool:
    """
    Check if a question contains aggregation language.
    Used to bias retrieval toward including summary chunks.
    """
    lower_q = question.lower()
    return any(kw in lower_q for kw in AGGREGATION_KEYWORDS)


# ── Query rewrite ─────────────────────────────────────────────────────────

async def rewrite_query(
    question: str,
    chat_history: list[dict],
) -> str:
    """
    Rewrite a question to be self-contained by resolving references
    to prior conversation turns.

    If the question is already self-contained, returns it unchanged.
    Uses a lightweight Groq call with a short system prompt.
    """
    if not chat_history:
        return question

    # Build a condensed history string (last 5 messages)
    recent = chat_history[-5:]
    history_str = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
        for m in recent
    )

    groq = _get_groq()

    try:
        response = await groq.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query rewriter. Given a conversation history and a "
                        "follow-up question, rewrite the question to be self-contained "
                        "(resolve pronouns, references to 'that', 'it', 'those', etc.). "
                        "Output ONLY the rewritten question, nothing else. "
                        "If the question is already self-contained, return it unchanged."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_str}\n\n"
                        f"Follow-up question: {question}\n\n"
                        f"Rewritten question:"
                    ),
                },
            ],
            max_tokens=200,
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten:
            logger.info(f"Query rewritten: '{question}' → '{rewritten}'")
            return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")

    return question


# ── Retrieve chunks ───────────────────────────────────────────────────────

async def retrieve_chunks(
    user_id: str,
    question: str,
    is_aggregation: bool,
) -> list[dict]:
    """
    Retrieve relevant chunks for answering the question.

    1. Embed the query
    2. Search Qdrant for top-K similar chunks
    3. If aggregation detected, also fetch summary chunks and prepend them
    """
    # Embed the query
    query_vector = embedding.embed_query(question)

    # Dense retrieval — top 5 chunks
    results = await vector_store.search(
        user_id=user_id,
        query_vector=query_vector,
        limit=5,
    )

    # If aggregation detected, fetch and prepend up to 3 summary chunks
    if is_aggregation:
        summary_chunks = await vector_store.get_summary_chunks(user_id)
        if summary_chunks:
            # Deduplicate: remove any summary chunks already in results
            result_ids = {r["id"] for r in results}
            new_summaries = [s for s in summary_chunks if s["id"] not in result_ids][:3]
            # Prepend summaries so they appear first in context
            results = new_summaries + results
            logger.info(f"Added {len(new_summaries)} summary chunks for aggregation query")

    return results


# ── System prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful document Q&A assistant. Your job is to answer questions based ONLY on the provided context chunks from the user's uploaded documents.

RULES:
1. Answer ONLY using information from the provided context chunks.
2. If the answer is not in the context, say: "I don't have enough information in the uploaded documents to answer this question."
3. When citing information, ALWAYS cite the page number or row range in square brackets, for example: [Page 4] or [Page 12]. For spreadsheets, cite the row range like [Rows 1-50]. For summary chunks, cite [Summary].
   CRITICAL: NEVER output chunk UUIDs, IDs, or write [CHUNK ...]. Always use the human-readable [Page X] or [Summary] tag.
4. Format your answers in clean, beautiful Markdown:
   - Use bold (**text**) for important terms and subheadings.
   - Use bullet points (* or -) or numbered lists for structure.
   - Use Markdown headings (e.g. ### Section Name) to organize long responses.
   - Place citations like [Page 3] directly after the relevant sentence or bullet point.
5. Do NOT make up, hallucinate, or infer information that is not explicitly stated in the context.
6. Be concise, accurate, and professional."""


def _build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM prompt."""
    context_parts = []
    for chunk in chunks:
        filename = chunk.get("filename", "unknown")
        chunk_type = chunk.get("chunk_type", "text")
        content = chunk.get("content", "")

        # Build clean human-friendly label for citation
        if chunk.get("page_number"):
            tag = f"Page {chunk['page_number']}"
        elif chunk.get("row_range_start"):
            tag = f"Rows {chunk['row_range_start']}-{chunk.get('row_range_end', '?')}"
        elif chunk_type == "summary":
            tag = "Summary"
        else:
            tag = f"Document: {filename}"

        # Do NOT include raw chunk_id in LLM context so LLM never cites chunk UUIDs
        context_parts.append(f"[{tag}] (Source: {filename})\n{content}")

    return "\n\n---\n\n".join(context_parts)


# ── Generate answer (streaming) ──────────────────────────────────────────

async def generate_answer_stream(
    question: str,
    chunks: list[dict],
) -> AsyncGenerator[str, None]:
    """
    Stream the LLM answer token-by-token.

    Yields individual tokens as they arrive from Groq.
    The caller is responsible for accumulating the full response
    for citation validation.
    """
    context = _build_context(chunks)

    groq = _get_groq()

    stream = await groq.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context chunks:\n\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        max_tokens=1000,
        temperature=0.1,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


# ── Citation validation ───────────────────────────────────────────────────

def validate_citations(
    full_response: str,
    context_chunks: list[dict],
) -> list[dict]:
    """
    Extract and validate citation references from the LLM response.
    Supports [Page X], [Rows X-Y], [Summary], and [CHUNK <id>].
    """
    valid_citations = []
    seen = set()

    for chunk in context_chunks:
        cid = chunk["id"]
        page = chunk.get("page_number")
        rows = chunk.get("row_range_start")
        fname = chunk.get("filename", "")

        is_cited = False
        # Match [Page X] or (Page X) or 【Page X】 or Page X
        if page is not None and re.search(rf'(?:\[|【|\()?\s*Page\s+{page}\b', full_response, re.IGNORECASE):
            is_cited = True
        # Match [Rows X-Y] or Rows X-Y
        elif rows is not None and re.search(rf'(?:\[|【|\()?\s*Rows?\s+{rows}\b', full_response, re.IGNORECASE):
            is_cited = True
        # Match [Summary]
        elif chunk.get("chunk_type") == "summary" and re.search(r'(?:\[|【|\()?\s*Summary\b', full_response, re.IGNORECASE):
            is_cited = True
        # Match legacy [CHUNK <id>]
        elif cid in full_response:
            is_cited = True

        if is_cited and cid not in seen:
            seen.add(cid)
            content = chunk.get("content", "")
            valid_citations.append({
                "chunk_id": cid,
                "filename": chunk.get("filename", "unknown"),
                "page_number": page,
                "row_range_start": chunk.get("row_range_start"),
                "row_range_end": chunk.get("row_range_end"),
                "content_preview": content[:200] + ("..." if len(content) > 200 else ""),
            })

    # If no explicit page tags matched in text but chunks were used, include top chunks
    if not valid_citations and context_chunks:
        for chunk in context_chunks[:3]:
            content = chunk.get("content", "")
            valid_citations.append({
                "chunk_id": chunk["id"],
                "filename": chunk.get("filename", "unknown"),
                "page_number": chunk.get("page_number"),
                "row_range_start": chunk.get("row_range_start"),
                "row_range_end": chunk.get("row_range_end"),
                "content_preview": content[:200] + ("..." if len(content) > 200 else ""),
            })

    return valid_citations
