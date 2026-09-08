"""
Query pipeline: classify intent → rewrite → detect aggregation → retrieve → generate → validate citations.

This is the core RAG logic. The LLM first classifies the user's intent to decide
whether document retrieval is needed. Chitchat/greetings skip retrieval entirely.
Implemented as plain Python functions — no LangChain/LangGraph.
"""

import json
import logging
import re
from typing import Any, AsyncGenerator
from uuid import UUID

from groq import AsyncGroq, APIError, APIStatusError, RateLimitError
import logfire

from app.config import settings
from app.services import embedding, vector_store, reranker
from app.services.chunking import count_tokens
from app.services import tabular_store

logger = logging.getLogger(__name__)

# Groq async client
_groq_client: AsyncGroq | None = None


def _get_groq() -> AsyncGroq:
    """Get or create the async Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


# ── Resilient Multi-Model Fallback Chain ──────────────────────────────────
# The USP of this project: when one model hits rate limits (429/TPM) or context limits (413),
# the system automatically and transparently switches to the next available model.
DEFAULT_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "qwen/qwen3.6-27b",
]

# Fast models for utility tasks (intent classification, query rewriting, SQL generation)
# These prioritize strict instruction-following, zero-preamble, and low latency
FAST_MODEL_CHAIN = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
]


def get_model_chain(fast: bool = False) -> list[str]:
    """
    Get the ordered list of models to try.
    If fast=True, uses fast instruction-following models for utility tasks.
    Otherwise starts with the configured settings.groq_model, then chains through
    distinct candidate models in DEFAULT_MODEL_FALLBACKS.
    """
    if fast:
        return list(FAST_MODEL_CHAIN)

    configured = settings.groq_model
    chain = [configured]
    for m in DEFAULT_MODEL_FALLBACKS:
        if m not in chain and m != configured:
            chain.append(m)
    return chain


def _is_rate_limit_or_recoverable(err: Exception) -> bool:
    """Check if an error is a rate limit, capacity, or recoverable status code."""
    status_code = getattr(err, "status_code", None)
    err_msg = str(err).lower()
    return (
        isinstance(err, (RateLimitError, APIStatusError, APIError))
        and (
            status_code in (413, 429, 500, 502, 503, 504)
            or "rate limit" in err_msg
            or "tokens per minute" in err_msg
            or "tpm" in err_msg
            or "too large" in err_msg
            or "decommissioned" in err_msg
            or "capacity" in err_msg
        )
    ) or (
        "rate limit" in err_msg
        or "tokens per minute" in err_msg
        or "tpm" in err_msg
        or "too large" in err_msg
    )


async def call_llm_with_fallback(
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
    fast: bool = False,
) -> Any:
    """
    Execute a non-streaming LLM call with automatic multi-model fallback.
    Tries each model in the fallback chain if rate-limited (429), token limit exceeded (413),
    or on temporary API errors.
    """
    groq = _get_groq()
    models = get_model_chain(fast=fast)
    last_err = None

    for model in models:
        try:
            logger.info(f"Invoking LLM with model: {model}")
            with logfire.span("llm.call", model=model, fast=fast, max_tokens=max_tokens) as span:
                response = await groq.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                usage = getattr(response, "usage", None)
                if usage:
                    span.set_attribute("prompt_tokens", usage.prompt_tokens)
                    span.set_attribute("completion_tokens", usage.completion_tokens)
                    span.set_attribute("total_tokens", usage.total_tokens)
                return response
        except Exception as err:
            if _is_rate_limit_or_recoverable(err):
                logger.warning(
                    f"Model '{model}' failed ({type(err).__name__}: {err}). "
                    f"Auto-switching to next model in fallback chain..."
                )
                last_err = err
                continue
            else:
                logger.error(f"Unrecoverable error on model '{model}': {err}")
                raise

    logger.error("All fallback models exhausted for non-streaming call.")
    if last_err:
        raise last_err
    raise RuntimeError("All models in the fallback chain failed.")


async def stream_llm_with_fallback(
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.1,
    fast: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Execute a streaming LLM call with automatic multi-model fallback.
    If the initial model is rate-limited or fails before stream starts,
    it automatically catches the error and switches to the next available model.
    """
    groq = _get_groq()
    models = get_model_chain(fast=fast)
    last_err = None

    for model in models:
        started = False
        try:
            logger.info(f"Starting LLM stream with model: {model}")
            with logfire.span("llm.stream", model=model, fast=fast, max_tokens=max_tokens) as span:
                stream = await groq.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                token_count = 0
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        started = True
                        token_count += 1
                        yield delta.content
                span.set_attribute("streamed_tokens", token_count)
                return  # Stream completed successfully

        except Exception as err:
            if _is_rate_limit_or_recoverable(err) and not started:
                logger.warning(
                    f"Model '{model}' stream failed ({type(err).__name__}: {err}). "
                    f"Auto-switching to next model in fallback chain..."
                )
                last_err = err
                continue
            else:
                logger.error(f"Error during stream with model '{model}': {err}")
                raise

    logger.error("All fallback models exhausted for streaming.")
    if last_err:
        raise last_err
    raise RuntimeError("All models in the fallback chain failed.")



# ── Intent classification ─────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """You are an intent classifier for a document Q&A assistant where users upload documents and ask questions about them.

Given the user's message, classify it into exactly one category:

- "retrieve": The user is asking ANY question that COULD be answered using uploaded documents. This includes factual questions, analysis requests, summarization, technical questions, questions about specific topics, or anything that might relate to document content. When in doubt, choose "retrieve".
- "chitchat": The user is making pure social conversation — greetings, thanks, or meta-questions about the assistant itself. Examples: "Hi", "Hello", "Thanks!", "How are you?", "What can you do?", "Who are you?"

IMPORTANT: If the user asks about ANY topic (python, finance, science, history, etc.), classify as "retrieve" because their uploaded documents may contain relevant information. Only classify as "chitchat" for obvious greetings and pleasantries.

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
    with logfire.span("rag.classify_intent", question=question) as span:
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
            span.set_attribute("intent", "chitchat")
            span.set_attribute("method", "heuristic")
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
            response = await call_llm_with_fallback(
                messages=messages,
                max_tokens=150,
                temperature=0.0,
                fast=True,
            )
            content = (response.choices[0].message.content or "").strip().lower()
            reasoning = (getattr(response.choices[0].message, "reasoning", None) or "").strip().lower()

            # Check content first
            if "chitchat" in content:
                logger.info(f"Intent classified: '{question}' → chitchat")
                span.set_attribute("intent", "chitchat")
                span.set_attribute("method", "llm")
                return "chitchat"
            if "retrieve" in content:
                logger.info(f"Intent classified: '{question}' → retrieve")
                span.set_attribute("intent", "retrieve")
                span.set_attribute("method", "llm")
                return "retrieve"

            # Fallback to reasoning if content was in reasoning
            if "chitchat" in reasoning:
                logger.info(f"Intent classified (from reasoning): '{question}' → chitchat")
                span.set_attribute("intent", "chitchat")
                span.set_attribute("method", "llm_reasoning")
                return "chitchat"
            if "retrieve" in reasoning:
                logger.info(f"Intent classified (from reasoning): '{question}' → retrieve")
                span.set_attribute("intent", "retrieve")
                span.set_attribute("method", "llm_reasoning")
                return "retrieve"

            # If the LLM returns something unexpected, default to retrieve
            logger.warning(f"Unknown intent '{content}' for '{question}', defaulting to retrieve")
            span.set_attribute("intent", "retrieve")
            span.set_attribute("method", "fallback_unknown")
            return "retrieve"

        except Exception as e:
            logger.warning(f"Intent classification failed: {e}, defaulting to retrieve")
            span.set_attribute("intent", "retrieve")
            span.set_attribute("method", "fallback_error")
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

    async for token in stream_llm_with_fallback(
        messages=messages,
        max_tokens=200,
        temperature=0.7,
    ):
        yield token


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

REWRITE_SYSTEM_PROMPT = """You are a search query rewriter for a document and tabular data assistant.
Given a conversation history and a user's follow-up message, rewrite it into a single standalone search question (maximum 20 words).

CRITICAL RULES:
1. Output ONLY the rewritten question. Never output an answer, explanation, essay, bullet list, or markdown headers.
2. If the user says "continue", "more", "tell me more", "go on", "elaborate": rewrite it into a question asking for deeper insights, key factors, or metrics about the previous topic (e.g. "What are more details and key factors behind [topic]?").
3. Strictly maximum 1 sentence under 20 words.
4. If the question is already self-contained, return it unchanged."""


async def rewrite_query(
    question: str,
    chat_history: list[dict],
) -> str:
    """
    Rewrite a question to be self-contained by resolving references to prior conversation turns.

    If the question is already self-contained, returns it unchanged.
    Uses a fast LLM call and enforces a strict single-sentence length guard.
    """
    if not chat_history:
        return question

    with logfire.span("rag.rewrite_query", original_question=question, history_turns=len(chat_history)) as span:
        # Build a condensed history string (last 5 messages)
        recent = chat_history[-5:]
        history_str = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
            for m in recent
        )

        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Conversation history:\n{history_str}\n\n"
                    f"Follow-up question: {question}\n\n"
                    f"Rewritten question:"
                ),
            },
        ]

        try:
            response = await call_llm_with_fallback(
                messages=messages,
                max_tokens=60,
                temperature=0.0,
                fast=True,
            )
            content = (response.choices[0].message.content or "").strip()
            reasoning = (getattr(response.choices[0].message, "reasoning", None) or "").strip()
            rewritten = content or reasoning

            # Strip any markdown code fences or quotes
            rewritten = re.sub(r"^[`'\"]+|[`'\"]+$", "", rewritten).strip()

            # If model generated a multi-line essay or headers, extract ONLY the first short sentence
            lines = [
                line.strip()
                for line in rewritten.splitlines()
                if line.strip() and not line.strip().startswith(("#", "**", "---", "###"))
            ]
            if lines:
                rewritten = lines[0]

            # Enforce maximum length limit (under 180 chars) to prevent runaway prompts
            if len(rewritten) > 180:
                rewritten = rewritten[:180].rsplit(".", 1)[0].strip()

            if rewritten:
                logger.info(f"Query rewritten: '{question}' → '{rewritten}'")
                span.set_attribute("rewritten_question", rewritten)
                return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")

        span.set_attribute("rewritten_question", question)
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
    with logfire.span("rag.retrieve_chunks", question=question, is_aggregation=is_aggregation) as span:
        # Embed the query with dense vector (semantic) and sparse BM25 vector (keyword)
        query_vector = embedding.embed_query(question)
        query_sparse = embedding.embed_sparse_query(question)

        # Hybrid candidate retrieval (Dense + BM25 via server-side RRF) — top 10 candidates
        candidates = await vector_store.search(
            user_id=user_id,
            query_vector=query_vector,
            query_sparse_vector=query_sparse,
            limit=10,
        )
        span.set_attribute("candidates_retrieved", len(candidates))
        span.set_attribute("search_mode", "hybrid_rrf")

        # Cross-encoder reranking with FlashRank
        results = reranker.rerank_chunks(
            query=question,
            chunks=candidates,
            top_k=5,
        )
        span.set_attribute("reranked_count", len(results))
        if results:
            span.set_attribute("top_score", results[0].get("score", 0.0))
            span.set_attribute("top_source", results[0].get("filename", "unknown"))
            span.set_attribute("retrieved_sources", list(set(r.get("filename", "unknown") for r in results)))

        # If aggregation detected, fetch and prepend up to 3 summary chunks
        if is_aggregation:
            summary_chunks = await vector_store.get_summary_chunks(user_id)
            if summary_chunks:
                # Deduplicate: remove any summary chunks already in results
                result_ids = {r["id"] for r in results}
                new_summaries = [s for s in summary_chunks if s["id"] not in result_ids][:3]
                # Prepend summaries so they appear first in context
                results = new_summaries + results
                span.set_attribute("summary_chunks_added", len(new_summaries))
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
6. NEVER claim, pretend, or hallucinate that you performed a web search. You do not have external web access and must answer solely from the provided document context.
7. Be concise, accurate, and professional."""


def _build_context(chunks: list[dict], max_context_tokens: int = 4000) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Enforces a strict token budget to prevent exceeding TPM rate limits (e.g. Groq 8k limit).
    """
    context_parts = []
    current_tokens = 0

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

        part = f"[{tag}] (Source: {filename})\n{content}"
        part_tokens = count_tokens(part)

        # Enforce budget so request never exceeds TPM limit
        if current_tokens + part_tokens > max_context_tokens:
            if not context_parts:
                # If first chunk alone is large, truncate it to fit
                char_limit = max_context_tokens * 4
                context_parts.append(part[:char_limit] + "\n... [truncated for length]")
            break

        context_parts.append(part)
        current_tokens += part_tokens

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
    with logfire.span("rag.generate_answer", question=question, chunks_count=len(chunks)) as span:
        context = _build_context(chunks)

        groq = _get_groq()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context chunks:\n\n{context}\n\n"
                    f"Question: {question}"
                ),
            },
        ]

        token_count = 0
        async for token in stream_llm_with_fallback(
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
        ):
            token_count += 1
            yield token
        span.set_attribute("tokens_generated", token_count)


# ── Citation validation ───────────────────────────────────────────────────

def validate_citations(
    full_response: str,
    context_chunks: list[dict],
) -> list[dict]:
    """
    Extract and validate citation references from the LLM response.
    Supports [Page X], [Rows X-Y], [Summary], and [CHUNK <id>].
    """
    with logfire.span("rag.validate_citations", chunks_available=len(context_chunks)) as span:
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

        span.set_attribute("valid_citations_count", len(valid_citations))
        return valid_citations


# ── Text-to-SQL pipeline ──────────────────────────────────────────────────

SQL_GENERATION_PROMPT = """You are an expert SQL query generator. Given a database schema and a user question, generate a DuckDB-compatible SQL SELECT query that answers the question.

RULES:
1. Generate ONLY a single SQL SELECT statement. No INSERT, UPDATE, DELETE, DROP, or any DDL.
2. Output ONLY the raw SQL — no explanation, no notes, no commentary.
3. Use the exact table and column names from the schema.
4. For aggregations (SUM, AVG, COUNT, MIN, MAX), apply them on the correct numeric columns.
5. If the user asks an open-ended, overview, or exploratory question (e.g. "tell me about the data", "summarize", "key factors", "overview", "what are the drivers of X"):
   Generate an analytical aggregation query! For example:
   - Group by the primary category or target column (like Churn, Status, Department) with COUNT(*) and percentages
   - Or calculate key averages, distributions, or metrics (e.g. AVG(MonthlyCharges), AVG(tenure))
   - Or select top informative columns with LIMIT 20
   Do NOT return CANNOT_ANSWER for general data analysis questions.
6. Only return CANNOT_ANSWER if the question has absolutely nothing to do with the schema or tables.
7. Always alias aggregated columns with meaningful names (e.g., total_customers, avg_monthly_charges).
8. Handle NULL or blank values appropriately: for text/varchar columns containing numbers or spaces, ALWAYS use TRY_CAST(TRIM(col) AS DOUBLE) instead of CAST to avoid conversion errors.
9. If the user asks for "all data" or something very broad, use LIMIT 50.
10. Use ILIKE for case-insensitive text matching when filtering by text values."""


SQL_ANSWER_PROMPT = """You are a helpful data analyst assistant. You have been given the results of a SQL query executed on the user's uploaded spreadsheet data.

RULES:
1. Present the data in a clear, well-formatted Markdown response.
2. Use tables for tabular results, bullet points for summaries.
3. If the result has numeric aggregations, highlight the key numbers in bold.
4. Add brief interpretive commentary (e.g., "The highest revenue was **$52,000** from the East region.").
5. If the result is empty, explain that no matching data was found.
6. Be concise but thorough.
7. Reference the data source as "your uploaded spreadsheet data".
8. Do NOT mention SQL queries, databases, or technical details unless the user explicitly asked for the SQL."""


def _extract_sql(content: str) -> str | None:
    """Extract a valid SQL SELECT statement from raw LLM output."""
    if not content:
        return None
    if "CANNOT_ANSWER" in content:
        return None
    # 1. Match inside ```sql ... ``` or ``` ... ```
    m = re.search(r"```(?:sql)?\s*(SELECT\b[\s\S]*?)```", content, re.IGNORECASE)
    if m:
        sql = m.group(1).strip().rstrip(";")
        return sql
    # 2. Match raw SELECT statement
    m = re.search(r"\b(SELECT\b[\s\S]*?)(?:;\s*$|\Z)", content, re.IGNORECASE)
    if m:
        sql = m.group(1).strip().rstrip(";")
        sql = re.sub(r"```.*$", "", sql, flags=re.DOTALL).strip().rstrip(";")
        return sql
    return None


async def generate_sql_query(
    question: str,
    schema_info: str,
    chat_history: list[dict] | None = None,
) -> str | None:
    """
    Ask the LLM to generate a SQL query for the given question and schema.

    Returns the SQL string, or None if the question can't be answered with SQL.
    """
    with logfire.span("rag.generate_sql", question=question) as span:
        messages: list[dict] = [
            {"role": "system", "content": SQL_GENERATION_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Database schema:\n\n{schema_info}\n\n"
                    f"User question: {question}\n\n"
                    f"SQL query:"
                ),
            },
        ]

        try:
            response = await call_llm_with_fallback(
                messages=messages,
                max_tokens=400,
                temperature=0.0,
                fast=True,
            )
            content = (response.choices[0].message.content or "").strip()
            reasoning = (getattr(response.choices[0].message, "reasoning", None) or "").strip()

            sql = _extract_sql(content) or _extract_sql(reasoning)
            if not sql:
                logger.info(f"LLM says question cannot be answered with SQL or gave non-SELECT output: '{question[:80]}'")
                span.set_attribute("sql_generated", False)
                return None

            logger.info(f"Generated SQL: {sql}")
            span.set_attribute("sql_generated", True)
            span.set_attribute("sql", sql)
            return sql

        except Exception as e:
            logger.warning(f"SQL generation failed: {e}")
            span.set_attribute("sql_generated", False)
            span.set_attribute("error", str(e))
            return None


def format_sql_result(result: dict) -> str:
    """
    Format SQL query results as a readable string for the LLM context.
    """
    columns = result["columns"]
    rows = result["rows"]

    if not rows:
        return "The query returned no results."

    # Build a simple text table
    lines = []
    header = " | ".join(str(c) for c in columns)
    lines.append(header)
    lines.append("-" * len(header))

    for row in rows:
        lines.append(" | ".join(str(v) for v in row))

    if result.get("truncated"):
        lines.append(f"\n... (showing first {result['row_count']} of many rows)")

    return "\n".join(lines)


async def generate_sql_answer_stream(
    question: str,
    sql_query: str,
    sql_result_text: str,
) -> AsyncGenerator[str, None]:
    """
    Stream an LLM answer that interprets the SQL results for the user.
    """
    with logfire.span("rag.generate_sql_answer", question=question, sql_query=sql_query) as span:
        groq = _get_groq()

        messages = [
            {"role": "system", "content": SQL_ANSWER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"SQL query executed:\n{sql_query}\n\n"
                    f"Query results:\n{sql_result_text}\n\n"
                    f"User's question: {question}\n\n"
                    f"Please provide a clear, formatted answer:"
                ),
            },
        ]

        token_count = 0
        async for token in stream_llm_with_fallback(
            messages=messages,
            max_tokens=1000,
            temperature=0.1,
        ):
            token_count += 1
            yield token
        span.set_attribute("tokens_generated", token_count)


async def check_tabular_data(user_id: str) -> bool:
    """
    Check if the user has any tabular data in DuckDB.
    Runs the check in a thread to avoid blocking the event loop.
    """
    import asyncio
    return await asyncio.to_thread(tabular_store.has_tabular_data, user_id)


async def get_schema_for_prompt(user_id: str) -> str:
    """
    Get the full DuckDB schema for the user, formatted for the LLM prompt.
    Runs in a thread to avoid blocking.
    """
    import asyncio
    return await asyncio.to_thread(tabular_store.get_table_schema, user_id)


async def execute_sql_query(user_id: str, sql: str) -> dict:
    """
    Execute a SQL query on the user's DuckDB.
    Runs in a thread to avoid blocking.
    """
    import asyncio
    with logfire.span("duckdb.execute_sql", sql=sql) as span:
        result = await asyncio.to_thread(tabular_store.execute_sql, user_id, sql)
        span.set_attribute("row_count", result.get("row_count", 0))
        span.set_attribute("columns", result.get("columns", []))
        return result


async def get_tabular_tables(user_id: str) -> list[str]:
    """
    Get list of table names in the user's DuckDB.
    Runs in a thread to avoid blocking.
    """
    import asyncio
    return await asyncio.to_thread(tabular_store.list_tables, user_id)
