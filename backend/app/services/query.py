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

import logfire

from app.config import settings
from app.services import embedding, vector_store, reranker
from app.services.chunking import count_tokens
from app.services import tabular_store

logger = logging.getLogger(__name__)

from app.services.llm_provider import (
    call_llm_with_cross_provider_fallback as call_llm_with_fallback,
    stream_llm_with_cross_provider_fallback as stream_llm_with_fallback,
    get_groq_client as _get_groq,
    get_generation_targets,
    get_fast_targets,
    is_recoverable_error as _is_rate_limit_or_recoverable,
)

# ── Resilient Multi-Provider Fallback Chain ──────────────────────────────
# When one model/provider hits rate limits (429/TPM) or context limits (413),
# the system automatically and transparently switches to the next available
# model across Groq, Google Gemini, and OpenRouter.
DEFAULT_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "qwen/qwen3.6-27b",
]
FAST_MODEL_CHAIN = [
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound-mini",
]


def get_model_chain(fast: bool = False) -> list[str]:
    """Compatibility helper returning the Groq model list."""
    if fast:
        return list(FAST_MODEL_CHAIN)
    configured = settings.groq_model
    chain = [configured]
    for m in DEFAULT_MODEL_FALLBACKS:
        if m not in chain and m != configured:
            chain.append(m)
    return chain




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


# ── Chat Auto-Naming ──────────────────────────────────────────────────────

TITLE_SYSTEM_PROMPT = """You are a back-end utility that generates highly concise, 3-to-5 word titles for chat logs.
Analyze the user's initial inquiry and the assistant's response.
Extract the core topic, task, or technical domain.

CRITICAL RULES:
- Output ONLY the title. Do not include markdown, bullet points, introductory phrases, or quotes.
- Do not use generic words like "Chat", "Conversation", "Discussion", or "Question".
- Keep it between 2 to 5 words maximum.
- Be specific. (e.g., instead of "Coding Help", use "React Auth State Debugging").
- If the interaction is pure social pleasantry/greeting with no substantive topic (e.g., "Hi" / "Hello"), output EXACTLY: "New conversation"."""


def _clean_title(raw: str) -> str:
    """Clean and normalize raw LLM output into a strict 2-to-5 word title."""
    if not raw:
        return ""

    # Remove markdown bold/italics, quotes, and backticks
    cleaned = re.sub(r"[*_`'\"]", "", raw).strip()

    # Strip conversational prefixes like "Title:", "Here is a title:", "Topic:"
    cleaned = re.sub(r"^(?:title|topic|here is a title|suggested title)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # Strip trailing punctuation
    cleaned = re.sub(r"[.,;:\-!?]+$", "", cleaned).strip()

    # Check for exact "New conversation" (preserve as is)
    if cleaned.lower() in ("new conversation", "new conversation.", "new chat"):
        return "New conversation"

    # Split into words and enforce 2 to 5 words limit
    words = cleaned.split()
    if not words:
        return ""

    # If first word is generic like "Chat", "Question", strip if length > 2
    if len(words) > 2 and words[0].lower() in ("chat", "conversation", "discussion", "question", "about"):
        words = words[1:]

    # Enforce maximum 5 words
    if len(words) > 5:
        words = words[:5]

    return " ".join(words)


async def generate_chat_title(
    chat_history: list[dict] | None,
    current_question: str,
    current_answer: str = "",
) -> str:
    """
    Generate a clean 2-to-5 word title for the chat session using both
    the user inquiry and the assistant response.
    Returns "New conversation" if the conversation is purely pleasantries.
    """
    with logfire.span("rag.generate_chat_title", question=current_question[:60]) as span:
        # Fast check: if only message and it's a simple greeting, return default immediately
        cleaned_q = current_question.strip().lower()
        cleaned_q = re.sub(r"[^\w\s]", "", cleaned_q)
        quick_greetings = {"hi", "hello", "hey", "hola", "howdy", "good morning", "good afternoon", "good evening"}
        if (not chat_history or len(chat_history) == 0) and cleaned_q in quick_greetings and not current_answer:
            span.set_attribute("title", "New conversation")
            span.set_attribute("method", "heuristic_greeting")
            return "New conversation"

        # Build context snippet from history + current exchange
        snippets = []
        if chat_history:
            for m in chat_history[-3:]:
                role = "User" if m.get("role") == "user" else "Assistant"
                snippets.append(f"{role}: {m.get('content', '')[:150]}")

        snippets.append(f"User: {current_question[:250]}")
        if current_answer:
            snippets.append(f"Assistant: {current_answer[:300]}")

        conversation_context = "\n".join(snippets)

        messages = [
            {"role": "system", "content": TITLE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Conversation snippet:\n{conversation_context}\n\nTitle:",
            },
        ]

        try:
            response = await call_llm_with_fallback(
                messages=messages,
                max_tokens=25,
                temperature=0.2,
                fast=True,
            )
            raw = (response.choices[0].message.content or "").strip()
            title = _clean_title(raw)
            if title:
                logger.info(f"Generated chat title: '{title}' for '{current_question[:50]}'")
                span.set_attribute("title", title)
                return title
        except Exception as e:
            logger.warning(f"Chat title generation failed: {e}")

        # Fallback: extract first 3-5 words from question
        q_words = [w for w in re.findall(r"\b\w+\b", current_question) if w.lower() not in {"what", "is", "the", "how", "can", "you", "a", "an", "tell", "me"}]
        if q_words:
            fallback = " ".join(q_words[:4]).title()
            span.set_attribute("title", fallback)
            span.set_attribute("method", "fallback_keyword")
            return fallback

        span.set_attribute("title", "New conversation")
        return "New conversation"



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


# ── Contextual Compression ────────────────────────────────────────────────
COMPRESSION_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def compress_chunk_content(
    content: str,
    query: str,
    max_sentences: int = 3,
    window_size: int = 1,
) -> str:
    """
    Extractively compresses a text chunk by retaining only sentences relevant to the query,
    expanded with adjacent sentence context (±1) to ensure grammatical and narrative continuity.

    Returns the original content unmodified if:
    - The chunk is already concise (<= 4 sentences).
    - No significant query term matches are found (failsafe to prevent accidental data loss).
    """
    if not content or not query:
        return content

    from app.services.chunking import _split_into_sentences
    sentences = _split_into_sentences(content)

    # Do not compress already compact chunks
    if len(sentences) <= 4:
        return content

    # Extract meaningful query keywords (alphanumeric, length > 1, not stop words)
    q_tokens = [
        w.lower() for w in re.findall(r"\b\w+\b", query)
        if len(w) > 1 and w.lower() not in COMPRESSION_STOP_WORDS
    ]
    if not q_tokens:
        return content

    q_set = set(q_tokens)

    # Score each sentence
    scored: list[tuple[float, int]] = []
    for i, s in enumerate(sentences):
        s_lower = s.lower()
        s_tokens = re.findall(r"\b\w+\b", s_lower)
        if not s_tokens:
            continue
        s_set = set(s_tokens)

        # 1. Exact query keyword overlap count
        overlap = len(q_set & s_set)
        if overlap == 0:
            continue

        # 2. Density bonus: higher score if matching words form a bigger fraction of the sentence
        density = overlap / len(s_set)

        # 3. Exact phrase match bonus (if 2+ consecutive query words appear verbatim in sentence)
        phrase_bonus = 0.0
        if len(q_tokens) >= 2:
            for j in range(len(q_tokens) - 1):
                bigram = f"{q_tokens[j]} {q_tokens[j+1]}"
                if bigram in s_lower:
                    phrase_bonus += 1.5

        total_score = overlap + (density * 2.0) + phrase_bonus
        scored.append((total_score, i))

    # If no sentences matched, return full content safely (zero data loss)
    if not scored:
        return content

    # Sort sentences by relevance score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top scoring sentences and expand each with adjacent window (±window_size)
    selected_indices: set[int] = set()
    for _, idx in scored[:max_sentences]:
        start = max(0, idx - window_size)
        end = min(len(sentences), idx + window_size + 1)
        for w in range(start, end):
            selected_indices.add(w)

    # If selected sentences encompass almost the entire chunk (> 75%), keep original chunk
    if len(selected_indices) >= len(sentences) * 0.75:
        return content

    # Reassemble selected sentences in chronological document order
    sorted_indices = sorted(selected_indices)
    parts = []
    prev_idx = -1
    for idx in sorted_indices:
        if prev_idx != -1 and idx > prev_idx + 1:
            parts.append("[...]")
        parts.append(sentences[idx].strip())
        prev_idx = idx

    return " ".join(parts)


def _build_context(
    chunks: list[dict],
    query: str = "",
    max_context_tokens: int = 4000,
    enable_compression: bool = True,
) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.
    Enforces a strict token budget and applies extractive contextual compression
    to prune distractor sentences while preserving citation metadata.
    """
    context_parts = []
    current_tokens = 0
    total_raw_tokens = 0

    for chunk in chunks:
        filename = chunk.get("filename", "unknown")
        chunk_type = chunk.get("chunk_type", "text")
        content = chunk.get("content", "")
        raw_chunk_tokens = count_tokens(content)
        total_raw_tokens += raw_chunk_tokens

        # Apply extractive contextual compression to text chunks
        if enable_compression and query and chunk_type == "text":
            content = compress_chunk_content(content, query)

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

    if total_raw_tokens > 0 and current_tokens < total_raw_tokens:
        savings = ((total_raw_tokens - current_tokens) / total_raw_tokens) * 100
        logger.info(
            f"Contextual compression reduced context: {total_raw_tokens} -> {current_tokens} tokens (-{savings:.1f}%)"
        )

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
        context = _build_context(chunks, query=question)

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
