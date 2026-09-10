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


# ── Document Scope & Ambiguity Analysis ──────────────────────────────────

class DocumentScopeResult:
    """
    Result of evaluating whether a user query targets a specific document,
    spans all documents, or is too ambiguous and needs clarification.
    """
    def __init__(
        self,
        status: str,  # "needs_clarification" | "resolved" | "all_docs" | "no_documents"
        target_doc_id: str | None = None,
        target_filename: str | None = None,
        is_tabular: bool = False,
        clarification_text: str | None = None,
        cleaned_question: str | None = None,
    ):
        self.status = status
        self.target_doc_id = target_doc_id
        self.target_filename = target_filename
        self.is_tabular = is_tabular
        self.clarification_text = clarification_text
        self.cleaned_question = cleaned_question or ""

    def __repr__(self) -> str:
        return (
            f"DocumentScopeResult(status={self.status}, "
            f"target_filename={self.target_filename}, "
            f"is_tabular={self.is_tabular})"
        )


def format_clarification_message(docs: list[dict], question: str) -> str:
    """Generate a clean, conversational markdown response listing user documents for clarification."""
    lines = [
        "I noticed you have several documents in your library:",
        "",
    ]
    for d in docs:
        fname = d.get("filename", "")
        ftype = d.get("file_type", "doc").upper()
        icon = "📊" if d.get("is_tabular") else ("📄" if ftype == "PDF" else "📝")
        extra = ""
        f_lower = fname.lower()
        if "2405.15793" in f_lower or "swe-bench" in f_lower:
            extra = " *(SWE-bench research paper)*"
        elif "verity" in f_lower:
            extra = " *(novel by Colleen Hoover)*"
        elif "it-ends-with-us" in f_lower or "it ends with us" in f_lower:
            extra = " *(novel by Colleen Hoover)*"
        elif "strangers" in f_lower:
            extra = " *(book / novel)*"
        elif "titan" in f_lower:
            extra = " *(Project Titan report)*"
        elif "churn" in f_lower:
            extra = " *(Telco customer churn dataset)*"
        elif "e commerce" in f_lower or "dataset" in f_lower:
            extra = " *(E-commerce sales dataset)*"
        elif "python" in f_lower:
            extra = " *(Python programming handbook)*"
        elif "offer" in f_lower:
            extra = " *(Offer letter document)*"
        lines.append(f"- {icon} **{fname}**{extra}")

    clean_q = question.strip()
    if clean_q.endswith("?"):
        clean_q = clean_q[:-1].strip()
    lines.append("")
    lines.append(f"**Which document would you like me to examine for \"{clean_q}\"?**")
    lines.append("*(Or reply with **all** if you would like an overview across all of them!)*")
    return "\n".join(lines)


DOCUMENT_SCOPE_SYSTEM_PROMPT = """You are a document-routing and ambiguity-detection analyst for an AI document assistant.
The user has a personal document library. Your job is to determine whether their question targets one specific document, applies across all documents, or is AMBIGUOUS because it assumes a specific document without naming which one.

Available documents in the user's workspace:
{document_catalog}

Decision Categories:
1. SPECIFIC: The question explicitly mentions a document by name, topic, or distinct keyword (or recent chat history already established the focus document, so this is a clear follow-up).
2. CROSS_DOC: The user is explicitly asking to compare, summarize, or search across multiple or all documents in their library (e.g., "compare all my documents", "what files do I have?", "search across all files").
3. GENERAL: The question is a general factual or conceptual inquiry (e.g. "What is machine learning?", "How do Python functions work?") that does not assume a specific proprietary story, paper, or report.
4. AMBIGUOUS: The user asks an underspecified question that clearly refers to a single document's contents (e.g., "What are the key findings?", "Summarize the document", "Who is the main character?", "What does the contract say?", "What was the conclusion?", "Explain the methodology") BUT there are multiple documents and the user never specified which one they mean, and recent chat history does NOT establish a focus document.

CRITICAL RULES:
- If the question is AMBIGUOUS, output:
  CATEGORY: AMBIGUOUS
- If SPECIFIC, output:
  CATEGORY: SPECIFIC
  DOCUMENT_INDEX: <number from 1 to N matching the document>
- If CROSS_DOC, output:
  CATEGORY: CROSS_DOC
- If GENERAL, output:
  CATEGORY: GENERAL
"""


def _find_matching_doc_by_keyword(text: str, docs: list[dict]) -> dict | None:
    text_lower = text.lower().strip()
    for d in docs:
        fname = d["filename"].lower()
        base_name = fname.rsplit(".", 1)[0]
        if base_name in text_lower or fname in text_lower:
            return d

        keywords = []
        if "2405.15793" in fname or "swe-bench" in fname:
            keywords.extend(["2405.15793", "swe-bench", "swe bench", "swebench", "research paper", "the paper"])
        elif "verity" in fname:
            keywords.extend(["verity", "colleen hoover"])
        elif "it-ends-with-us" in fname or "it ends with us" in fname:
            keywords.extend(["it ends with us", "ends with us", "lily bloom"])
        elif "strangers" in fname:
            keywords.extend(["strangers again", "can we be strangers"])
        elif "titan" in fname:
            keywords.extend(["titan report", "project titan", "titan"])
        elif "churn" in fname:
            keywords.extend(["telco", "churn", "customer churn"])
        elif "e commerce" in fname or "dataset" in fname:
            keywords.extend(["e-commerce", "ecommerce", "e commerce"])
        elif "python" in fname:
            keywords.extend(["python handbook", "ultimate python", "python book"])
        elif "offer" in fname:
            keywords.extend(["offer letter", "yash mittal offer", "offer"])

        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                return d
    return None


async def analyze_document_scope(
    question: str,
    chat_history: list[dict] | None = None,
    available_documents: list[dict] | None = None,
) -> DocumentScopeResult:
    """
    Evaluate user query against available documents.
    Detects ambiguity, explicit mentions, chat history carryover, or clarification responses.
    """
    if not available_documents:
        return DocumentScopeResult(
            status="no_documents",
            clarification_text="You don't have any uploaded documents yet. Please upload a document to get started.",
        )

    # If user has only 1 document in library, zero ambiguity
    if len(available_documents) == 1:
        d = available_documents[0]
        return DocumentScopeResult(
            status="resolved",
            target_doc_id=str(d["id"]),
            target_filename=d["filename"],
            is_tabular=d.get("is_tabular", False),
            cleaned_question=question,
        )

    # 1. Check for @mention tag (e.g. "@verity what happened?" or "@SWE-bench key findings")
    at_match = re.search(r"@([A-Za-z0-9_.-]+)", question)
    if at_match:
        tag = at_match.group(1).lower()
        matched = _find_matching_doc_by_keyword(tag, available_documents)
        if matched:
            cleaned = re.sub(r"@[A-Za-z0-9_.-]+\s*", "", question).strip()
            return DocumentScopeResult(
                status="resolved",
                target_doc_id=str(matched["id"]),
                target_filename=matched["filename"],
                is_tabular=matched.get("is_tabular", False),
                cleaned_question=cleaned,
            )

    # 2. Check if the user is answering a prior clarification question
    if chat_history and len(chat_history) >= 1:
        last_msg = chat_history[-1]
        if last_msg.get("role") == "assistant":
            last_content = last_msg.get("content", "")
            if "Which document would you like me to examine" in last_content or "Which document would you like me to focus on" in last_content:
                # Did user say "all" or "everything"?
                q_clean = question.strip().lower()
                if q_clean in ("all", "both", "all of them", "everything", "all documents"):
                    orig_q = ""
                    for m in reversed(chat_history[:-1]):
                        if m.get("role") == "user":
                            orig_q = m.get("content", "")
                            break
                    return DocumentScopeResult(status="all_docs", cleaned_question=orig_q or question)

                # Match document from answer
                matched = _find_matching_doc_by_keyword(question, available_documents)
                if matched:
                    orig_q = ""
                    for m in reversed(chat_history[:-1]):
                        if m.get("role") == "user":
                            orig_q = m.get("content", "")
                            break
                    cleaned_q = f"{orig_q} in {matched['filename']}" if orig_q else f"Information from {matched['filename']}"
                    return DocumentScopeResult(
                        status="resolved",
                        target_doc_id=str(matched["id"]),
                        target_filename=matched["filename"],
                        is_tabular=matched.get("is_tabular", False),
                        cleaned_question=cleaned_q,
                    )

    # 3. Check for explicit library-wide queries
    q_lower = question.lower().strip()
    if re.search(r"\b(all\s+documents|all\s+my\s+documents|all\s+files|across\s+all|compare\s+(the\s+)?(documents|files|books)|what\s+documents\s+do\s+i\s+have|list\s+(all\s+)?(my\s+)?(documents|files))\b", q_lower):
        return DocumentScopeResult(status="all_docs", cleaned_question=question)

    # 4. Check for prominent document title/name in the question
    matched = _find_matching_doc_by_keyword(question, available_documents)
    if matched:
        return DocumentScopeResult(
            status="resolved",
            target_doc_id=str(matched["id"]),
            target_filename=matched["filename"],
            is_tabular=matched.get("is_tabular", False),
            cleaned_question=question,
        )

    # 4b. Fast heuristic for standard ambiguous queries in multi-doc library
    ambiguous_patterns = [
        r"^(what\s+are\s+the\s+)?key\s+findings\b",
        r"^(can\s+you\s+)?summarize(\s+(the\s+)?(document|file|paper|book|dataset|it|this))?\b",
        r"^what\s+is\s+(this|the)\s+(document|file|paper|book)\s+about\b",
        r"^(what\s+are\s+the\s+)?(main\s+takeaways|conclusions)\b",
        r"^(give\s+me\s+a\s+)?summary(\s+of\s+the\s+(document|file|paper))?\b",
    ]
    if not chat_history and any(re.search(p, q_lower) for p in ambiguous_patterns):
        return DocumentScopeResult(
            status="needs_clarification",
            clarification_text=format_clarification_message(available_documents, question),
        )

    # 5. Fast LLM Ambiguity / Scope Classification
    catalog_lines = []
    for i, d in enumerate(available_documents, 1):
        catalog_lines.append(f"[{i}] {d['filename']} ({'Tabular' if d.get('is_tabular') else 'Text Document'})")
    catalog_str = "\n".join(catalog_lines)

    messages = [
        {
            "role": "system",
            "content": DOCUMENT_SCOPE_SYSTEM_PROMPT.replace("{document_catalog}", catalog_str),
        },
    ]
    if chat_history:
        for m in chat_history[-3:]:
            messages.append({"role": m["role"], "content": m["content"][:200]})
    messages.append({"role": "user", "content": question})

    try:
        response = await call_llm_with_fallback(
            messages=messages,
            max_tokens=150,
            temperature=0.0,
            fast=True,
        )
        resp_text = (response.choices[0].message.content or "").strip()
        resp_reasoning = (getattr(response.choices[0].message, "reasoning", None) or "").strip()
        combined = f"{resp_text}\n{resp_reasoning}".upper()

        if "CATEGORY: AMBIGUOUS" in combined or "AMBIGUOUS" in resp_text.upper():
            return DocumentScopeResult(
                status="needs_clarification",
                clarification_text=format_clarification_message(available_documents, question),
                cleaned_question=question,
            )
        elif "CATEGORY: SPECIFIC" in combined or "SPECIFIC" in resp_text.upper():
            idx_match = re.search(r"DOCUMENT_INDEX:\s*(\d+)", combined)
            if idx_match:
                doc_idx = int(idx_match.group(1)) - 1
                if 0 <= doc_idx < len(available_documents):
                    target = available_documents[doc_idx]
                    return DocumentScopeResult(
                        status="resolved",
                        target_doc_id=str(target["id"]),
                        target_filename=target["filename"],
                        is_tabular=target.get("is_tabular", False),
                        cleaned_question=question,
                    )
        elif "CATEGORY: CROSS_DOC" in combined:
            return DocumentScopeResult(status="all_docs", cleaned_question=question)

    except Exception as e:
        logger.warning(f"Document scope analysis failed: {e}")

    # Fallback to all_docs
    return DocumentScopeResult(status="all_docs", cleaned_question=question)


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

DANGLING_END_WORDS = {
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as", "into", "onto", "upon", "about",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "we", "you", "they", "it", "he", "she", "i", "me", "us", "him", "her", "them",
    "my", "your", "its", "their", "our", "his",
    "that", "this", "these", "those", "which", "whose", "whom", "who", "what", "where", "when", "why", "how",
}

TITLE_SYSTEM_PROMPT = """You generate concise, self-contained chat names (3 to 6 words) for the sidebar of an AI document assistant application.
Analyze the user's inquiry and the assistant's response to identify the core topic, book, dataset, or technical question being discussed.

CRITICAL RULES:
1. The chat name MUST be a complete, self-contained phrase or title (3 to 6 words).
2. It MUST make complete grammatical sense on its own. NEVER cut off mid-sentence or trail off.
3. NEVER end on dangling words, prepositions, conjunctions, pronouns, or auxiliary verbs (e.g., NEVER end with 'be', 'to', 'of', 'the', 'a', 'in', 'on', 'and', 'for', 'with', 'we', 'can').
4. Be specific to the actual subject. Do NOT use redundant label prefixes like 'Book Summary:', 'Document Analysis:', or 'Question About:' — focus directly on the actual subject or work title so the entire 3 to 6 words are meaningful (e.g., instead of 'Book Summary: Can We Be', use 'Can We Be Friends Summary' or the full book title).
5. Output ONLY the title text. Do not include quotes, markdown, bullet points, colons, or introductory preamble.
6. If the conversation is purely social pleasantries or a greeting with no substantive topic (e.g., 'Hi' / 'Hello'), output EXACTLY: New conversation."""


def _repair_dangling_title(title: str, context_text: str = "") -> str:
    """If a title ends on an incomplete dangling word, attempt to complete it from context or fix it."""
    words = title.split()
    if not words:
        return title

    # If ending on a dangling word (e.g. "Can We Be" -> "Be")
    if words[-1].lower() in DANGLING_END_WORDS and context_text:
        # Search context (question/answer) for the sequence of words and capture non-punctuation continuation
        search_phrase = " ".join(words)
        pattern = re.compile(rf"\b{re.escape(search_phrase)}\s+([^\n\r.,;:!?]+)", re.IGNORECASE)
        match = pattern.search(context_text)
        if match:
            continuation_words = match.group(1).strip().split()
            added = []
            for w in continuation_words:
                if len(words) + len(added) >= 6:
                    break
                added.append(w)
            # Trim trailing dangling words from added segment
            while added and added[-1].lower() in DANGLING_END_WORDS:
                added.pop()
            if added:
                return " ".join(words + added)

    # If still ending in a dangling word, trim backward if we have at least 3 words left
    while len(words) > 3 and words[-1].lower() in DANGLING_END_WORDS:
        words.pop()

    # If still dangling and <= 3 words, append a clarifying noun rather than leaving it broken
    if words and words[-1].lower() in DANGLING_END_WORDS:
        if words[-1].lower() in {"be", "is", "are", "can", "could", "should", "will"}:
            words.append("Overview")
        else:
            words.pop()
            if not words:
                return "Document Overview"

    return " ".join(words)


def _clean_title(raw: str, max_words: int = 6) -> str:
    """Clean and normalize raw LLM output into a complete 3-to-6 word chat name."""
    if not raw:
        return ""

    # Remove markdown bold/italics, quotes, and backticks
    cleaned = re.sub(r"[*_`'\"]", "", raw).strip()

    # Strip conversational prefixes like "Title:", "Here is a title:", "Topic:", "Chat Name:"
    cleaned = re.sub(
        r"^(?:title|topic|here is a title|suggested title|chat name|chat title|suggested name)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # Strip redundant category prefixes like "Book Summary:", "Document Summary:", "Summary:"
    cleaned = re.sub(
        r"^(?:book summary|document summary|paper summary|text summary|executive summary|summary|overview|analysis)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # Strip trailing punctuation
    cleaned = re.sub(r"[.,;:\-!?]+$", "", cleaned).strip()

    # Check for exact "New conversation" (preserve as is)
    if cleaned.lower() in ("new conversation", "new conversation.", "new chat"):
        return "New conversation"

    # Split into words and enforce 3 to 6 words limit
    words = cleaned.split()
    if not words:
        return ""

    # If first word is generic like "Chat", "Conversation", "Discussion", "Question", strip if length > 2
    if len(words) > 2 and words[0].lower() in ("chat", "conversation", "discussion", "question", "about"):
        words = words[1:]

    # Enforce maximum words (up to 6 words)
    if len(words) > max_words:
        # If dropping leading article ("The", "A", "An") helps fit in 6 words
        if len(words) == max_words + 1 and words[0].lower() in ("the", "a", "an"):
            words = words[1:]
        else:
            words = words[:max_words]

    # Trim trailing dangling words if longer than 3 words
    while len(words) > 3 and words[-1].lower() in DANGLING_END_WORDS:
        words.pop()

    return " ".join(words)


async def generate_chat_title(
    chat_history: list[dict] | None,
    current_question: str,
    current_answer: str = "",
) -> str:
    """
    Generate a complete 3-to-6 word title for the chat session using both
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
                max_tokens=60,
                temperature=0.2,
                fast=True,
            )
            raw = (response.choices[0].message.content or "").strip()
            title = _clean_title(raw, max_words=6)
            combined_context = f"{current_question} {current_answer}"
            title = _repair_dangling_title(title, combined_context)
            if title:
                logger.info(f"Generated chat title: '{title}' for '{current_question[:50]}'")
                span.set_attribute("title", title)
                return title
        except Exception as e:
            logger.warning(f"Chat title generation failed: {e}")

        # Fallback: extract substantive keywords from question, avoiding stop/dangling words
        q_words = [
            w for w in re.findall(r"\b\w+\b", current_question)
            if w.lower() not in DANGLING_END_WORDS and w.lower() not in {"what", "how", "tell", "give", "please", "explain"}
        ]
        if q_words:
            fallback = " ".join(q_words[:5]).title()
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
    doc_id_filter: str | None = None,
    filename_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve relevant chunks for answering the question.

    1. Embed the query
    2. Search Qdrant for top-K similar chunks (optionally scoped to a specific document)
    3. If aggregation detected, also fetch summary chunks and prepend them
    """
    with logfire.span("rag.retrieve_chunks", question=question, is_aggregation=is_aggregation, doc_id_filter=doc_id_filter) as span:
        # Embed the query with dense vector (semantic) and sparse BM25 vector (keyword)
        query_vector = embedding.embed_query(question)
        query_sparse = embedding.embed_sparse_query(question)

        # Hybrid candidate retrieval (Dense + BM25 via server-side RRF) — top 10 candidates
        candidates = await vector_store.search(
            user_id=user_id,
            query_vector=query_vector,
            query_sparse_vector=query_sparse,
            limit=10,
            doc_id_filter=doc_id_filter,
            filename_filter=filename_filter,
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
            summary_chunks = await vector_store.get_summary_chunks(user_id, doc_id_filter=doc_id_filter)
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
5. If the user asks an open-ended, overview, exploratory, or summary question about the spreadsheet/table data (e.g. "key findings", "summary", "overview", "what are the trends", "tell me about the data", "findings in <filename>"):
   You MUST generate a meaningful analytical aggregation query on the main data table!
   - For example: SELECT count(*) AS total_records, round(avg(CashbackAmount), 2) AS avg_cashback, sum(Churn) AS churn_count FROM <table>;
   - Or group by the primary category or status column with count(*);
   - Do NOT return CANNOT_ANSWER for exploratory data analysis, summaries, or key findings questions on tabular datasets.
6. ONLY return CANNOT_ANSWER if the user question has NOTHING to do with any tabular data and is clearly asking about a non-tabular file (such as a literary novel plot, a signed offer letter, or a research paper's methodology).
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


TABULAR_OVERVIEW_PROMPT = """You are an expert data analyst assistant. The user is asking a question about an uploaded spreadsheet/tabular dataset.
You have access to the table schema, column names, types, and sample rows from DuckDB.

RULES:
1. Provide a clear, structured, and insightful response based on the dataset's columns and sample data.
2. If the user asked for "key findings", "summary", or "overview", analyze the business domain, the variables tracked, key metrics, and highlight notable patterns visible in the schema and sample data.
3. Suggest specific quantitative questions the user can ask next (e.g. churn rates by customer status, average tenure, correlation between satisfaction score and churn, etc.).
4. Use clean Markdown formatting with bullet points and bold highlights.
5. Reference the dataset by its filename if provided, or as "your uploaded spreadsheet data".
6. Do NOT mention DuckDB, table prefixes (like t_...), or database internals."""


async def generate_tabular_overview_stream(
    question: str,
    schema_info: str,
    filename: str = "",
) -> AsyncGenerator[str, None]:
    """
    Stream an overview/insight response for tabular data when SQL cannot be generated or fails.
    """
    with logfire.span("rag.tabular_overview", question=question, filename=filename) as span:
        messages = [
            {"role": "system", "content": TABULAR_OVERVIEW_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Dataset: {filename}\n\n"
                    f"Database Schema and Sample Rows:\n{schema_info}\n\n"
                    f"User question: {question}\n\n"
                    f"Please provide an insightful, structured response answering the user's question based on this dataset:"
                ),
            },
        ]
        token_count = 0
        async for token in stream_llm_with_fallback(
            messages=messages,
            max_tokens=1000,
            temperature=0.2,
        ):
            token_count += 1
            yield token
        span.set_attribute("tokens_generated", token_count)


async def get_schema_for_prompt(user_id: str, doc_id: UUID | None = None) -> str:
    """
    Get the DuckDB schema for the user (optionally scoped to a specific doc_id), formatted for the LLM prompt.
    Runs in a thread to avoid blocking.
    """
    import asyncio
    return await asyncio.to_thread(tabular_store.get_table_schema, user_id, doc_id)


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
