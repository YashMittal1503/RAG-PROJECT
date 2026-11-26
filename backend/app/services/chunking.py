"""
Text chunking logic for PDF/TXT.

Page-boundary-aware splitting with token-based sizing. Each PDF page produces its own
chunk(s) — content is NEVER merged across pages. This ensures accurate page citations
and respects the embedding model's context window.

Spreadsheets (XLSX/CSV) are not chunked: they are stored as DuckDB tables and answered
through Text-to-SQL (see ingestion.py and tabular_store.py).

Every chunk is a ChunkData object that carries its content and metadata.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from app.services.parsing import PageText

logger = logging.getLogger(__name__)

# We use cl100k_base (GPT-4 tokenizer) as a proxy for token counting.
# It's not the exact BGE tokenizer, but it's a reasonable approximation for chunk sizing purposes.
_tokenizer = tiktoken.get_encoding("cl100k_base")

# ── Chunk size parameters ─────────────────────────────────────────────────
# MAX_CHUNK_TOKENS = 512 matches the context window of BAAI/bge-small-en-v1.5.
# Tokens beyond 512 are silently truncated by the embedding model, so this
# ensures every token in a chunk is actually embedded and searchable.
MIN_CHUNK_TOKENS = 100
TARGET_CHUNK_TOKENS = 400
MAX_CHUNK_TOKENS = 512
OVERLAP_FRACTION = 0.10        # ~10% overlap between consecutive chunks within a page


# ── Data class for chunk output ───────────────────────────────────────────

@dataclass
class ChunkData:
    """A single chunk ready for embedding and storage."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    chunk_index: int = 0
    chunk_type: str = "text"           # "text" or "summary"
    page_number: Optional[int] = None
    row_range_start: Optional[int] = None
    row_range_end: Optional[int] = None
    token_count: int = 0


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(_tokenizer.encode(text))


# ── Text chunking (PDF / TXT) ────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, preserving delimiter and meaning.
    Oversized sentences exceeding MAX_CHUNK_TOKENS (e.g. dense research paper tables,
    citations, code, or formulas) are automatically sub-split.
    """
    import re
    # Split on sentence-ending punctuation followed by whitespace, or on double newlines
    raw_parts = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    sentences: list[str] = []
    for part in raw_parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        # If a single sentence/block exceeds MAX_CHUNK_TOKENS, break it into smaller sub-sentences
        if count_tokens(cleaned) > MAX_CHUNK_TOKENS:
            sub_parts = re.split(r'(?<=[;:])\s+|\n+', cleaned)
            for sp in sub_parts:
                sp_clean = sp.strip()
                if not sp_clean:
                    continue
                if count_tokens(sp_clean) > MAX_CHUNK_TOKENS:
                    # Hard word slice to guarantee chunk fits under MAX_CHUNK_TOKENS
                    words = sp_clean.split()
                    w_buf: list[str] = []
                    for w in words:
                        w_buf.append(w)
                        if len(w_buf) >= 200:  # ~250-300 tokens, safely under 512
                            sentences.append(" ".join(w_buf))
                            w_buf = []
                    if w_buf:
                        sentences.append(" ".join(w_buf))
                else:
                    sentences.append(sp_clean)
        else:
            sentences.append(cleaned)
    return sentences


def _chunk_single_page(
    sentences: list[str],
    page_number: int,
    chunk_index_start: int,
) -> list[ChunkData]:
    """
    Chunk the sentences of a single page into one or more chunks.

    Rules:
    - If the page's total tokens fit within MAX_CHUNK_TOKENS, produce ONE chunk.
    - If the page exceeds MAX_CHUNK_TOKENS, split within the page using
      sentence-accumulation with overlap.
    - Each chunk gets exactly this page's page_number.
    - Overlap only happens within the same page, never across pages.
    """
    if not sentences:
        return []

    # Fast path: check if the entire page fits in one chunk
    total_tokens = sum(count_tokens(s) for s in sentences)
    if total_tokens <= MAX_CHUNK_TOKENS:
        content = " ".join(sentences)
        return [ChunkData(
            content=content,
            chunk_index=chunk_index_start,
            chunk_type="text",
            page_number=page_number,
            token_count=total_tokens,
        )]

    # Slow path: split the page into multiple chunks
    chunks: list[ChunkData] = []
    chunk_index = chunk_index_start
    overlap_tokens = int(TARGET_CHUNK_TOKENS * OVERLAP_FRACTION)

    current_sentences: list[str] = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = count_tokens(sent)

        # If adding this sentence exceeds MAX_CHUNK_TOKENS and we already have content:
        if current_tokens + sent_tokens > MAX_CHUNK_TOKENS and current_sentences:
            chunk_text_content = " ".join(current_sentences)
            chunks.append(ChunkData(
                content=chunk_text_content,
                chunk_index=chunk_index,
                chunk_type="text",
                page_number=page_number,
                token_count=current_tokens,
            ))
            chunk_index += 1

            # Compute overlap sentences from the end of current chunk
            overlap_sents: list[str] = []
            overlap_tok = 0
            if sent_tokens < MAX_CHUNK_TOKENS:
                for s in reversed(current_sentences):
                    st = count_tokens(s)
                    if overlap_tok + st > overlap_tokens or overlap_tok + st + sent_tokens > MAX_CHUNK_TOKENS:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tok += st

            current_sentences = overlap_sents
            current_tokens = overlap_tok

        # Add current sentence
        current_sentences.append(sent)
        current_tokens += sent_tokens

    # Finalize remaining chunk for this page
    if current_sentences:
        chunk_text_content = " ".join(current_sentences)
        chunks.append(ChunkData(
            content=chunk_text_content,
            chunk_index=chunk_index,
            chunk_type="text",
            page_number=page_number,
            token_count=current_tokens,
        ))

    return chunks


def chunk_text(pages: list[PageText], filename: str = "") -> list[ChunkData]:
    """
    Chunk text content (from PDF or TXT) into page-boundary-aware chunks.

    Strategy:
    1. Process each page independently — content NEVER crosses page boundaries.
    2. Each page produces one or more chunks, each tagged with exact page_number.
    3. If a page fits within MAX_CHUNK_TOKENS (512), it becomes a single chunk.
    4. If a page exceeds MAX_CHUNK_TOKENS, it's split into multiple chunks
       with intra-page overlap (no cross-page overlap).
    5. This ensures citation accuracy: every chunk maps to exactly one page.
    """
    chunks: list[ChunkData] = []
    chunk_index = 0

    for page in pages:
        text = page.text.strip()
        if not text:
            continue

        sentences = _split_into_sentences(text)
        if not sentences:
            continue

        page_chunks = _chunk_single_page(sentences, page.page_number, chunk_index)
        chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    return chunks


def chunk_text_from_string(text: str, filename: str = "") -> list[ChunkData]:
    """
    Convenience wrapper for chunking a plain text string (TXT files).
    Wraps it as a single "page" and delegates to chunk_text.
    """
    pages = [PageText(page_number=1, text=text)]
    return chunk_text(pages, filename=filename)
