"""
Text and spreadsheet chunking logic.

Two main strategies:
1. Text chunking (PDF/TXT): page-boundary-aware splitting with token-based sizing.
   Each PDF page produces its own chunk(s) — content is NEVER merged across pages.
   This ensures accurate page citations and respects the embedding model's context window.
2. Spreadsheet chunking (XLSX/CSV): row-batch chunks + one summary chunk per sheet

Every chunk is a ChunkData object that carries its content and metadata.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

import tiktoken

from app.services.parsing import PageText, SheetData

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
SPREADSHEET_ROWS_PER_CHUNK = 20  # Default rows per chunk for spreadsheets


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


# ── Spreadsheet chunking ─────────────────────────────────────────────────

def chunk_spreadsheet(
    sheet: SheetData,
    filename: str = "",
    rows_per_chunk: int = SPREADSHEET_ROWS_PER_CHUNK,
) -> list[ChunkData]:
    """
    Chunk a spreadsheet sheet into row-batch chunks + one summary chunk.

    Row-batch chunks:
    - Each chunk contains a batch of rows with column headers repeated
      so the chunk is self-describing out of context.

    Summary chunk:
    - Contains column names, data types, row count, and precomputed
      aggregates for numeric columns (sum, mean, min, max).
    - Tagged with chunk_type="summary" for retrieval biasing.
    """
    chunks: list[ChunkData] = []
    chunk_index = 0
    df = sheet.df
    n_rows = len(df)
    num_cols = len(df.columns)
    col_header = " | ".join(str(c) for c in df.columns)

    # Scale rows per chunk for wide tables so chunks don't exceed token limits
    effective_rows_per_chunk = rows_per_chunk
    if num_cols > 10:
        effective_rows_per_chunk = max(5, min(rows_per_chunk, 300 // num_cols))

    # ── Row-level chunks ──────────────────────────────────────────────
    for start_row in range(0, n_rows, effective_rows_per_chunk):
        end_row = min(start_row + effective_rows_per_chunk, n_rows)
        batch = df.iloc[start_row:end_row]

        # Format each row as pipe-delimited values
        row_lines = []
        for _, row in batch.iterrows():
            row_lines.append(" | ".join(str(v) for v in row.values))

        chunk_content = (
            f"Source: {filename}, Sheet: {sheet.name}, "
            f"Rows {start_row + 1}-{end_row}\n"
            f"Columns: {col_header}\n"
            f"---\n"
            + "\n".join(row_lines)
        )

        token_count = count_tokens(chunk_content)

        chunks.append(ChunkData(
            content=chunk_content,
            chunk_index=chunk_index,
            chunk_type="text",
            row_range_start=start_row + 1,   # 1-indexed for display
            row_range_end=end_row,
            token_count=token_count,
        ))
        chunk_index += 1

    # ── Summary chunk ─────────────────────────────────────────────────
    summary_lines = [
        f"Summary of {filename}, Sheet: {sheet.name}",
        f"Total rows: {n_rows}",
        "",
        "Columns and data types:",
    ]

    for col in df.columns:
        dtype = str(df[col].dtype)
        summary_lines.append(f"  - {col} ({dtype})")

    # Compute aggregates for numeric columns (cap at 15 columns to prevent oversized summary)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        summary_lines.append("")
        summary_lines.append("Numeric column aggregates:")
        for col in numeric_cols[:15]:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                summary_lines.append(
                    f"  - {col}: "
                    f"sum={col_data.sum():.4g}, "
                    f"mean={col_data.mean():.4g}, "
                    f"min={col_data.min():.4g}, "
                    f"max={col_data.max():.4g}, "
                    f"count={len(col_data)}"
                )
        if len(numeric_cols) > 15:
            summary_lines.append(f"  ... and {len(numeric_cols) - 15} more numeric columns")

    summary_content = "\n".join(summary_lines)
    summary_tokens = count_tokens(summary_content)

    chunks.append(ChunkData(
        content=summary_content,
        chunk_index=chunk_index,
        chunk_type="summary",
        token_count=summary_tokens,
    ))

    return chunks
