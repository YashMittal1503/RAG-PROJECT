"""
Text and spreadsheet chunking logic.

Two main strategies:
1. Text chunking (PDF/TXT): paragraph-aware splitting with token-based sizing
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
MIN_CHUNK_TOKENS = 200
TARGET_CHUNK_TOKENS = 500
MAX_CHUNK_TOKENS = 800
OVERLAP_FRACTION = 0.10        # ~10% overlap between consecutive chunks
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
                        if len(w_buf) >= 300:  # ~380-450 tokens
                            sentences.append(" ".join(w_buf))
                            w_buf = []
                    if w_buf:
                        sentences.append(" ".join(w_buf))
                else:
                    sentences.append(sp_clean)
        else:
            sentences.append(cleaned)
    return sentences


def chunk_text(pages: list[PageText], filename: str = "") -> list[ChunkData]:
    """
    Chunk text content (from PDF or TXT) into overlapping chunks.

    Strategy:
    1. Split all pages into sentences (with oversized sentence protection).
    2. Pre-compute token counts to prevent redundant encoding.
    3. Accumulate sentences into bounded chunks with smooth overlap.
    4. Deterministic forward progress via for-loop (guaranteed no infinite loops).
    """
    chunks: list[ChunkData] = []
    chunk_index = 0

    # Build list of (sentence, page_number, token_count) tuples
    sentence_pages: list[tuple[str, int, int]] = []
    for page in pages:
        sentences = _split_into_sentences(page.text)
        for sent in sentences:
            sentence_pages.append((sent, page.page_number, count_tokens(sent)))

    if not sentence_pages:
        return chunks

    overlap_tokens = int(TARGET_CHUNK_TOKENS * OVERLAP_FRACTION)
    current_sentences: list[str] = []
    current_tokens = 0
    current_pages: set[int] = set()

    for sent, page_num, sent_tokens in sentence_pages:
        # If adding this sentence exceeds MAX_CHUNK_TOKENS and we already have content:
        if current_tokens + sent_tokens > MAX_CHUNK_TOKENS and current_sentences:
            chunk_text_content = " ".join(current_sentences)
            primary_page = min(current_pages) if current_pages else None
            chunks.append(ChunkData(
                content=chunk_text_content,
                chunk_index=chunk_index,
                chunk_type="text",
                page_number=primary_page,
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
            current_pages = {page_num} if overlap_sents else set()

        # Add current sentence
        current_sentences.append(sent)
        current_tokens += sent_tokens
        current_pages.add(page_num)

    # Finalize remaining chunk
    if current_sentences:
        chunk_text_content = " ".join(current_sentences)
        primary_page = min(current_pages) if current_pages else None
        chunks.append(ChunkData(
            content=chunk_text_content,
            chunk_index=chunk_index,
            chunk_type="text",
            page_number=primary_page,
            token_count=current_tokens,
        ))

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
