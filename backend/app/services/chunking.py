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
# It's not the exact BGE tokenizer, but it's a reasonable approximation
# for chunk sizing purposes.
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
    Split text into sentences, preserving the delimiter.
    Simple heuristic: split on '. ', '! ', '? ', and newlines.
    """
    import re
    # Split on sentence-ending punctuation followed by whitespace,
    # or on double newlines (paragraph breaks)
    parts = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(pages: list[PageText], filename: str = "") -> list[ChunkData]:
    """
    Chunk text content (from PDF or TXT) into overlapping chunks.

    Strategy:
    1. Split all pages into sentences
    2. Accumulate sentences until we hit the target token count
    3. When a chunk is full, start the next one with overlap from the end
       of the previous chunk
    4. Track which page each sentence came from for citation metadata
    """
    chunks: list[ChunkData] = []
    chunk_index = 0

    # Build a list of (sentence, page_number) tuples
    sentence_pages: list[tuple[str, int]] = []
    for page in pages:
        sentences = _split_into_sentences(page.text)
        for sent in sentences:
            sentence_pages.append((sent, page.page_number))

    if not sentence_pages:
        return chunks

    overlap_tokens = int(TARGET_CHUNK_TOKENS * OVERLAP_FRACTION)
    current_sentences: list[str] = []
    current_tokens = 0
    current_pages: set[int] = set()

    i = 0
    while i < len(sentence_pages):
        sent, page_num = sentence_pages[i]
        sent_tokens = count_tokens(sent)

        # If a single sentence exceeds max tokens, we have to include it alone
        if sent_tokens > MAX_CHUNK_TOKENS and not current_sentences:
            chunks.append(ChunkData(
                content=sent,
                chunk_index=chunk_index,
                chunk_type="text",
                page_number=page_num,
                token_count=sent_tokens,
            ))
            chunk_index += 1
            i += 1
            continue

        # Would adding this sentence exceed the max?
        if current_tokens + sent_tokens > MAX_CHUNK_TOKENS and current_sentences:
            # Finalize the current chunk
            chunk_text_content = " ".join(current_sentences)
            # Use the most common page number, or the first one
            primary_page = min(current_pages) if current_pages else None
            chunks.append(ChunkData(
                content=chunk_text_content,
                chunk_index=chunk_index,
                chunk_type="text",
                page_number=primary_page,
                token_count=current_tokens,
            ))
            chunk_index += 1

            # Calculate overlap: take sentences from the end of current chunk
            # that total approximately overlap_tokens
            overlap_sents: list[str] = []
            overlap_tok = 0
            for s in reversed(current_sentences):
                st = count_tokens(s)
                if overlap_tok + st > overlap_tokens:
                    break
                overlap_sents.insert(0, s)
                overlap_tok += st

            current_sentences = overlap_sents
            current_tokens = overlap_tok
            current_pages = set()
            # Don't increment i — re-process this sentence with the new chunk
            continue

        # Add sentence to current chunk
        current_sentences.append(sent)
        current_tokens += sent_tokens
        current_pages.add(page_num)
        i += 1

    # Don't forget the last chunk
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
    col_header = " | ".join(str(c) for c in df.columns)

    # ── Row-level chunks ──────────────────────────────────────────────
    for start_row in range(0, n_rows, rows_per_chunk):
        end_row = min(start_row + rows_per_chunk, n_rows)
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

    # Compute aggregates for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        summary_lines.append("")
        summary_lines.append("Numeric column aggregates:")
        for col in numeric_cols:
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

    summary_content = "\n".join(summary_lines)
    summary_tokens = count_tokens(summary_content)

    chunks.append(ChunkData(
        content=summary_content,
        chunk_index=chunk_index,
        chunk_type="summary",
        token_count=summary_tokens,
    ))

    return chunks
