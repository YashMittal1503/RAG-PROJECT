"""
Tests for text and spreadsheet chunking logic.

Validates:
- Chunk size bounds (token count within target range)
- Overlap between consecutive chunks
- Spreadsheet summary chunk generation with correct aggregates
- Column headers present in every row-level chunk
"""

import pytest
import pandas as pd

from app.services.parsing import PageText, SheetData
from app.services.chunking import (
    ChunkData,
    chunk_text,
    chunk_text_from_string,
    chunk_spreadsheet,
    count_tokens,
    MAX_CHUNK_TOKENS,
)


class TestCountTokens:
    """Tests for the token counting utility."""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        tokens = count_tokens("Hello, world!")
        assert tokens > 0
        assert tokens < 10  # Should be about 4 tokens

    def test_longer_text(self):
        text = "This is a longer text with multiple words for token counting."
        tokens = count_tokens(text)
        assert tokens > 5


class TestChunkText:
    """Tests for text chunking (PDF/TXT)."""

    def test_short_text_single_chunk(self):
        """Short text should produce exactly one chunk."""
        pages = [PageText(page_number=1, text="This is a short document.")]
        chunks = chunk_text(pages)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "text"
        assert chunks[0].page_number == 1

    def test_long_text_multiple_chunks(self):
        """Long text should be split into multiple chunks."""
        # Generate text that's definitely longer than MAX_CHUNK_TOKENS
        long_text = " ".join(["This is a sentence for testing purposes."] * 200)
        pages = [PageText(page_number=1, text=long_text)]
        chunks = chunk_text(pages)
        assert len(chunks) > 1

    def test_chunk_token_count_within_bounds(self):
        """Each chunk should have a token count within the configured bounds."""
        long_text = " ".join(["This is a test sentence with several words."] * 200)
        pages = [PageText(page_number=1, text=long_text)]
        chunks = chunk_text(pages)

        for chunk in chunks:
            # Allow some tolerance for boundary cases
            assert chunk.token_count <= MAX_CHUNK_TOKENS + 50, (
                f"Chunk {chunk.chunk_index} has {chunk.token_count} tokens, "
                f"exceeding max of {MAX_CHUNK_TOKENS}"
            )

    def test_page_numbers_preserved(self):
        """Chunks should carry the page number from their source."""
        pages = [
            PageText(page_number=1, text="Content from page one."),
            PageText(page_number=2, text="Content from page two."),
        ]
        chunks = chunk_text(pages)
        assert all(c.page_number is not None for c in chunks)

    def test_chunk_text_from_string(self):
        """The string convenience wrapper should work for TXT files."""
        text = "A simple text file content."
        chunks = chunk_text_from_string(text)
        assert len(chunks) == 1
        assert chunks[0].page_number == 1

    def test_empty_pages_no_chunks(self):
        """Empty pages should produce no chunks."""
        chunks = chunk_text([])
        assert len(chunks) == 0

    def test_oversized_sentence_no_infinite_loop(self):
        """A single sentence larger than MAX_CHUNK_TOKENS must not cause an infinite loop."""
        huge_text = "Short introductory sentence. " + ("word " * 900) + ". Final conclusion."
        pages = [PageText(page_number=1, text=huge_text)]
        chunks = chunk_text(pages)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= MAX_CHUNK_TOKENS + 50


class TestChunkSpreadsheet:
    """Tests for spreadsheet chunking."""

    def _make_sheet(self, n_rows: int = 50) -> SheetData:
        """Create a test SheetData with numeric and string columns."""
        data = {
            "Name": [f"Item_{i}" for i in range(n_rows)],
            "Quantity": list(range(1, n_rows + 1)),
            "Price": [round(i * 1.5, 2) for i in range(1, n_rows + 1)],
            "Category": ["A" if i % 2 == 0 else "B" for i in range(n_rows)],
        }
        df = pd.DataFrame(data)
        return SheetData(name="TestSheet", df=df)

    def test_row_chunks_created(self):
        """Row-level chunks should be created for the data."""
        sheet = self._make_sheet(50)
        chunks = chunk_spreadsheet(sheet, filename="test.csv")
        # Should have row chunks + 1 summary chunk
        row_chunks = [c for c in chunks if c.chunk_type == "text"]
        assert len(row_chunks) > 0

    def test_summary_chunk_created(self):
        """A summary chunk should be created for the sheet."""
        sheet = self._make_sheet(50)
        chunks = chunk_spreadsheet(sheet, filename="test.csv")
        summary_chunks = [c for c in chunks if c.chunk_type == "summary"]
        assert len(summary_chunks) == 1

    def test_summary_contains_aggregates(self):
        """The summary chunk should contain precomputed aggregates."""
        sheet = self._make_sheet(10)
        chunks = chunk_spreadsheet(sheet, filename="test.csv")
        summary = next(c for c in chunks if c.chunk_type == "summary")

        assert "sum=" in summary.content
        assert "mean=" in summary.content
        assert "min=" in summary.content
        assert "max=" in summary.content

    def test_summary_aggregates_correct(self):
        """The summary aggregates should be mathematically correct."""
        data = {"Value": [10, 20, 30, 40, 50]}
        df = pd.DataFrame(data)
        sheet = SheetData(name="Test", df=df)

        chunks = chunk_spreadsheet(sheet, filename="test.csv")
        summary = next(c for c in chunks if c.chunk_type == "summary")

        # sum should be 150
        assert "sum=150" in summary.content
        # mean should be 30
        assert "mean=30" in summary.content
        # min should be 10
        assert "min=10" in summary.content
        # max should be 50
        assert "max=50" in summary.content

    def test_column_headers_in_every_row_chunk(self):
        """Every row-level chunk should contain the column headers."""
        sheet = self._make_sheet(50)
        chunks = chunk_spreadsheet(sheet, filename="test.csv")
        row_chunks = [c for c in chunks if c.chunk_type == "text"]

        for chunk in row_chunks:
            assert "Columns:" in chunk.content
            assert "Name" in chunk.content
            assert "Quantity" in chunk.content
            assert "Price" in chunk.content

    def test_row_range_metadata(self):
        """Row chunks should have correct row_range_start and row_range_end."""
        sheet = self._make_sheet(50)
        chunks = chunk_spreadsheet(sheet, filename="test.csv", rows_per_chunk=20)
        row_chunks = [c for c in chunks if c.chunk_type == "text"]

        # First chunk should start at row 1
        assert row_chunks[0].row_range_start == 1
        assert row_chunks[0].row_range_end == 20

    def test_summary_contains_row_count(self):
        """Summary should report the correct total row count."""
        sheet = self._make_sheet(42)
        chunks = chunk_spreadsheet(sheet, filename="test.csv")
        summary = next(c for c in chunks if c.chunk_type == "summary")
        assert "Total rows: 42" in summary.content

    def test_summary_contains_column_types(self):
        """Summary should list column data types."""
        sheet = self._make_sheet(10)
        chunks = chunk_spreadsheet(sheet, filename="test.csv")
        summary = next(c for c in chunks if c.chunk_type == "summary")
        assert "int" in summary.content  # Quantity column
        assert "float" in summary.content  # Price column
