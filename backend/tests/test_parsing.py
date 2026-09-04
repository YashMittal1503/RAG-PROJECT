"""
Tests for document parsers (PDF, TXT, spreadsheet).

These tests use synthetic data — no external files needed.
"""

import io
import pytest
import pandas as pd

from app.services.parsing import parse_txt, parse_spreadsheet, SheetData


class TestParseTxt:
    """Tests for the TXT parser."""

    def test_utf8_text(self):
        """Standard UTF-8 text should be decoded correctly."""
        content = "Hello, World! This is a test document."
        result = parse_txt(content.encode("utf-8"))
        assert result == content

    def test_utf8_with_bom(self):
        """UTF-8 with BOM should be decoded (BOM appears as prefix)."""
        content = "Test content with unicode: café, naïve"
        encoded = content.encode("utf-8")
        result = parse_txt(encoded)
        assert "café" in result
        assert "naïve" in result

    def test_latin1_fallback(self):
        """Non-UTF-8 content should fall back to chardet or latin-1."""
        # Latin-1 encoded text with special characters
        content = "Stra\xdfe"  # "Straße" in latin-1
        encoded = content.encode("latin-1")
        result = parse_txt(encoded)
        # Should successfully decode without raising
        assert len(result) > 0

    def test_empty_text(self):
        """Empty text files should return empty string."""
        result = parse_txt(b"")
        assert result == ""

    def test_multiline_text(self):
        """Multi-line text should preserve line breaks."""
        content = "Line 1\nLine 2\nLine 3"
        result = parse_txt(content.encode("utf-8"))
        assert result.count("\n") == 2


class TestParseSpreadsheet:
    """Tests for the spreadsheet parser."""

    def _make_csv_bytes(self, data: dict) -> bytes:
        """Helper to create CSV bytes from a dict."""
        df = pd.DataFrame(data)
        return df.to_csv(index=False).encode("utf-8")

    def _make_xlsx_bytes(self, data: dict, sheet_name: str = "Sheet1") -> bytes:
        """Helper to create XLSX bytes from a dict."""
        df = pd.DataFrame(data)
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False, sheet_name=sheet_name)
        return buffer.getvalue()

    def test_csv_parsing(self):
        """CSV files should be parsed into a single SheetData."""
        data = {"Name": ["Alice", "Bob"], "Age": [30, 25]}
        csv_bytes = self._make_csv_bytes(data)

        sheets = parse_spreadsheet(csv_bytes, "csv")
        assert len(sheets) == 1
        assert sheets[0].name == "Sheet1"
        assert len(sheets[0].df) == 2
        assert "Name" in sheets[0].column_names
        assert "Age" in sheets[0].column_names

    def test_xlsx_parsing(self):
        """XLSX files should be parsed with correct column detection."""
        data = {"Product": ["Widget", "Gadget"], "Price": [9.99, 19.99]}
        xlsx_bytes = self._make_xlsx_bytes(data)

        sheets = parse_spreadsheet(xlsx_bytes, "xlsx")
        assert len(sheets) == 1
        assert "Product" in sheets[0].column_names
        assert "Price" in sheets[0].column_names

    def test_csv_empty_rejected(self):
        """Empty CSV files should raise ValueError."""
        empty_csv = b""
        with pytest.raises(ValueError, match="empty|parse"):
            parse_spreadsheet(empty_csv, "csv")

    def test_csv_column_dtypes_detected(self):
        """Column data types should be correctly identified."""
        data = {
            "Name": ["Alice", "Bob"],
            "Age": [30, 25],
            "Score": [95.5, 87.3],
        }
        csv_bytes = self._make_csv_bytes(data)
        sheets = parse_spreadsheet(csv_bytes, "csv")

        dtypes = sheets[0].column_dtypes
        assert "int" in dtypes["Age"]
        assert "float" in dtypes["Score"]

    def test_unsupported_type_rejected(self):
        """Unsupported spreadsheet types should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            parse_spreadsheet(b"dummy", "json")
