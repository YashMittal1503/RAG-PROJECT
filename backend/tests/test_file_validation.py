"""
Tests for file type validation (magic bytes).

Tests that:
- Valid file types are correctly identified
- Invalid/unsupported files are rejected
- Macro-enabled formats (.xlsm) are explicitly blocked
- Edge cases (empty files, wrong extensions) are handled
"""

import pytest
from app.utils.file_validation import validate_file_type


class TestValidateFileType:
    """Tests for the validate_file_type function."""

    def test_pdf_valid(self):
        """PDF magic bytes (%PDF) should be recognized."""
        # Real PDF starts with %PDF-1.x
        pdf_header = b"%PDF-1.4 fake pdf content" + b"\x00" * 2048
        file_type, mime = validate_file_type(pdf_header, "test.pdf")
        assert file_type == "pdf"

    def test_txt_valid(self):
        """Plain text files should be recognized."""
        txt_content = b"Hello, this is a plain text file with enough content to detect."
        file_type, mime = validate_file_type(txt_content, "readme.txt")
        assert file_type == "txt"

    def test_csv_with_txt_extension_fallback(self):
        """CSV identified as text/plain with .csv extension should map to csv."""
        csv_content = b"name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        file_type, mime = validate_file_type(csv_content, "data.csv")
        assert file_type == "csv"

    def test_unsupported_file_type_rejected(self):
        """Binary files that aren't in the allowlist should be rejected."""
        # ELF binary header (Linux executable)
        elf_header = b"\x7fELF" + b"\x00" * 2048
        with pytest.raises(ValueError, match="Unsupported file type"):
            validate_file_type(elf_header, "program.exe")

    def test_xlsm_blocked_by_extension(self):
        """Macro-enabled .xlsm files should be blocked even before magic-byte check."""
        # Content doesn't matter — extension check catches it first
        fake_content = b"PK\x03\x04" + b"\x00" * 2048
        with pytest.raises(ValueError, match="Macro-enabled"):
            validate_file_type(fake_content, "budget.xlsm")

    def test_docm_blocked_by_extension(self):
        """.docm files should also be blocked."""
        fake_content = b"PK\x03\x04" + b"\x00" * 2048
        with pytest.raises(ValueError, match="Macro-enabled"):
            validate_file_type(fake_content, "report.docm")

    def test_xlsx_identified_as_zip(self):
        """XLSX files (which are ZIP containers) with .xlsx extension should be accepted."""
        # Real XLSX starts with PK (ZIP signature)
        zip_header = b"PK\x03\x04" + b"\x00" * 2048
        file_type, mime = validate_file_type(zip_header, "data.xlsx")
        assert file_type == "xlsx"

    def test_empty_filename_handling(self):
        """Should handle files with unusual names gracefully."""
        txt_content = b"Some plain text content here for detection purposes."
        file_type, mime = validate_file_type(txt_content, "noextension")
        assert file_type == "txt"  # Detected as text/plain by content

    def test_case_insensitive_extension_check(self):
        """Extension blocking should be case-insensitive."""
        fake_content = b"PK\x03\x04" + b"\x00" * 2048
        with pytest.raises(ValueError, match="Macro-enabled"):
            validate_file_type(fake_content, "BUDGET.XLSM")
