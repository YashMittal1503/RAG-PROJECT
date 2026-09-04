"""
Document parsers for PDF, TXT, and spreadsheet (XLSX/CSV) files.

Each parser takes raw file bytes and returns structured content
ready for the chunking step.
"""

import io
import logging
from dataclasses import dataclass, field

import chardet
import fitz  # PyMuPDF
import pandas as pd

logger = logging.getLogger(__name__)


# ── Data classes for parser output ────────────────────────────────────────

@dataclass
class PageText:
    """A single page of extracted text from a PDF."""
    page_number: int       # 1-indexed
    text: str


@dataclass
class SheetData:
    """Parsed data from a single spreadsheet sheet."""
    name: str              # Sheet name (or "Sheet1" for CSV)
    df: pd.DataFrame
    column_names: list[str] = field(default_factory=list)
    column_dtypes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.column_names:
            self.column_names = list(self.df.columns)
        if not self.column_dtypes:
            self.column_dtypes = {
                col: str(dtype) for col, dtype in self.df.dtypes.items()
            }


# ── PDF parser ────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> list[PageText]:
    """
    Extract text from a PDF page by page using PyMuPDF.

    Raises ValueError if the PDF has no extractable text (e.g., scanned images).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[PageText] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # Plain text extraction, no OCR
        if text.strip():
            pages.append(PageText(page_number=page_num + 1, text=text.strip()))

    doc.close()

    # Check if we got any meaningful text
    total_chars = sum(len(p.text) for p in pages)
    if total_chars < 50:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may contain only scanned images, which are not supported yet. "
            "Please use a PDF with selectable text."
        )

    return pages


# ── TXT parser ────────────────────────────────────────────────────────────

def parse_txt(file_bytes: bytes) -> str:
    """
    Decode a text file with encoding detection.

    Tries UTF-8 first (most common), then falls back to chardet detection,
    and finally to latin-1 (which never fails but may produce garbled text).
    """
    # Try UTF-8 first — it's by far the most common encoding
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # Use chardet to detect the encoding
    detection = chardet.detect(file_bytes)
    detected_encoding = detection.get("encoding")

    if detected_encoding:
        try:
            return file_bytes.decode(detected_encoding)
        except (UnicodeDecodeError, LookupError):
            pass

    # Final fallback: latin-1 (ISO 8859-1) accepts any byte sequence
    logger.warning("Falling back to latin-1 encoding for text file")
    return file_bytes.decode("latin-1")


# ── Spreadsheet parser ────────────────────────────────────────────────────

def parse_spreadsheet(file_bytes: bytes, file_type: str) -> list[SheetData]:
    """
    Parse an XLSX or CSV file into structured sheet data.

    For XLSX: reads all sheets.
    For CSV: reads as a single "sheet".

    Returns a list of SheetData objects.
    """
    buffer = io.BytesIO(file_bytes)
    sheets: list[SheetData] = []

    if file_type == "csv":
        # CSV is always a single sheet
        try:
            df = pd.read_csv(buffer)
        except Exception as e:
            raise ValueError(f"Could not parse CSV file: {e}")

        if df.empty:
            raise ValueError("The CSV file is empty or has no readable data.")

        sheets.append(SheetData(name="Sheet1", df=df))

    elif file_type == "xlsx":
        # Read all sheets from the Excel file
        try:
            excel_file = pd.ExcelFile(buffer, engine="openpyxl")
        except Exception as e:
            raise ValueError(f"Could not open XLSX file: {e}")

        if not excel_file.sheet_names:
            raise ValueError("The XLSX file has no sheets.")

        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            if not df.empty:
                sheets.append(SheetData(name=sheet_name, df=df))

        if not sheets:
            raise ValueError("All sheets in the XLSX file are empty.")

    else:
        raise ValueError(f"Unsupported spreadsheet type: {file_type}")

    return sheets
