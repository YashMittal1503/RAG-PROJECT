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


# ── PDF parser with OCR Fallback ──────────────────────────────────────────

_ocr_engine = None


def _get_ocr_engine():
    """Lazy initialize RapidOCR engine singleton."""
    global _ocr_engine
    if _ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
            logger.info("RapidOCR ONNX engine initialized successfully for PDF OCR fallback.")
        except Exception as e:
            logger.warning(f"Could not initialize RapidOCR engine: {e}")
            _ocr_engine = False
    return _ocr_engine if _ocr_engine is not False else None


def _ocr_page(page: fitz.Page) -> str:
    """
    Render a PDF page to a high-resolution pixmap image and run RapidOCR.
    Used when native text extraction yields fewer than 50 characters (scanned pages).
    """
    ocr = _get_ocr_engine()
    if not ocr:
        return ""
    try:
        import numpy as np
        # Render at 150 DPI (matrix zoom = 150/72 ~ 2.08) for optimal speed and accuracy
        zoom = 150 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        result, _ = ocr(img)
        if not result:
            return ""

        lines = []
        for item in result:
            if len(item) >= 2 and item[1]:
                text_str = str(item[1]).strip()
                if text_str:
                    lines.append(text_str)
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"OCR failed for page {page.number + 1}: {e}")
        return ""


def parse_pdf(file_bytes: bytes) -> list[PageText]:
    """
    Extract text from a PDF page by page with multi-tier layout and OCR fallback.

    1. Checks for empty byte streams.
    2. Opens with PyMuPDF; attempts decryption with blank passwords if encrypted.
    3. Extracts text using layout-aware reading order (sort=True) to preserve multi-column formatting.
    4. If a page has < 50 characters of native text, falls back to embedded RapidOCR on that page.
    5. If a page is purely an illustration/blank, produces a clean metadata placeholder.
    6. Catches page-level errors individually so a corrupt page never aborts the document.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("The uploaded PDF file is completely empty (0 bytes).")

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        # Fallback repair attempt
        try:
            doc = fitz.open(stream=file_bytes)
        except Exception:
            raise ValueError(f"Could not open or parse PDF file: {e}")

    # Handle encryption/password protection
    if doc.is_encrypted:
        authenticated = False
        for pwd in ["", " ", "123456", "1234"]:
            if doc.authenticate(pwd):
                authenticated = True
                break
        if not authenticated:
            doc.close()
            raise ValueError(
                "This PDF is password-protected and encrypted. "
                "Please upload an unlocked/decrypted version of the document."
            )

    pages: list[PageText] = []
    total_pages = len(doc)

    if total_pages == 0:
        doc.close()
        raise ValueError("The uploaded PDF file contains 0 pages.")

    for page_num in range(total_pages):
        try:
            page = doc[page_num]
            # 1. Native text extraction with layout sorting (preserves column reading order)
            text = page.get_text("text", sort=True).strip()

            # 2. If page has little to no selectable text (scanned page, form, image), run OCR fallback
            if len(text) < 50:
                ocr_text = _ocr_page(page).strip()
                if ocr_text:
                    if text:
                        text = f"{text}\n\n{ocr_text}"
                    else:
                        text = ocr_text

            # 3. If still empty (e.g. blank page, drawing, or unreadable diagram), use safe placeholder
            if not text:
                text = f"[Page {page_num + 1}: Non-text graphic, illustration, or blank page]"

            pages.append(PageText(page_number=page_num + 1, text=text))

        except Exception as page_err:
            logger.warning(f"Error extracting page {page_num + 1}: {page_err}. Emitting placeholder.")
            pages.append(PageText(page_number=page_num + 1, text=f"[Page {page_num + 1}: Content could not be rendered]"))

    doc.close()
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
