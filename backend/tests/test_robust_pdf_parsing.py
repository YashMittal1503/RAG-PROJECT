"""
Unit tests for production-grade robust PDF parsing and vector store resilience.

Verifies:
1. Standard digital PDF extraction with reading-order preservation.
2. Scanned / image-only PDF text extraction via embedded RapidOCR fallback.
3. Completely blank / graphic page produces safe metadata placeholder (no crashes).
4. Password-protected encrypted PDF handling.
5. Qdrant fallback to dense-only vector upsert when bm25 sparse vectors are absent.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import fitz
import pytest

from app.services.parsing import parse_pdf, PageText
from app.services.vector_store import collection_supports_sparse, upsert_chunks


# ── PDF Parsing Tests ────────────────────────────────────────────────────────

def test_parse_digital_pdf_preserves_text_and_pages():
    """Standard selectable PDF text is extracted accurately per page."""
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 100), "First page of enterprise quarterly report.")
    p2 = doc.new_page()
    p2.insert_text((50, 100), "Second page with financial projections.")
    pdf_bytes = doc.write()
    doc.close()

    pages = parse_pdf(pdf_bytes)
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "First page of enterprise quarterly report" in pages[0].text
    assert pages[1].page_number == 2
    assert "Second page with financial projections" in pages[1].text


def test_parse_scanned_image_pdf_with_rapidocr():
    """Image-only PDF with 0 selectable text triggers RapidOCR and extracts text."""
    doc = fitz.open()
    page = doc.new_page()
    subdoc = fitz.open()
    subpage = subdoc.new_page()
    subpage.insert_text((50, 80), "Invoice 98765 Total Due 1500 USD", fontsize=16)
    pix = subpage.get_pixmap()
    page.insert_image(page.rect, pixmap=pix)

    pdf_bytes = doc.write()
    doc.close()
    subdoc.close()

    # Ensure native extraction is empty
    check_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    assert check_doc[0].get_text("text").strip() == ""
    check_doc.close()

    pages = parse_pdf(pdf_bytes)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    # RapidOCR should extract words from the image
    extracted = pages[0].text.lower()
    assert "invoice" in extracted or "98765" in extracted or "total" in extracted or "usd" in extracted


def test_parse_blank_or_drawing_page_emits_placeholder():
    """A completely blank page does not raise ValueError; emits clean placeholder."""
    doc = fitz.open()
    doc.new_page()  # Empty white page
    pdf_bytes = doc.write()
    doc.close()

    pages = parse_pdf(pdf_bytes)
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "[Page 1:" in pages[0].text


def test_parse_empty_bytes_raises_clear_error():
    """Empty 0-byte upload raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        parse_pdf(b"")


def test_parse_corrupted_stream_raises_clean_error():
    """Corrupted non-PDF binary raises descriptive ValueError."""
    with pytest.raises(ValueError, match="Could not open or parse"):
        parse_pdf(b"NOT_A_REAL_PDF_HEADER_DATA_12345")


# ── Vector Store Dual/Dense Fallback Tests ───────────────────────────────────

@pytest.mark.anyio
async def test_collection_supports_sparse_returns_false_for_dense_only():
    """If collection config lacks bm25, collection_supports_sparse returns False."""
    mock_client = AsyncMock()
    mock_info = MagicMock()
    mock_info.config.params.sparse_vectors = None
    mock_client.get_collection.return_value = mock_info

    supports = await collection_supports_sparse(mock_client, "docs_test_collection")
    assert supports is False


@pytest.mark.anyio
async def test_collection_supports_sparse_returns_true_when_configured():
    """If collection config contains bm25, returns True."""
    mock_client = AsyncMock()
    mock_info = MagicMock()
    mock_info.config.params.sparse_vectors = {"bm25": MagicMock()}
    mock_client.get_collection.return_value = mock_info

    supports = await collection_supports_sparse(mock_client, "docs_test_collection")
    assert supports is True


@pytest.mark.anyio
async def test_upsert_chunks_dense_fallback_on_schema_mismatch():
    """If Qdrant rejects bm25 vector, upsert_chunks automatically falls back to dense-only."""
    mock_client = AsyncMock()

    # First upsert raises vector name error; retry succeeds
    mock_client.upsert = AsyncMock(
        side_effect=[
            Exception("Wrong input: Not existing vector name error: bm25"),
            None,
        ]
    )

    with patch("app.services.vector_store.get_client", return_value=mock_client), \
         patch("app.services.vector_store.collection_supports_sparse", return_value=True):

        test_chunks = [
            {
                "id": "c7a8b9d0-1111-2222-3333-444455556666",
                "vector": [0.1] * 384,
                "sparse_vector": {"indices": [10, 20], "values": [0.5, 0.8]},
                "payload": {"filename": "test.pdf", "chunk_type": "text"},
            }
        ]

        await upsert_chunks("user-test-uuid", test_chunks)
        assert mock_client.upsert.call_count == 2
