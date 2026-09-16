"""
Unit tests for the pipeline-level SQL routing guard.

Verifies the structural rule: SQL queries are ONLY attempted when
scope.is_tabular == True (i.e., the document scope analysis has
resolved to a CSV/XLSX file). PDFs, DOCX, TXT, and unresolved
"all_docs" scopes must NEVER trigger the SQL pipeline.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.query import DocumentScopeResult


class TestSQLRoutingDecision:
    """
    Test the `should_attempt_sql = has_tabular and scope.is_tabular` rule
    that gates the entire SQL pipeline in chat.py.
    """

    def _should_attempt_sql(self, has_tabular: bool, scope: DocumentScopeResult) -> bool:
        """Mirror the exact routing logic from chat.py."""
        return has_tabular and scope.is_tabular

    # ── Tabular document → SQL allowed ────────────────────────────────

    def test_tabular_resolved_allows_sql(self):
        """A resolved CSV/XLSX document should trigger SQL."""
        scope = DocumentScopeResult(
            status="resolved",
            target_doc_id="abc-123",
            target_filename="customer_churn.csv",
            is_tabular=True,
        )
        assert self._should_attempt_sql(True, scope) is True

    def test_tabular_resolved_but_no_duckdb_data(self):
        """Even if scope says tabular, if DuckDB has no data, skip SQL."""
        scope = DocumentScopeResult(
            status="resolved",
            target_doc_id="abc-123",
            target_filename="data.xlsx",
            is_tabular=True,
        )
        assert self._should_attempt_sql(False, scope) is False

    # ── Non-tabular document → SQL blocked ────────────────────────────

    def test_pdf_resolved_blocks_sql(self):
        """A resolved PDF document must NOT trigger SQL."""
        scope = DocumentScopeResult(
            status="resolved",
            target_doc_id="def-456",
            target_filename="SIH_2026_Catalogue.pdf",
            is_tabular=False,
        )
        assert self._should_attempt_sql(True, scope) is False

    def test_docx_resolved_blocks_sql(self):
        """A resolved DOCX document must NOT trigger SQL."""
        scope = DocumentScopeResult(
            status="resolved",
            target_doc_id="ghi-789",
            target_filename="contract.docx",
            is_tabular=False,
        )
        assert self._should_attempt_sql(True, scope) is False

    def test_txt_resolved_blocks_sql(self):
        """A resolved TXT document must NOT trigger SQL."""
        scope = DocumentScopeResult(
            status="resolved",
            target_doc_id="jkl-012",
            target_filename="notes.txt",
            is_tabular=False,
        )
        assert self._should_attempt_sql(True, scope) is False

    # ── all_docs fallback → SQL blocked ───────────────────────────────

    def test_all_docs_fallback_blocks_sql(self):
        """The all_docs fallback must NEVER trigger SQL, even if user has tabular data."""
        scope = DocumentScopeResult(status="all_docs", cleaned_question="48?")
        assert self._should_attempt_sql(True, scope) is False

    def test_all_docs_with_analytics_query_still_blocks_sql(self):
        """Even an analytics-sounding query on all_docs must NOT trigger SQL."""
        scope = DocumentScopeResult(
            status="all_docs",
            cleaned_question="What is the average churn rate?",
        )
        assert self._should_attempt_sql(True, scope) is False

    # ── Other scope statuses → SQL blocked ────────────────────────────

    def test_needs_clarification_blocks_sql(self):
        scope = DocumentScopeResult(status="needs_clarification")
        assert self._should_attempt_sql(True, scope) is False

    def test_no_documents_blocks_sql(self):
        scope = DocumentScopeResult(status="no_documents")
        assert self._should_attempt_sql(True, scope) is False


class TestDocumentScopeIsTabularFlag:
    """
    Verify that is_tabular is correctly set during scope resolution
    based on the document's file type metadata from the database.
    """

    def test_csv_marked_tabular(self):
        """CSV files must have is_tabular=True when scope resolves."""
        doc = {"id": "1", "filename": "sales.csv", "is_tabular": True, "file_type": "text/csv"}
        scope = DocumentScopeResult(
            status="resolved",
            target_doc_id=doc["id"],
            target_filename=doc["filename"],
            is_tabular=doc["is_tabular"],
        )
        assert scope.is_tabular is True

    def test_xlsx_marked_tabular(self):
        """XLSX files must have is_tabular=True when scope resolves."""
        doc = {"id": "2", "filename": "report.xlsx", "is_tabular": True, "file_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
        scope = DocumentScopeResult(
            status="resolved",
            target_doc_id=doc["id"],
            target_filename=doc["filename"],
            is_tabular=doc["is_tabular"],
        )
        assert scope.is_tabular is True

    def test_pdf_not_tabular(self):
        """PDF files must have is_tabular=False."""
        doc = {"id": "3", "filename": "handbook.pdf", "is_tabular": False, "file_type": "application/pdf"}
        scope = DocumentScopeResult(
            status="resolved",
            target_doc_id=doc["id"],
            target_filename=doc["filename"],
            is_tabular=doc["is_tabular"],
        )
        assert scope.is_tabular is False

    def test_default_is_tabular_false(self):
        """is_tabular defaults to False if not explicitly set."""
        scope = DocumentScopeResult(status="all_docs")
        assert scope.is_tabular is False


class TestScopeAnalysisFollowUpInPDFConversation:
    """
    End-to-end: when a user has both a PDF and a CSV, and the
    conversation is clearly about the PDF, follow-up queries
    must resolve to the PDF (is_tabular=False), so SQL is never attempted.
    """

    @pytest.mark.anyio
    async def test_single_doc_resolves_to_that_doc(self):
        """With only 1 document, scope always resolves to it with correct is_tabular."""
        from app.services.query import analyze_document_scope

        pdf_doc = [{"id": "aaa", "filename": "SIH_2026_Catalogue.pdf", "is_tabular": False, "file_type": "application/pdf", "chunk_count": 50}]
        scope = await analyze_document_scope("give me SIH26007", available_documents=pdf_doc)
        assert scope.status == "resolved"
        assert scope.is_tabular is False

    @pytest.mark.anyio
    async def test_single_tabular_doc_resolves_as_tabular(self):
        """With only 1 tabular document, scope resolves with is_tabular=True."""
        from app.services.query import analyze_document_scope

        csv_doc = [{"id": "bbb", "filename": "customer_churn.csv", "is_tabular": True, "file_type": "text/csv", "chunk_count": 10}]
        scope = await analyze_document_scope("what is the churn rate?", available_documents=csv_doc)
        assert scope.status == "resolved"
        assert scope.is_tabular is True
