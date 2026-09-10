"""
Tests for Document Scope & Ambiguity Clarification Router
"""

import pytest
from app.services.query import (
    analyze_document_scope,
    DocumentScopeResult,
    _find_matching_doc_by_keyword,
)

SAMPLE_DOCS = [
    {"id": "doc-1", "filename": "2405.15793v3.pdf", "file_type": "pdf", "is_tabular": False},
    {"id": "doc-2", "filename": "The Ultimate Python Handbook.pdf", "file_type": "pdf", "is_tabular": False},
    {"id": "doc-3", "filename": "Verity-By-Colleen-Hoover.pdf", "file_type": "pdf", "is_tabular": False},
    {"id": "doc-4", "filename": "sample_titan_report.txt", "file_type": "txt", "is_tabular": False},
    {"id": "doc-5", "filename": "WA_Fn-UseC_-Telco-Customer-Churn.csv", "file_type": "csv", "is_tabular": True},
]


def test_find_matching_doc_by_keyword():
    # Matches SWE-bench paper
    res1 = _find_matching_doc_by_keyword("What are the findings in the SWE-bench paper?", SAMPLE_DOCS)
    assert res1 is not None
    assert res1["id"] == "doc-1"

    # Matches Verity
    res2 = _find_matching_doc_by_keyword("Tell me about Verity by Colleen Hoover", SAMPLE_DOCS)
    assert res2 is not None
    assert res2["id"] == "doc-3"

    # Matches Python Handbook
    res3 = _find_matching_doc_by_keyword("How do functions work in the Python handbook?", SAMPLE_DOCS)
    assert res3 is not None
    assert res3["id"] == "doc-2"

    # Matches Titan report
    res4 = _find_matching_doc_by_keyword("What is the battery range in the Titan report?", SAMPLE_DOCS)
    assert res4 is not None
    assert res4["id"] == "doc-4"

    # Ambiguous question without keywords returns None
    res5 = _find_matching_doc_by_keyword("What are the key findings?", SAMPLE_DOCS)
    assert res5 is None


@pytest.mark.anyio
async def test_scope_single_document_workspace():
    """If user only has 1 document, there is zero ambiguity."""
    single_doc = [SAMPLE_DOCS[0]]
    scope = await analyze_document_scope(
        question="What are the key findings?",
        chat_history=[],
        available_documents=single_doc,
    )
    assert scope.status == "resolved"
    assert scope.target_doc_id == "doc-1"
    assert scope.target_filename == "2405.15793v3.pdf"


@pytest.mark.anyio
async def test_scope_no_documents_workspace():
    """If user has no documents, returns no_documents."""
    scope = await analyze_document_scope(
        question="What are the key findings?",
        chat_history=[],
        available_documents=[],
    )
    assert scope.status == "no_documents"
    assert "upload a document" in scope.clarification_text.lower()


@pytest.mark.anyio
async def test_scope_at_mention():
    """@mention explicitly targets a document."""
    scope = await analyze_document_scope(
        question="@verity What happens in chapter 3?",
        chat_history=[],
        available_documents=SAMPLE_DOCS,
    )
    assert scope.status == "resolved"
    assert scope.target_doc_id == "doc-3"
    assert scope.target_filename == "Verity-By-Colleen-Hoover.pdf"
    assert "@verity" not in scope.cleaned_question


@pytest.mark.anyio
async def test_scope_cross_document_query():
    """Explicit cross-document queries search across all files."""
    scope = await analyze_document_scope(
        question="Compare all my documents and summarize them",
        chat_history=[],
        available_documents=SAMPLE_DOCS,
    )
    assert scope.status == "all_docs"


@pytest.mark.anyio
async def test_scope_answering_clarification():
    """When user replies to a clarification question, it resolves to the chosen doc."""
    chat_history = [
        {"role": "user", "content": "What are the key findings?"},
        {
            "role": "assistant",
            "content": (
                "I noticed you have several documents in your library:\n"
                "- 📄 2405.15793v3.pdf *(SWE-bench research paper)*\n"
                "- 📄 Verity-By-Colleen-Hoover.pdf\n\n"
                '**Which document would you like me to examine for "What are the key findings?"?**'
            ),
        },
    ]

    scope = await analyze_document_scope(
        question="The SWE-bench paper",
        chat_history=chat_history,
        available_documents=SAMPLE_DOCS,
    )
    assert scope.status == "resolved"
    assert scope.target_doc_id == "doc-1"
    assert scope.target_filename == "2405.15793v3.pdf"
    assert "key findings" in scope.cleaned_question.lower()


@pytest.mark.anyio
async def test_scope_ambiguous_query_triggers_clarification():
    """An ambiguous query in a multi-doc library triggers clarification."""
    scope = await analyze_document_scope(
        question="What are the key findings?",
        chat_history=[],
        available_documents=SAMPLE_DOCS,
    )
    assert scope.status == "needs_clarification"
    assert scope.clarification_text is not None
    assert "2405.15793v3.pdf" in scope.clarification_text
    assert "Verity-By-Colleen-Hoover.pdf" in scope.clarification_text
    assert "Which document would you like me to examine" in scope.clarification_text
