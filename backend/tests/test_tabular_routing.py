import uuid
import pytest
from unittest.mock import patch
from app.services.query import (
    analyze_document_scope,
    get_schema_for_prompt,
    generate_tabular_overview_stream,
    _extract_sql,
)

SAMPLE_DOCS = [
    {"id": "doc-1", "filename": "E Commerce Dataset.xlsx", "file_type": "xlsx", "is_tabular": True},
    {"id": "doc-2", "filename": "Can We Be Strangers Again.pdf", "file_type": "pdf", "is_tabular": False},
    {"id": "doc-3", "filename": "WA_Fn-UseC_-Telco-Customer-Churn.csv", "file_type": "csv", "is_tabular": True},
]


@pytest.mark.anyio
async def test_scope_resolves_tabular_document_from_clarification_response():
    """Replying to clarification with tabular spreadsheet filename resolves is_tabular=True."""
    chat_history = [
        {"role": "user", "content": "What are the key findings?"},
        {
            "role": "assistant",
            "content": (
                "I noticed you have several documents in your library:\n"
                "- E Commerce Dataset.xlsx\n"
                "- Can We Be Strangers Again.pdf\n"
                "Which document would you like me to examine?"
            ),
        },
    ]

    scope = await analyze_document_scope(
        question="E Commerce Dataset.xlsx",
        chat_history=chat_history,
        available_documents=SAMPLE_DOCS,
    )
    assert scope.status == "resolved"
    assert scope.target_filename == "E Commerce Dataset.xlsx"
    assert scope.is_tabular is True
    assert scope.target_doc_id == "doc-1"
    assert "key findings" in scope.cleaned_question.lower()


@pytest.mark.anyio
async def test_get_schema_for_prompt_accepts_optional_doc_id():
    """get_schema_for_prompt delegates doc_id to tabular_store."""
    test_uid = "user-123"
    test_doc = uuid.uuid4()
    with patch("app.services.tabular_store.get_table_schema") as mock_get:
        mock_get.return_value = "Table: t_test_123 (10 rows)"
        schema = await get_schema_for_prompt(test_uid, test_doc)
        mock_get.assert_called_once_with(test_uid, test_doc)
        assert "Table: t_test_123" in schema


def test_extract_sql_helper():
    """_extract_sql parses SELECT statements properly and excludes CANNOT_ANSWER."""
    assert _extract_sql("CANNOT_ANSWER") is None
    assert _extract_sql("```sql\nSELECT count(*) FROM table1;\n```") == "SELECT count(*) FROM table1"
    assert _extract_sql("SELECT id, name FROM users WHERE active = 1;") == "SELECT id, name FROM users WHERE active = 1"


@pytest.mark.anyio
async def test_generate_tabular_overview_stream():
    """generate_tabular_overview_stream yields streamed tokens without error."""
    async def mock_stream(*args, **kwargs):
        for token in ["This ", "dataset ", "contains ", "customer ", "metrics."]:
            yield token

    with patch("app.services.query.stream_llm_with_fallback", side_effect=mock_stream):
        tokens = []
        async for tok in generate_tabular_overview_stream(
            question="What are the key findings?",
            schema_info="Table: t_customers (100 rows)\nColumns: id, churn, tenure",
            filename="E Commerce Dataset.xlsx",
        ):
            tokens.append(tok)
        assert "".join(tokens) == "This dataset contains customer metrics."
