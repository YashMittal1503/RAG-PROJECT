import uuid
import pytest
from unittest.mock import patch
from app.services.query import analyze_document_scope
from app.services.tabular_qa import (
    extract_sql,
    generate_tabular_overview_stream,
    get_schema_for_prompt,
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


def testextract_sql_helper():
    """extract_sql parses SELECT statements properly and excludes CANNOT_ANSWER."""
    assert extract_sql("CANNOT_ANSWER") is None
    assert extract_sql("```sql\nSELECT count(*) FROM table1;\n```") == "SELECT count(*) FROM table1"
    assert extract_sql("SELECT id, name FROM users WHERE active = 1;") == "SELECT id, name FROM users WHERE active = 1"


@pytest.mark.anyio
async def test_generate_tabular_overview_stream():
    """generate_tabular_overview_stream yields streamed tokens without error."""
    async def mock_stream(*args, **kwargs):
        for token in ["This ", "dataset ", "contains ", "customer ", "metrics."]:
            yield token

    with patch("app.services.tabular_qa.stream_llm_with_fallback", side_effect=mock_stream):
        tokens = []
        async for tok in generate_tabular_overview_stream(
            question="What are the key findings?",
            schema_info="Table: t_customers (100 rows)\nColumns: id, churn, tenure",
            filename="E Commerce Dataset.xlsx",
        ):
            tokens.append(tok)
        assert "".join(tokens) == "This dataset contains customer metrics."


# ── Spreadsheet profiles for the document router ──────────────────────────

def test_tabular_profile_lists_every_sheet_and_prefers_descriptive_columns():
    from app.services.tabular_qa import tabular_profile

    tables = [(f"t_b06ad4a4_sheet{i}", ["Row_ID", f"Name_{i}", f"Amount_{i}", "Ref_Code"]) for i in range(40)]
    tables.append(("t_b06ad4a4_customers", ["Customer_ID", "Customer_Name", "State"]))
    text = tabular_profile(tables, max_chars=2400)
    assert len(text) <= 2400
    assert "customers (Customer_Name" in text            # the 41st sheet still shows up
    assert text.split(";")[0].startswith("sheet0 (Name_0")  # descriptive columns before ID/code columns
    assert tabular_profile([("t_0254388c_sheet1", ["Title", "Director"])]) == "sheet1 (Title, Director)"


@pytest.mark.anyio
async def test_attach_tabular_profiles_matches_tables_to_documents_by_uuid_prefix():
    from app.services import tabular_qa

    doc_id = "0254388c-1111-4222-8333-444455556666"
    docs = [{"id": doc_id, "filename": "movies.csv", "is_tabular": True},
            {"id": "aaaaaaaa-0000-0000-0000-000000000000", "filename": "book.pdf", "is_tabular": False}]
    cols = {"0254388c": [("t_0254388c_sheet1", ["Series_Title", "Director", "Metascore"])]}
    with patch.object(tabular_qa.tabular_store, "table_columns_by_doc", return_value=cols):
        await tabular_qa.attach_tabular_profiles("u", docs)
    assert docs[0]["tabular_profile"] == "sheet1 (Series_Title, Director, Metascore)"
    assert "tabular_profile" not in docs[1]


@pytest.mark.anyio
async def test_router_sees_spreadsheet_columns_in_its_catalog():
    """A question that names no file ('How many movies have no Metascore?') must be routable by columns."""
    from unittest.mock import AsyncMock, MagicMock

    docs = [{"id": "d1", "filename": "Novel.pdf", "is_tabular": False},
            {"id": "d2", "filename": "top1000.csv", "is_tabular": True,
             "tabular_profile": "sheet1 (Series_Title, Director, Metascore)"}]
    reply = MagicMock(choices=[MagicMock(message=MagicMock(content="CATEGORY: SPECIFIC\nDOCUMENT_INDEX: 2", reasoning=None))])
    llm = AsyncMock(return_value=reply)
    with patch("app.services.query.call_llm_with_fallback", llm):
        scope = await analyze_document_scope("How many movies have no Metascore?", None, docs)
    system_prompt = llm.call_args.kwargs["messages"][0]["content"]
    assert "top1000.csv (Tabular) — sheets/columns: sheet1 (Series_Title, Director, Metascore)" in system_prompt
    assert scope.status == "resolved" and scope.is_tabular and scope.target_doc_id == "d2"


def test_table_columns_by_doc_reads_real_duckdb(tmp_path, monkeypatch):
    import pandas as pd

    from app.services import tabular_store as ts

    monkeypatch.setattr(ts, "DATA_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(ts, "_db_path", lambda uid: str(tmp_path / f"{uid}.duckdb"))
    doc = uuid.UUID("0254388c-1111-4222-8333-444455556666")
    ts.store_dataframe("u1", doc, "Sheet1", pd.DataFrame({"Title": ["A"], "Director": ["B"]}))
    out = ts.table_columns_by_doc("u1")
    assert list(out) == ["0254388c"]
    table, cols = out["0254388c"][0]
    assert table.startswith("t_0254388c_") and cols[:2] == ["Title", "Director"]
