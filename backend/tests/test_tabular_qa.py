"""
Tests for the Text-to-SQL pipeline: sandboxed execution, Markdown formatting, schema
profile, and the plan (overview skip, SQL repair, fallback) with the LLM mocked.
Uses real DuckDB files in a temp directory.
"""

import uuid
from pathlib import Path

import pandas as pd
import pytest

from app.services import tabular_qa as tq
from app.services import tabular_store as ts
from app.services.query import QueryClassification

CSV = Path(__file__).resolve().parents[1] / "eval" / "data" / "smoke_corpus" / "vessel_arrivals.csv"
USER = "00000000-0000-4000-8000-000000000a11"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A user DuckDB holding the smoke vessel-arrivals sheet; returns (doc_id, table)."""
    monkeypatch.setattr(ts, "_DATA_DIR", tmp_path)
    doc = uuid.uuid4()
    df = pd.read_csv(CSV)
    df["Monthly Charges"] = 1.5
    df["2024 sales"] = range(len(df))
    ts.store_dataframe(USER, doc, "Sheet1", df)
    (table,) = ts.list_tables(USER)
    return doc, table


# ── sandbox / validation ───────────────────────────────────────────────

def test_file_and_network_access_is_blocked(store):
    for sql in ("SELECT content FROM read_text('.env')",
                "SELECT * FROM read_csv('requirements.txt')",
                "SELECT * FROM read_parquet('x.parquet')"):
        with pytest.raises(ValueError):
            ts.execute_sql(USER, sql)


def test_only_single_select_statements(store):
    _, t = store
    for sql in (f"SELECT 1; DROP TABLE {t}", f"DELETE FROM {t}", f"ATTACH 'other.duckdb'",
                "PRAGMA database_list", f"UPDATE {t} SET crew = 0"):
        with pytest.raises(ValueError):
            ts.execute_sql(USER, sql)


def test_ctes_functions_and_keywords_inside_strings_are_allowed(store):
    _, t = store
    assert ts.execute_sql(USER, f"WITH x AS (SELECT * FROM {t}) SELECT count(*) FROM x")["rows"] == [[10]]
    r = ts.execute_sql(USER, f"SELECT replace(vessel, 'a', 'A') AS v FROM {t} WHERE berth <> 'delete me' ORDER BY 1 LIMIT 1")
    assert r["rows"] == [["GAnnet"]]
    # CAST -> TRY_CAST outside literals: dirty text becomes NULL instead of an error
    assert ts.execute_sql(USER, f"SELECT CAST(vessel AS DOUBLE) AS n FROM {t} LIMIT 1")["rows"] == [[None]]


def test_timeout_stops_runaway_queries(store):
    with pytest.raises(ValueError, match="took longer"):
        ts.execute_sql(USER, "SELECT count(*) FROM range(1000000000000) a, range(1000) b", timeout_s=1)


def test_row_cap_and_truncation_flag(store):
    r = ts.execute_sql(USER, "SELECT * FROM range(250)")
    assert r["row_count"] == ts.MAX_RESULT_ROWS and r["truncated"] is True


# ── storage, schema profile, formatting ────────────────────────────────

def test_columns_are_sanitized_and_dates_typed(store):
    doc, t = store
    cols = ts.execute_sql(USER, f"SELECT * FROM {t} LIMIT 1")["columns"]
    assert "Monthly_Charges" in cols and "col_2024_sales" in cols
    assert ts.execute_sql(USER, f"SELECT count(*) FROM {t} WHERE date >= DATE '2025-03-08'")["rows"] == [[4]]


def test_schema_is_markdown_with_categories_ranges_and_headers(store):
    doc, t = store
    schema = ts.get_table_schema(USER, doc)
    assert f"### Table `{t}` (10 rows)" in schema
    assert "| column | type | values | nulls |" in schema and "|---|---|---|---|" in schema
    assert "one of: 'cargo', 'ferry', 'fishing'" in schema              # exact filter spellings
    assert "| `date` | DATE | 2025-03-01 to 2025-03-12 |" in schema
    assert "(header: Monthly Charges)" in schema and "(header: 2024 sales)" in schema
    assert "Sample rows (first 3):" in schema
    assert "| --- | --- | --- | ---: |" in schema                       # sample table has a header separator


def test_markdown_table_escapes_and_formats():
    md = ts.markdown_table(["name", "value", "note"], [["a", 1234567.891234, "x|y"], ["b", 2.0, None], ["c", 3, "two\nlines"]])
    lines = md.splitlines()
    assert lines[0] == "| name | value | note |"
    assert lines[1] == "| --- | ---: | --- |"                       # numeric column right-aligned
    assert lines[2] == "| a | 1,234,567.8912 | x\\|y |"
    assert lines[3] == "| b | 2 | — |"
    assert lines[4] == "| c | 3 | two lines |"


def test_format_sql_result_and_user_table():
    multi = {"columns": ["type", "total"], "rows": [["cargo", 1150.0], ["fishing", 55.5]], "truncated": False}
    text = tq.format_sql_result(multi)
    assert text.startswith("| type | total |\n| --- | ---: |") and "(2 rows)" in text
    assert tq.result_table_for_user(multi).startswith("| type | total |")
    scalar = {"columns": ["n"], "rows": [[10]], "truncated": False}
    assert tq.result_table_for_user(scalar) is None                    # a single value needs no table
    assert tq.format_sql_result({"columns": ["n"], "rows": []}) == "The query returned no rows."
    trunc = {"columns": ["n"], "rows": [[i] for i in range(100)], "truncated": True}
    assert "truncated" in tq.format_sql_result(trunc)
    assert "Showing the first 50 rows" in tq.result_table_for_user(trunc)


def test_extract_sql_variants():
    assert tq.extract_sql("```sql\nWITH x AS (SELECT 1) SELECT * FROM x;\n```") == "WITH x AS (SELECT 1) SELECT * FROM x"
    assert tq.extract_sql("Here you go:\nSELECT a FROM t; SELECT b FROM t;") == "SELECT a FROM t"
    assert tq.extract_sql("CANNOT_ANSWER") is None
    assert tq.extract_sql("I don't know") is None


# ── plan: overview skip, repair, fallback ───────────────────────────────

def _fake_llm(monkeypatch, sql_replies):
    """Mock SQL generation (returns scripted SQL) and the answer streams; records calls."""
    calls = {"generate": [], "stream": []}
    replies = list(sql_replies)

    async def generate_sql_query(question, schema_info, chat_history=None, *, aggregation=None,
                                 failed_sql=None, error=None):
        calls["generate"].append({"aggregation": aggregation, "failed_sql": failed_sql, "error": error})
        return replies.pop(0) if replies else None

    async def fake_stream(messages, max_tokens, temperature=0.1, fast=False):
        calls["stream"].append(messages)
        for tok in ["There ", "are ", "results."]:
            yield tok

    monkeypatch.setattr(tq, "generate_sql_query", generate_sql_query)
    monkeypatch.setattr(tq, "stream_llm_with_fallback", fake_stream)
    return calls


@pytest.mark.anyio
async def test_overview_question_skips_sql(store, monkeypatch):
    doc, _ = store
    calls = _fake_llm(monkeypatch, [])
    plan = await tq.plan_tabular_answer(USER, "What does my data contain?", doc,
                                        QueryClassification("retrieve", "global", False))
    assert plan.route == "tabular_overview" and plan.sql is None and calls["generate"] == []
    answer = "".join([t async for t in tq.stream_tabular_answer(plan, "What does my data contain?", "arrivals.csv")])
    assert answer == "There are results."
    assert "Data profile" in calls["stream"][0][1]["content"]


@pytest.mark.anyio
async def test_sql_route_passes_aggregation_hint_and_appends_exact_table(store, monkeypatch):
    doc, t = store
    calls = _fake_llm(monkeypatch, [f"SELECT vessel_type, sum(cargo_tonnes) AS total FROM {t} GROUP BY 1 ORDER BY 1"])
    plan = await tq.plan_tabular_answer(USER, "Total cargo per type?", doc, QueryClassification("retrieve", "targeted", True))
    assert plan.route == "sql" and plan.attempts == 1 and calls["generate"][0]["aggregation"] is True
    answer = "".join([t async for t in tq.stream_tabular_answer(plan, "Total cargo per type?")])
    assert answer.startswith("There are results.")
    assert "| vessel_type | total |" in answer and "| cargo | 1,150 |" in answer
    # The answering LLM saw the result as a Markdown table and was told not to reproduce it
    prompt = calls["stream"][0]
    assert "| vessel_type | total |" in prompt[1]["content"] and "Do NOT reproduce the table" in prompt[0]["content"]


@pytest.mark.anyio
async def test_failed_sql_is_repaired_once_with_the_error(store, monkeypatch):
    doc, t = store
    calls = _fake_llm(monkeypatch, [f"SELECT no_such_column FROM {t}", f"SELECT count(*) AS n FROM {t}"])
    plan = await tq.plan_tabular_answer(USER, "How many arrivals?", doc, QueryClassification("retrieve", "targeted", True))
    assert plan.route == "sql" and plan.attempts == 2 and plan.result["rows"] == [[10]] and plan.sql_error is None
    second = calls["generate"][1]
    assert second["failed_sql"].endswith("no_such_column FROM " + t) and "no_such_column" in second["error"]


@pytest.mark.anyio
async def test_sql_that_keeps_failing_falls_back_to_an_honest_overview(store, monkeypatch):
    doc, t = store
    calls = _fake_llm(monkeypatch, [f"SELECT bad1 FROM {t}", f"SELECT bad2 FROM {t}"])
    plan = await tq.plan_tabular_answer(USER, "How many arrivals?", doc, QueryClassification())
    assert plan.route == "tabular_overview" and plan.attempts == 2 and "bad2" in plan.sql_error
    "".join([t async for t in tq.stream_tabular_answer(plan, "How many arrivals?")])
    assert "couldn't calculate" in calls["stream"][0][1]["content"]


@pytest.mark.anyio
async def test_no_schema_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_DATA_DIR", tmp_path)
    assert await tq.plan_tabular_answer(USER, "q", None, QueryClassification()) is None
