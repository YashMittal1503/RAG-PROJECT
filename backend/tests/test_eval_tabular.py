"""Offline tests for spreadsheet / Text-to-SQL support in the eval harness."""

import re
import sys
import types
from pathlib import Path

import pytest

import app.services as app_services
from eval.corpus import _index_tabular, eval_user_id, load_manifest, read_pages
from eval.dataset import load_dataset, validate_against_corpus
from eval.schema import GoldItem, SQLResult
from eval.scoring import score_item_offline
from eval.scoring.sql import rows_match, score_sql
from eval.systems.vanilla import VanillaRAG

DATA = Path(__file__).resolve().parents[1] / "eval" / "data"


@pytest.fixture
def docs():
    return load_manifest(DATA / "smoke_corpus.yaml")


# ── SQL result matching ─────────────────────────────────────────────────

def test_rows_match_ignores_row_and_column_order_and_float_noise():
    gold = [["cargo", 1150], ["ferry", 0], ["fishing", 55.5]]
    assert rows_match([["fishing", 55.50000001], ["cargo", 1150.0], ["ferry", 0]], gold)
    assert rows_match([[1150, "Cargo "], [0, "ferry"], [55.5, "fishing"]], gold)       # column order, case, spaces
    assert rows_match([["cargo", "1,150"], ["ferry", "0"], ["fishing", "55.5"]], gold)  # numeric strings
    assert not rows_match([["cargo", 1150], ["ferry", 0]], gold)                         # missing row
    assert not rows_match([["cargo", 1151], ["ferry", 0], ["fishing", 55.5]], gold)      # wrong value


def test_rows_match_allows_extra_columns_but_not_missing_ones():
    assert rows_match([["Iron Duke", 512.0]], [["Iron Duke"]])
    assert not rows_match([["Iron Duke"]], [["Iron Duke", 512]])
    assert not rows_match([["Morning Star", 340.0]], [["Iron Duke"]])


def test_rows_match_respects_order_when_required():
    gold = [["Iron Duke"], ["Morning Star"]]
    assert rows_match([["Morning Star"], ["Iron Duke"]], gold, ordered=False)
    assert not rows_match([["Morning Star"], ["Iron Duke"]], gold, ordered=True)
    assert rows_match([["Iron Duke"], ["Morning Star"]], gold, ordered=True)


def test_score_sql_applicability_and_truncation():
    assert score_sql("rag", None, None, None) == {"sql_executed": None, "sql_result_match": None}
    assert score_sql("sql", None, None, [[10]]) == {"sql_executed": 0.0, "sql_result_match": 0.0}
    ok = SQLResult(columns=["n"], rows=[[10]])
    assert score_sql("sql", "SELECT 10", ok, [[10]]) == {"sql_executed": 1.0, "sql_result_match": 1.0}
    assert score_sql("sql", "SELECT 10", ok, None) == {"sql_executed": 1.0, "sql_result_match": None}
    truncated = SQLResult(columns=["n"], rows=[[10]], truncated=True)
    assert score_sql("sql", "SELECT 10", truncated, [[10]])["sql_result_match"] == 0.0


def test_sql_item_needs_gold_rows_or_answer():
    with pytest.raises(ValueError):
        GoldItem(id="s", question="how many?", expected_route="sql", category="tabular_sql")
    GoldItem(id="s", question="how many?", expected_route="sql", category="tabular_sql", gold_sql_rows=[[3]])


# ── corpus: spreadsheets ────────────────────────────────────────────────

def test_manifest_loads_spreadsheet_and_renders_it_for_validation(docs):
    arrivals = next(d for d in docs if d.id == "arrivals")
    assert arrivals.is_tabular and not next(d for d in docs if d.id == "lighthouse").is_tabular
    (page, text), = read_pages(arrivals)
    assert page == 1 and "Iron Duke" in text and "vessel_type" in text


def test_validate_flags_sql_item_pointing_at_a_text_document(docs):
    item = GoldItem(id="bad", question="how many?", docs=["rules"], expected_doc="rules",
                    expected_route="sql", category="tabular_sql", gold_sql_rows=[[1]])
    problems = validate_against_corpus([item], {"rules": "text", "arrivals": "csv"}, tabular_docs={"arrivals"})
    assert problems and "not a spreadsheet" in problems[0]


def test_tabular_indexing_and_gold_rows_against_real_duckdb(docs, tmp_path, monkeypatch):
    """Store the smoke CSV the way the app does, then check every smoke SQL item's gold rows
    against what its reference query actually returns. Guards the gold data itself."""
    from app.services import tabular_store

    monkeypatch.setattr(tabular_store, "_DATA_DIR", tmp_path)
    user = "00000000-0000-4000-8000-00000000test"
    arrivals = next(d for d in docs if d.id == "arrivals")

    assert _index_tabular(arrivals, user, tabular_store) == 10
    assert _index_tabular(arrivals, user, tabular_store) == 10   # idempotent re-index
    (table,) = tabular_store.list_tables(user)

    sql_items = [i for i in load_dataset(DATA / "smoke.jsonl") if i.expected_route == "sql"]
    assert len(sql_items) >= 3
    for item in sql_items:
        sql = re.sub(r"\barrivals\b", table, item.gold_sql)
        result = tabular_store.execute_sql(user, sql)
        assert rows_match(result["rows"], item.gold_sql_rows, item.ordered), item.id


# ── vanilla adapter: the real tabular_qa pipeline over a real DuckDB ────

class _Scope:
    def __init__(self, doc):
        self.status = "resolved"
        self.target_doc_id = doc.uuid
        self.target_filename = doc.filename
        self.clarification_text = None
        self.cleaned_question = ""
        self.is_tabular = doc.is_tabular


@pytest.fixture
def eval_store(docs, tmp_path, monkeypatch):
    """The smoke CSV indexed into a temp DuckDB for the eval tenant, as `python -m eval index` does."""
    from app.services import tabular_store

    monkeypatch.setattr(tabular_store, "_DATA_DIR", tmp_path)
    arrivals = next(d for d in docs if d.id == "arrivals")
    _index_tabular(arrivals, eval_user_id(), tabular_store)
    return arrivals, tabular_store.list_tables(eval_user_id())[0]


def _install_fakes(monkeypatch, doc, *, qc, sql_replies=()):
    """Fake query-module steps (classify / scope / rewrite) and the LLM calls inside tabular_qa."""
    from app.services import tabular_qa

    calls = {"sql": []}
    replies = list(sql_replies)
    q = types.ModuleType("app.services.query")

    async def classify_query(question, history):
        return qc

    async def analyze_document_scope(question, history, library):
        calls["library"] = library
        return _Scope(doc)

    async def rewrite_query(question, history):
        return question

    async def retrieve_chunks(**kw):
        calls["rag"] = True
        return []

    q.classify_query = classify_query
    q.analyze_document_scope = analyze_document_scope
    q.rewrite_query = rewrite_query
    q.retrieve_chunks = retrieve_chunks
    monkeypatch.setitem(sys.modules, "app.services.query", q)
    monkeypatch.setattr(app_services, "query", q, raising=False)

    async def generate_sql_query(question, schema_info, chat_history=None, **kw):
        calls["sql"].append(kw)
        return replies.pop(0) if replies else None

    async def fake_stream(messages, max_tokens, temperature=0.1, fast=False):
        for tok in ["There ", "are ", "10 ", "arrivals."]:
            yield tok

    monkeypatch.setattr(tabular_qa, "generate_sql_query", generate_sql_query)
    monkeypatch.setattr(tabular_qa, "stream_llm_with_fallback", fake_stream)
    return calls


SQL_ITEM = GoldItem(id="s", question="How many arrivals?", docs=["arrivals"], answer="10", gold_sql_rows=[[10]],
                    expected_route="sql", expected_doc="arrivals", category="tabular_sql",
                    expected_scope="targeted", expected_aggregation=True)


@pytest.mark.anyio
async def test_sql_route_records_query_result_context_and_classification(monkeypatch, docs, eval_store):
    from app.services.query import QueryClassification

    arrivals, table = eval_store
    calls = _install_fakes(monkeypatch, arrivals, qc=QueryClassification("retrieve", "targeted", True),
                           sql_replies=[f"SELECT count(*) AS n FROM {table}"])
    pred = await VanillaRAG(docs).answer(SQL_ITEM)

    assert pred.error is None and pred.route == "sql" and pred.chosen_doc == "arrivals"
    assert pred.sql_result.rows == [[10]] and "Result:" in pred.contexts[0].text
    assert pred.answer.strip() == "There are 10 arrivals."            # scalar result: no table appended
    assert calls["sql"][0]["aggregation"] is True and "rag" not in calls
    assert next(d for d in calls["library"] if d["filename"] == arrivals.filename)["is_tabular"] is True
    assert {"schema", "sql_generate", "sql_execute", "generate"} <= set(pred.timings_ms)

    s = score_item_offline(SQL_ITEM, pred)
    assert s.sql_executed == 1.0 and s.sql_result_match == 1.0 and s.route_correct == 1.0
    assert s.scope_correct == 1.0 and s.aggregation_correct == 1.0


@pytest.mark.anyio
async def test_failing_sql_is_repaired_then_falls_back_to_overview(monkeypatch, docs, eval_store):
    from app.services.query import QueryClassification

    arrivals, table = eval_store
    qc = QueryClassification("retrieve", "targeted", True)
    _install_fakes(monkeypatch, arrivals, qc=qc, sql_replies=[f"SELECT nope FROM {table}", f"SELECT count(*) FROM {table}"])
    pred = await VanillaRAG(docs).answer(SQL_ITEM)
    assert pred.route == "sql" and pred.sql_result.rows == [[10]]      # repaired on the 2nd attempt

    _install_fakes(monkeypatch, arrivals, qc=qc, sql_replies=[f"SELECT nope FROM {table}", f"SELECT nope2 FROM {table}"])
    pred = await VanillaRAG(docs).answer(SQL_ITEM)
    assert pred.route == "tabular_overview" and "nope2" in pred.sql_error and pred.sql_result is None
    assert pred.contexts[0].text.startswith("### Table")                # answered from the data profile
    assert score_item_offline(SQL_ITEM, pred).sql_result_match == 0.0


@pytest.mark.anyio
async def test_overview_question_skips_sql(monkeypatch, docs, eval_store):
    from app.services.query import QueryClassification

    arrivals, _ = eval_store
    calls = _install_fakes(monkeypatch, arrivals, qc=QueryClassification("retrieve", "global", False))
    item = GoldItem(id="o", question="What does the log contain?", docs=["arrivals"], answer="A log.",
                    expected_route="tabular_overview", expected_doc="arrivals", category="global_summary")
    pred = await VanillaRAG(docs).answer(item)
    assert pred.route == "tabular_overview" and pred.sql is None and calls["sql"] == []
    assert score_item_offline(item, pred).route_correct == 1.0


@pytest.mark.anyio
async def test_no_tabular_data_falls_through_to_rag(monkeypatch, docs, tmp_path):
    from app.services import tabular_store
    from app.services.query import QueryClassification

    monkeypatch.setattr(tabular_store, "_DATA_DIR", tmp_path)          # empty: nothing indexed
    arrivals = next(d for d in docs if d.id == "arrivals")
    calls = _install_fakes(monkeypatch, arrivals, qc=QueryClassification())
    pred = await VanillaRAG(docs).answer(SQL_ITEM)
    assert pred.route == "rag" and calls.get("rag") is True
