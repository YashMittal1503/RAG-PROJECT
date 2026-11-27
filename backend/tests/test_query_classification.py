"""
Tests for the LLM query classifier (intent / scope / aggregation) that replaced the
keyword-based is_overview_query and detect_aggregation, and for the eval's
classification scoring (`python -m eval classify` and per-item scope accuracy).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

import app.services.query as q
from app.services.query import QueryClassification, classify_query, parse_query_classification
from eval.classify import load_cases, score_cases
from eval.schema import GoldItem, Prediction
from eval.scoring import score_item_offline

DATA = Path(__file__).resolve().parents[1] / "eval" / "data"


def _llm_reply(content="", reasoning=""):
    msg = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ── parsing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ('{"intent":"retrieve","scope":"global","aggregation":false}', ("retrieve", "global", False)),
    ('```json\n{"intent": "retrieve", "scope": "targeted", "aggregation": true}\n```', ("retrieve", "targeted", True)),
    ('Sure! {"intent":"chitchat","scope":"targeted","aggregation":"false"}', ("chitchat", "targeted", False)),
    ('{"intent":"weird","scope":"everything","aggregation":"true"}', ("retrieve", "targeted", True)),  # unknown values -> safe defaults
])
def test_parse_query_classification(raw, expected):
    qc = parse_query_classification(raw)
    assert (qc.intent, qc.scope, qc.aggregation) == expected


def test_parse_falls_back_to_reasoning_and_rejects_garbage():
    qc = parse_query_classification("", reasoning='thinking... {"intent":"retrieve","scope":"metadata","aggregation":false}')
    assert qc.scope == "metadata"
    assert parse_query_classification("no json here") is None


def test_is_overview_only_for_global_scope():
    assert QueryClassification(scope="global").is_overview
    for scope in ("section", "metadata", "targeted"):
        assert not QueryClassification(scope=scope).is_overview


# ── classify_query with the LLM mocked ─────────────────────────────────

@pytest.mark.anyio
async def test_classify_query_uses_llm_and_history(monkeypatch):
    seen = {}

    async def fake_llm(messages, max_tokens, temperature=0.0, fast=False):
        seen["messages"] = messages
        return _llm_reply('{"intent":"retrieve","scope":"targeted","aggregation":true}')

    monkeypatch.setattr(q, "call_llm_with_fallback", fake_llm)
    history = [{"role": "user", "content": "total revenue 2023?"}, {"role": "assistant", "content": "$1.2M"}]
    qc = await classify_query("and for 2024?", history)
    assert (qc.intent, qc.scope, qc.aggregation, qc.method) == ("retrieve", "targeted", True, "llm")
    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "user"]


@pytest.mark.anyio
async def test_greetings_skip_the_llm(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("LLM should not be called for a greeting")

    monkeypatch.setattr(q, "call_llm_with_fallback", boom)
    qc = await classify_query("Hello!")
    assert qc.intent == "chitchat" and qc.method == "greeting"


@pytest.mark.anyio
async def test_llm_failure_defaults_to_plain_retrieval(monkeypatch):
    async def fail(*a, **k):
        raise RuntimeError("all providers down")

    monkeypatch.setattr(q, "call_llm_with_fallback", fail)
    qc = await classify_query("Summarize the book")
    # No keyword fallback: an outage degrades to normal targeted RAG, never a wrong special route.
    assert (qc.intent, qc.scope, qc.aggregation, qc.method) == ("retrieve", "targeted", False, "fallback")


def test_keyword_helpers_are_gone():
    for name in ("is_overview_query", "detect_aggregation", "OVERVIEW_KEYWORDS", "AGGREGATION_KEYWORDS", "classify_intent"):
        assert not hasattr(q, name), name


# ── eval: classification dataset and per-item scoring ──────────────────

def test_classification_dataset_loads_and_covers_every_label():
    cases = load_cases(DATA / "query_classification.jsonl")
    assert len(cases) >= 40
    assert {c.expected_scope for c in cases if c.expected_scope} == {"global", "section", "metadata", "targeted"}
    assert {c.expected_aggregation for c in cases if c.expected_aggregation is not None} == {True, False}
    assert any(c.expected_intent == "chitchat" for c in cases) and any(c.history for c in cases)


def test_score_cases_accuracy_confusion_and_mistakes():
    cases = load_cases(DATA / "query_classification.jsonl")[:3]  # qc-001..003: global, no aggregation
    preds = [
        {"intent": "retrieve", "scope": "global", "aggregation": False},
        {"intent": "retrieve", "scope": "targeted", "aggregation": False},
        {"intent": "retrieve", "scope": "global", "aggregation": True},
    ]
    r = score_cases(cases, preds)
    assert r["accuracy"]["intent"] == 1.0
    assert r["accuracy"]["scope"] == pytest.approx(2 / 3) and r["accuracy"]["aggregation"] == pytest.approx(2 / 3)
    assert r["scope_confusion"] == {"global -> global": 2, "global -> targeted": 1}
    assert [m["id"] for m in r["mistakes"]] == ["qc-002", "qc-003"]


def test_full_run_scores_scope_and_aggregation_per_item():
    item = GoldItem(id="s", question="Summarize it", docs=["lighthouse"], answer="A story.",
                    evidence=[{"doc": "lighthouse", "quote": "Morning, Elias!"}],
                    expected_scope="global", expected_aggregation=False)
    good = Prediction(item_id="s", answer="A story.", query_scope="global", aggregation=False)
    bad = Prediction(item_id="s", answer="A story.", query_scope="targeted", aggregation=True)
    assert (score_item_offline(item, good).scope_correct, score_item_offline(item, good).aggregation_correct) == (1.0, 1.0)
    assert (score_item_offline(item, bad).scope_correct, score_item_offline(item, bad).aggregation_correct) == (0.0, 0.0)
    unlabeled = item.model_copy(update={"expected_scope": None, "expected_aggregation": None})
    assert score_item_offline(unlabeled, bad).scope_correct is None
