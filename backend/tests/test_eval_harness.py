"""Offline tests for the eval harness: schema, dataset loading, judge parsing/caching, reports, calibration."""

import json
from pathlib import Path

import pytest

from eval.calibrate import cohens_kappa
from eval.dataset import filter_items, load_dataset, validate_against_corpus
from eval.report import bootstrap_ci, compare, render_comparison, render_markdown, summarize
from eval.schema import Evidence, GoldItem, ItemScore
from eval.scoring.judge import Judge, build_user_prompt, parse_verdict

DATA = Path(__file__).resolve().parents[1] / "eval" / "data"


# ── schema / dataset ────────────────────────────────────────────────────

def test_answerable_rag_item_requires_answer_and_evidence():
    with pytest.raises(ValueError):
        GoldItem(id="x", question="q?", answer="", evidence=[Evidence(doc="d", quote="q")])
    with pytest.raises(ValueError):
        GoldItem(id="x", question="q?", answer="a")
    # Unanswerable and chitchat items need neither.
    GoldItem(id="u", question="q?", answerable=False, category="unanswerable")
    GoldItem(id="c", question="hi", expected_route="chitchat")


def test_smoke_dataset_loads_and_filters():
    items = load_dataset(DATA / "smoke.jsonl")
    assert len(items) >= 10
    assert {i.category for i in items} >= {"implicit_inference", "unanswerable", "multi_turn", "multi_hop"}
    assert all(i.split == "holdout" for i in filter_items(items, split="holdout"))
    assert len(filter_items(items, categories=["unanswerable"])) == 2
    assert len(filter_items(items, limit=3)) == 3


def test_load_dataset_reports_bad_lines_and_duplicates(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"id":"a","question":"q","answerable":false,"category":"unanswerable"}\nnot json\n')
    with pytest.raises(ValueError, match="line 2"):
        load_dataset(p)
    p.write_text('{"id":"a","question":"q","answerable":false,"category":"unanswerable"}\n' * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(p)


def test_validate_against_corpus():
    item = GoldItem(id="a", question="q", docs=["d"], answer="x",
                    evidence=[Evidence(doc="d", quote="sack of flour"), Evidence(doc="d", quote="never written")])
    problems = validate_against_corpus([item], {"d": "He tossed a sack of flour onto the jetty."})
    assert len(problems) == 1 and "never written" in problems[0]
    assert "unknown doc" in validate_against_corpus([item], {})[0]


# ── judge ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,label", [
    ('{"label": "correct", "reason": "matches"}', "correct"),
    ('```json\n{"label":"partial","reason":"half"}\n```', "partial"),
    ("Verdict: WRONG because it names Pellow", "wrong"),
])
def test_parse_verdict(raw, label):
    assert parse_verdict(raw)[0] == label


def test_parse_verdict_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("I am not sure")


def test_user_prompt_marks_unanswerable():
    item = GoldItem(id="u", question="Who was his wife?", answerable=False, category="unanswerable")
    prompt = build_user_prompt(item, "His wife was Ada.")
    assert "UNANSWERABLE" in prompt and "ANSWERABLE: no" in prompt


class _FakeCompletions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply

        class _Msg:
            content = reply

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


class _FakeClient:
    def __init__(self, replies):
        self.chat = type("Chat", (), {})()
        self.chat.completions = _FakeCompletions(replies)


@pytest.mark.anyio
async def test_judge_caches_and_retries(tmp_path, monkeypatch):
    import eval.scoring.judge as judge_mod

    async def _no_sleep(_):
        return None

    monkeypatch.setattr(judge_mod.asyncio, "sleep", _no_sleep)
    client = _FakeClient([RuntimeError("429"), '{"label":"correct","reason":"ok"}'])
    judge = Judge(model="fake-judge", cache_dir=tmp_path, client=client)
    item = GoldItem(id="a", question="q", answer="Elias", evidence=[Evidence(doc="d", quote="Elias")])

    v1 = await judge.grade(item, "It is Elias.")
    assert v1.label == "correct" and not v1.cached
    assert client.chat.completions.calls == 2  # one failure, one success

    v2 = await judge.grade(item, "It is Elias.")
    assert v2.cached and client.chat.completions.calls == 2  # served from cache


# ── report / compare / calibration ──────────────────────────────────────

def _score(i, cat="single_fact", judge=1.0, r10=1.0, err=None):
    return ItemScore(item_id=f"i{i}", category=cat, answerable=True, difficulty="medium",
                     judge_score=judge, evidence_recall_at_k={10: r10}, false_refusal=0.0,
                     total_ms=100.0 + i, error=err)


def test_bootstrap_ci_bounds():
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert 0.35 < lo < 0.5 < hi < 0.65
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_summarize_and_render():
    scores = [_score(i) for i in range(5)] + [_score(9, cat="multi_hop", judge=0.0, r10=0.0)]
    summary = summarize(scores)
    assert summary["overall"]["n_items"] == 6
    assert summary["overall"]["correctness (judge)"]["mean"] == pytest.approx(5 / 6)
    assert summary["by_category"]["multi_hop"]["evidence recall@10"]["mean"] == 0.0
    md = render_markdown(summary, {"system": "vanilla", "run_id": "r"})
    assert "| multi_hop | 1 |" in md and "correctness (judge)" in md
    json.dumps(summary)  # must be JSON-serializable for summary.json


def test_compare_detects_improvement_and_no_change():
    a = [_score(i, judge=0.0) for i in range(40)]
    b = [_score(i, judge=1.0) for i in range(40)]
    cmp = compare(a, b)
    m = cmp["metrics"]["correctness (judge)"]
    assert cmp["n_common_items"] == 40
    assert m["significant"] and m["b_better"] and m["delta"] == pytest.approx(1.0)
    assert not cmp["metrics"]["evidence recall@10"]["significant"]
    assert "better" in render_comparison(cmp, "A", "B")


def test_compare_lower_is_better_metric():
    a = [_score(i, err="boom") for i in range(30)]
    b = [_score(i) for i in range(30)]
    m = compare(a, b)["metrics"]["error rate"]
    assert m["significant"] and m["b_better"]


def test_cohens_kappa():
    assert cohens_kappa(["correct", "wrong"] * 5, ["correct", "wrong"] * 5) == pytest.approx(1.0)
    assert cohens_kappa(["correct", "wrong", "correct", "wrong"], ["wrong", "correct", "wrong", "correct"]) < 0
