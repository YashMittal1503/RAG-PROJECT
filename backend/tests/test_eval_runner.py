"""End-to-end test of the eval runner with a fake system and fake judge (offline)."""

import csv
from pathlib import Path

import pytest

import eval.runner as runner
from eval import calibrate
from eval.report import compare, load_scores
from eval.schema import Citation, GoldItem, Prediction, RetrievedContext
from eval.scoring.judge import Verdict
from eval.systems.base import RAGSystem

DATA = Path(__file__).resolve().parents[1] / "eval" / "data"


class _OracleSystem(RAGSystem):
    """Answers with the gold answer and retrieves the gold evidence, so scores should be near-perfect."""

    name = "oracle"

    def __init__(self, docs, items, **kw):
        super().__init__(docs, **kw)
        self.items = {i.id: i for i in items}
        self.indexed = False

    async def index(self, force=False):
        self.indexed = True
        return {}

    async def answer(self, item: GoldItem) -> Prediction:
        if item.expected_route == "chitchat":
            return Prediction(item_id=item.id, answer="Hello!", route="chitchat")
        if not item.answerable:
            return Prediction(item_id=item.id, answer="This information is not available in the provided document.",
                              route="rag", chosen_doc=item.expected_doc)
        return Prediction(
            item_id=item.id,
            answer=f"{item.answer} [Page 1]",
            contexts=[RetrievedContext(text=e.quote, doc=e.doc, page=e.page) for e in item.evidence],
            citations=[Citation(doc=e.doc, page=e.page) for e in item.evidence],
            route="rag",
            chosen_doc=item.expected_doc,
            timings_ms={"generate": 10.0},
        )


class _BrokenSystem(_OracleSystem):
    name = "broken"

    async def answer(self, item):
        return Prediction(item_id=item.id, answer="I don't know.", route="rag")


class _Judge:
    model = "fake-judge"

    async def grade(self, item, answer):
        ok = (not item.answerable and "not available" in answer) or (item.answer and item.answer in answer)
        return Verdict("correct" if ok else "wrong", 1.0 if ok else 0.0, "fake")

    async def key_point_recall(self, item, answer):
        return {"key_point_recall": 1.0, "missed_key_points": []}


class _Ragas:
    """Stands in for RagasScorer: faithful when the answer is the gold answer."""

    def config(self):
        return {"ragas_version": "fake", "model": "fake-judge", "metrics": ["faithfulness"]}

    async def score(self, item, pred, refused):
        if refused or not pred.contexts:
            return {"ragas": {}, "unsupported_claims": []}
        ok = bool(item.answer) and item.answer in pred.answer
        return {"ragas": {"faithfulness": 1.0 if ok else 0.0}, "unsupported_claims": [] if ok else ["x"]}


@pytest.mark.anyio
async def test_full_run_rescore_compare_calibrate(tmp_path, monkeypatch):
    from eval.dataset import load_dataset

    items = load_dataset(DATA / "smoke.jsonl")

    def _fake_get_system(name, docs, **kw):
        return {"oracle": _OracleSystem, "broken": _BrokenSystem}[name](docs, items)

    monkeypatch.setattr(runner, "get_system", _fake_get_system)
    common = dict(dataset_path=str(DATA / "smoke.jsonl"), corpus_path=str(DATA / "smoke_corpus.yaml"),
                  out_dir=tmp_path, judge=_Judge(), ragas=_Ragas())

    good = await runner.run("oracle", **common)
    bad = await runner.run("broken", **common)

    for f in ("items.jsonl", "predictions.jsonl", "scores.jsonl", "summary.json", "meta.json", "report.md"):
        assert (good / f).exists(), f

    scores = load_scores(good)
    gradable = [s for s in scores if s.judge_score is not None]
    assert gradable and all(s.judge_score == 1.0 for s in gradable)
    rag = [s for s in scores if s.evidence_recall_at_k]
    assert all(s.evidence_recall_at_k[10] == 1.0 for s in rag)
    assert all(s.correct_refusal == 1.0 for s in scores if not s.answerable)

    # Rescore without a judge keeps offline metrics and drops judge labels.
    faithful = [s for s in scores if s.ragas]
    assert faithful and all(s.ragas["faithfulness"] == 1.0 for s in faithful)
    report = (good / "report.md").read_text(encoding="utf-8")
    assert "## By document" in report and "faithfulness (RAGAS)" in report

    # Rescore without a judge keeps offline metrics and drops judge labels and RAGAS.
    await runner.rescore(good, use_judge=False)
    rescored = load_scores(good)
    assert all(s.judge_score is None and not s.ragas for s in rescored)
    await runner.rescore(good, judge=_Judge(), ragas=_Ragas())
    assert load_scores(good)[0].judge_score is not None

    cmp = compare(load_scores(bad), load_scores(good))
    assert cmp["metrics"]["correctness (judge)"]["b_better"] is True

    csv_path = calibrate.export(good, n=5)
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["human_label"] = r["judge_label"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    res = calibrate.score(csv_path)
    assert res["n"] == 5 and res["agreement"] == 1.0
