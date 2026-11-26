"""
Tests for the HTML dashboard (eval/view.py) and the size-proportional draft plan (eval/draft.py).
Builds a tiny synthetic run directory; no models, no network.
"""

import json
import re

from eval.draft import MAX_FACTS, MIN_FACTS, existing_fact_counts, fact_target, plan_drafts
from eval.runner import write_outputs
from eval.schema import GoldItem, ItemScore, Prediction, RetrievedContext
from eval.view import write_comparison, write_index, write_report


def _run(tmp_path, name, judge):
    items = [
        GoldItem(id="a", question="Who kept the light?", docs=["lh"], answer="Crane",
                 evidence=[{"doc": "lh", "quote": "Crane"}], category="single_fact", expected_doc="lh"),
        GoldItem(id="b", question="How many arrivals </script><b>x</b>?", docs=["arr"], answer="10",
                 gold_sql_rows=[[10]], category="tabular_sql", expected_route="sql", expected_doc="arr"),
    ]
    preds = [
        Prediction(item_id="a", answer="Thaddeus Crane", route="rag", contexts=[RetrievedContext(text="the old keeper, Crane", doc="lh", page=1)]),
        Prediction(item_id="b", answer="10 arrivals", route="sql", sql="SELECT count(*) FROM t",
                   sql_result={"columns": ["n"], "rows": [[10]]}),
    ]
    scores = [
        ItemScore(item_id="a", category="single_fact", answerable=True, difficulty="easy", doc="lh", judge_label="correct" if judge else "wrong",
                  judge_score=1.0 if judge else 0.0, ragas={"faithfulness": 0.5}, unsupported_claims=["made up claim"],
                  evidence_recall_at_k={5: 1.0, 10: 1.0}, mrr=1.0, total_ms=1200),
        ItemScore(item_id="b", category="tabular_sql", answerable=True, difficulty="easy", doc="arr", sql_executed=1.0,
                  sql_result_match=1.0, judge_score=1.0, total_ms=800),
    ]
    run_dir = tmp_path / "runs" / name
    run_dir.mkdir(parents=True)
    write_outputs(run_dir, items, preds, scores, {"run_id": name, "system": "vanilla", "dataset": "golden.jsonl"})
    return run_dir


def _payload(html_text):
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html_text, re.S).group(1))


def test_report_embeds_rows_and_escapes_script_tags(tmp_path):
    run = _run(tmp_path, "r1", judge=True)
    page = (run / "report.html").read_text(encoding="utf-8")   # written automatically by write_outputs
    assert page.count("</script>") == 2                       # the question's "</script>" can't close the data block
    data = _payload(page)
    assert data["mode"] == "run" and len(data["run"]["rows"]) == 2
    row = data["run"]["rows"][0]
    assert row["item"]["answer"] == "Crane" and row["score"]["unsupported_claims"] == ["made up claim"]
    assert data["run"]["summary"]["overall"]["correctness (judge)"]["mean"] == 1.0
    assert "faithfulness (RAGAS)" in data["desc"]
    assert write_report(run) == run / "report.html"


def test_comparison_and_index(tmp_path):
    a, b = _run(tmp_path, "r1", judge=False), _run(tmp_path, "r2", judge=True)
    out = write_comparison(a, b)
    data = _payload(out.read_text(encoding="utf-8"))
    assert data["mode"] == "compare" and data["base"]["name"] == "r1"
    assert data["cmp"]["metrics"]["correctness (judge)"]["delta"] == 0.5
    index = _payload(write_index(tmp_path / "runs").read_text(encoding="utf-8"))
    assert [r["name"] for r in index["runs"]] == ["r2", "r1"] and index["runs"][0]["has_report"]


def test_fact_targets_scale_with_document_length():
    assert fact_target(1_800) == MIN_FACTS                     # 6-page slide deck
    assert fact_target(52_000) == 13                           # 88-page annual report
    assert fact_target(145_000) == MAX_FACTS                   # 300-page novel
    plan = plan_drafts({"deck": 1_800, "report": 52_000}, {"deck": 6, "report": 5}, oversample=1.5)
    assert plan["deck"]["draft"] == 0                          # already above target
    assert plan["report"] == {"tokens": 52_000, "target": 13, "existing": 5, "draft": 12}


def test_existing_fact_counts_only_counts_targeted_answerable_rag_items():
    items = [
        GoldItem(id="1", question="q", docs=["d"], answer="a", evidence=[{"doc": "d", "quote": "a"}], category="single_fact", expected_doc="d"),
        GoldItem(id="2", question="q", docs=["d"], answer="a", evidence=[{"doc": "d", "quote": "a"}], category="global_summary", expected_doc="d"),
        GoldItem(id="3", question="q", docs=["d"], answer="a", evidence=[{"doc": "d", "quote": "a"}], category="single_fact",
                 expected_doc="d", expected_scope="metadata"),
        GoldItem(id="4", question="q", docs=["d"], answerable=False, category="unanswerable", expected_doc="d"),
    ]
    assert existing_fact_counts(items) == {"d": 1}


def test_failed_judge_call_is_recorded_not_fatal():
    import asyncio

    from eval.scoring import score_item
    from eval.scoring.judge import retry_wait

    class Boom:
        async def grade(self, *a):
            raise RuntimeError("Error code: 429 - Too Many Requests")

    item = GoldItem(id="a", question="q", docs=["d"], answer="x", evidence=[{"doc": "d", "quote": "x"}], expected_doc="d")
    s = asyncio.run(score_item(item, Prediction(item_id="a", answer="x"), judge=Boom()))
    assert s.judge_label is None and s.scoring_errors and "429" in s.scoring_errors[0]
    assert s.exact_match == 1.0                                   # offline metrics still computed
    assert retry_wait(RuntimeError("Error code: 429"), 0) == 15 and retry_wait(ValueError("bad json"), 1) == 2
