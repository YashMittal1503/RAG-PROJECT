"""Offline tests for the eval harness scorers (no network, no models)."""

import pytest

from eval.schema import Citation, Evidence, GoldItem, Prediction, RetrievedContext
from eval.scoring import score_item, score_item_offline
from eval.scoring.abstention import is_refusal, score_abstention
from eval.scoring.answer import contains_answer, exact_match, normalize, token_f1
from eval.scoring.citations import score_citations
from eval.scoring.retrieval import mrr, ndcg_at_k, quote_in_context, recall_at_k, score_retrieval
from eval.scoring.routing import score_routing


# ── answer ──────────────────────────────────────────────────────────────

def test_normalize_strips_case_punct_articles():
    assert normalize("The  Great, Gatsby!") == "great gatsby"


def test_exact_match_and_aliases():
    assert exact_match("thaddeus crane.", ["Thaddeus Crane"]) == 1.0
    assert exact_match("Crane", ["Thaddeus Crane", "Crane"]) == 1.0
    assert exact_match("Pellow", ["Thaddeus Crane"]) == 0.0


def test_contains_answer_requires_whole_phrase():
    assert contains_answer("The narrator is **Elias**, as Pellow says.", ["Elias"]) == 1.0
    assert contains_answer("Eliasson was there", ["Elias"]) == 0.0


def test_token_f1():
    assert token_f1("a sack of flour", ["sack of flour"]) == pytest.approx(1.0)
    assert token_f1("flour", ["sack of flour"]) == pytest.approx(2 * 1 * (1 / 3) / (1 + 1 / 3))
    assert token_f1("", ["x"]) == 0.0


# ── retrieval ───────────────────────────────────────────────────────────

def test_quote_in_context_exact_and_fuzzy():
    ctx = "He called up to me, “Morning, Elias! Has the old man worked you to the bone yet?”"
    assert quote_in_context("Morning, Elias! Has the old man worked you to the bone yet?", ctx)
    # Formatting noise (markdown, line breaks, curly quotes) should still match.
    assert quote_in_context("Morning,  Elias!\nHas the old man **worked** you", ctx)
    assert not quote_in_context("The storm came on Saturday night", ctx)


def test_quote_longer_than_context_does_not_match():
    assert not quote_in_context("a very long quote " * 10, "a very long quote")


def test_recall_mrr_ndcg():
    # 2 evidence spans; rank1 has nothing, rank2 has ev0, rank3 has ev0+ev1.
    hits = [set(), {0}, {0, 1}]
    assert recall_at_k(hits, 2, 1) == 0.0
    assert recall_at_k(hits, 2, 2) == 0.5
    assert recall_at_k(hits, 2, 3) == 1.0
    assert mrr(hits) == 0.5
    assert 0 < ndcg_at_k(hits, 2, 10) < 1
    assert ndcg_at_k([{0}, {1}], 2, 10) == pytest.approx(1.0)
    assert mrr([set(), set()]) == 0.0


def test_score_retrieval_end_to_end():
    r = score_retrieval(["sack of flour"], ["nothing here", "he tossed a sack of flour onto the jetty"])
    assert r["recall_at_k"][1] == 0.0
    assert r["recall_at_k"][5] == 1.0
    assert r["mrr"] == 0.5


# ── abstention ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "This information is not available in the provided document excerpts.",
    "I couldn't find any relevant sections answering your question.",
    "The document does not mention the narrator's wife.",
    "",
])
def test_refusals_detected(text):
    assert is_refusal(text)


def test_substantive_answer_is_not_refusal():
    long = "The narrator is Elias [Page 1]. " * 15 + "The document does not mention his surname."
    assert not is_refusal(long)
    assert not is_refusal("Six days of shore leave per quarter [Page 1].")


def test_score_abstention_directions():
    assert score_abstention("Not available in the provided document.", answerable=False)["correct_refusal"] == 1.0
    assert score_abstention("Not available in the provided document.", answerable=True)["false_refusal"] == 1.0
    assert score_abstention("Six days.", answerable=True)["false_refusal"] == 0.0


# ── citations / routing ─────────────────────────────────────────────────

def test_citation_precision_recall():
    ev = [Evidence(doc="a", page=3, quote="x"), Evidence(doc="a", page=7, quote="y")]
    r = score_citations([Citation(doc="a", page=3), Citation(doc="b", page=3)], ev)
    assert r["citation_precision"] == 0.5
    assert r["citation_recall"] == 0.5
    assert score_citations([], ev) == {"citation_precision": 0.0, "citation_recall": 0.0}
    assert score_citations([Citation(page=1)], [Evidence(doc="a", quote="x")])["citation_precision"] is None


def test_routing():
    assert score_routing("rag", "rag", "rules", "rules") == {"route_correct": 1.0, "doc_correct": 1.0}
    assert score_routing("rag", "clarify", "rules", None) == {"route_correct": 0.0, "doc_correct": 0.0}
    assert score_routing("rag", None, None, None) == {"route_correct": None, "doc_correct": None}


# ── item scoring ────────────────────────────────────────────────────────

def _item(**kw):
    base = dict(
        id="t1", question="What is the narrator's name?", docs=["lh"], answer="Elias",
        evidence=[Evidence(doc="lh", page=1, quote="Morning, Elias!")],
        category="implicit_inference", expected_doc="lh",
    )
    base.update(kw)
    return GoldItem(**base)


def test_score_item_offline_full():
    pred = Prediction(
        item_id="t1",
        answer="The narrator is Elias [Page 1].",
        contexts=[RetrievedContext(text="unrelated"), RetrievedContext(text="“Morning, Elias!” he called")],
        citations=[Citation(doc="lh", page=1)],
        route="rag", chosen_doc="lh", timings_ms={"retrieve": 100, "generate": 400},
    )
    s = score_item_offline(_item(), pred)
    assert s.exact_match == 1.0
    assert s.evidence_recall_at_k[1] == 0.0 and s.evidence_recall_at_k[5] == 1.0
    assert s.mrr == 0.5
    assert s.citation_precision == 1.0
    assert s.false_refusal == 0.0 and s.correct_refusal is None
    assert s.route_correct == 1.0 and s.doc_correct == 1.0
    assert s.total_ms == 500


class _FakeJudge:
    model = "fake"

    def __init__(self):
        self.calls = 0

    async def grade(self, item, answer):
        from eval.scoring.judge import Verdict
        self.calls += 1
        return Verdict("correct", 1.0, "ok")


@pytest.mark.anyio
async def test_score_item_uses_judge_only_when_gradable():
    judge = _FakeJudge()
    s = await score_item(_item(), Prediction(item_id="t1", answer="Elias"), judge)
    assert s.judge_label == "correct" and s.judge_score == 1.0

    chit = GoldItem(id="c", question="hi", expected_route="chitchat")
    s2 = await score_item(chit, Prediction(item_id="c", answer="Hello!"), judge)
    assert s2.judge_label is None
    assert judge.calls == 1

    errored = await score_item(_item(), Prediction(item_id="t1", answer="", error="boom"), judge)
    assert errored.judge_label is None and judge.calls == 1
