"""
Scoring: turns (GoldItem, Prediction) pairs into ItemScore objects.

Lexical, retrieval, citation, abstention and routing metrics are deterministic
and offline. Answer correctness uses the LLM judge when one is supplied.
"""

from __future__ import annotations

from eval.schema import GoldItem, ItemScore, Prediction
from eval.scoring.abstention import score_abstention
from eval.scoring.answer import contains_answer, exact_match, token_f1
from eval.scoring.citations import score_citations
from eval.scoring.retrieval import DEFAULT_KS, score_retrieval
from eval.scoring.routing import score_routing


def score_item_offline(item: GoldItem, pred: Prediction, ks: tuple[int, ...] = DEFAULT_KS) -> ItemScore:
    """Every metric that needs no LLM."""
    s = ItemScore(
        item_id=item.id,
        category=item.category,
        answerable=item.answerable,
        difficulty=item.difficulty,
        total_ms=pred.total_ms,
        prompt_tokens=pred.prompt_tokens,
        completion_tokens=pred.completion_tokens,
        error=pred.error,
    )

    if item.answerable and item.answers:
        # Long-form answers: count containment as an exact match, otherwise fall back to strict EM.
        s.exact_match = max(exact_match(pred.answer, item.answers), contains_answer(pred.answer, item.answers))
        s.token_f1 = token_f1(pred.answer, item.answers)

    if item.evidence:
        r = score_retrieval([e.quote for e in item.evidence], [c.text for c in pred.contexts], ks)
        s.evidence_recall_at_k = r["recall_at_k"]
        s.mrr = r["mrr"]
        s.ndcg_at_10 = r["ndcg_at_10"]
        c = score_citations(pred.citations, item.evidence)
        s.citation_precision = c["citation_precision"]
        s.citation_recall = c["citation_recall"]

    a = score_abstention(pred.answer, item.answerable)
    s.refused, s.false_refusal, s.correct_refusal = a["refused"], a["false_refusal"], a["correct_refusal"]

    r = score_routing(item.expected_route, pred.route, item.expected_doc, pred.chosen_doc)
    s.route_correct, s.doc_correct = r["route_correct"], r["doc_correct"]
    return s


async def score_item(item: GoldItem, pred: Prediction, judge=None) -> ItemScore:
    s = score_item_offline(item, pred)
    # Judge only where there is something to judge against: a gold answer, or an unanswerable item.
    gradable = item.answers or not item.answerable
    if judge is not None and pred.error is None and gradable:
        v = await judge.grade(item, pred.answer)
        s.judge_label, s.judge_score, s.judge_reason = v.label, v.score, v.reason
    return s
