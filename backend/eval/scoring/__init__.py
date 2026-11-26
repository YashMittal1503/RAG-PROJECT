"""
Scoring: turns (GoldItem, Prediction) pairs into ItemScore objects.

Lexical, retrieval, citation, abstention, routing and SQL-result metrics are
deterministic and offline. Answer correctness and key-point recall use the LLM
judge; faithfulness, answer relevancy, context precision and context recall use
RAGAS (same judge model) when a RagasScorer is supplied.
"""

from __future__ import annotations

from eval.schema import GoldItem, ItemScore, Prediction
from eval.scoring.abstention import score_abstention
from eval.scoring.answer import contains_answer, exact_match, token_f1
from eval.scoring.citations import score_citations
from eval.scoring.retrieval import DEFAULT_KS, score_retrieval
from eval.scoring.routing import score_classification, score_routing
from eval.scoring.sql import score_sql


def score_item_offline(item: GoldItem, pred: Prediction, ks: tuple[int, ...] = DEFAULT_KS) -> ItemScore:
    """Every metric that needs no LLM."""
    s = ItemScore(
        item_id=item.id,
        category=item.category,
        answerable=item.answerable,
        difficulty=item.difficulty,
        doc=item.expected_doc or (item.docs[0] if len(item.docs) == 1 else None),
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
    c = score_classification(item.expected_scope, pred.query_scope, item.expected_aggregation, pred.aggregation)
    s.scope_correct, s.aggregation_correct = c["scope_correct"], c["aggregation_correct"]

    q = score_sql(item.expected_route, pred.sql, pred.sql_result, item.gold_sql_rows, item.ordered)
    s.sql_executed, s.sql_result_match = q["sql_executed"], q["sql_result_match"]
    return s


async def score_item(item: GoldItem, pred: Prediction, judge=None, ragas=None) -> ItemScore:
    """Offline metrics, plus judge correctness / key-point recall (judge) and RAGAS metrics (ragas)."""
    s = score_item_offline(item, pred)
    if pred.error is not None:
        return s

    # A scorer that still fails after its retries (e.g. a provider rate limit) must not sink the
    # whole run: record it, keep the other metrics, and let `rescore` fill it in later (cached
    # calls are free, so only the failed ones are repeated).
    if judge is not None:
        # Correctness only where there is something to judge against: a gold answer, or an unanswerable item.
        try:
            if item.answers or not item.answerable:
                v = await judge.grade(item, pred.answer)
                s.judge_label, s.judge_score, s.judge_reason = v.label, v.score, v.reason
            if item.key_points and pred.answer.strip():
                kp = await judge.key_point_recall(item, pred.answer)
                s.key_point_recall, s.missed_key_points = kp["key_point_recall"], kp["missed_key_points"]
        except Exception as e:
            s.scoring_errors.append(f"judge: {e}"[:300])

    if ragas is not None:
        try:
            r = await ragas.score(item, pred, refused=s.refused)
            s.ragas, s.unsupported_claims = r["ragas"], r["unsupported_claims"]
        except Exception as e:
            s.scoring_errors.append(f"ragas: {e}"[:300])
    return s
