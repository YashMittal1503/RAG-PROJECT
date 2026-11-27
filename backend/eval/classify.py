"""
Accuracy of the query classifier (app.services.query.classify_query) on labelled questions.

Cheap and fast: one LLM call per question, no retrieval or generation. Use it while
tuning QUERY_CLASSIFIER_PROMPT; the full `run` also scores scope / aggregation on
items that declare expected_scope / expected_aggregation.

  python -m eval classify                       # eval/data/query_classification.jsonl
  python -m eval classify --dataset my_cases.jsonl --concurrency 2

Each line: {"id", "question", "history"?, "expected_intent"?, "expected_scope"?,
"expected_aggregation"?, "notes"?}. Fields left out aren't scored.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from eval.schema import QueryScope, Turn


class ClassificationCase(BaseModel):
    id: str
    question: str
    history: list[Turn] = Field(default_factory=list)
    expected_intent: Literal["retrieve", "chitchat"] | None = None
    expected_scope: QueryScope | None = None
    expected_aggregation: bool | None = None
    notes: str = ""


def load_cases(path: str | Path) -> list[ClassificationCase]:
    cases, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        case = ClassificationCase.model_validate_json(line)
        if case.id in seen:
            raise ValueError(f"line {n}: duplicate id {case.id}")
        seen.add(case.id)
        cases.append(case)
    return cases


def score_cases(cases: list[ClassificationCase], predictions: list[dict]) -> dict:
    """Accuracy per field, scope confusion counts and the list of mistakes."""
    fields = {"intent": "expected_intent", "scope": "expected_scope", "aggregation": "expected_aggregation"}
    totals = {f: [0, 0] for f in fields}          # field -> [correct, n]
    confusion: Counter = Counter()                 # (expected_scope, predicted_scope)
    mistakes = []
    for case, pred in zip(cases, predictions):
        wrong = {}
        for f, exp_attr in fields.items():
            expected = getattr(case, exp_attr)
            if expected is None:
                continue
            got = pred.get(f)
            totals[f][1] += 1
            if got == expected:
                totals[f][0] += 1
            else:
                wrong[f] = {"expected": expected, "got": got}
        if case.expected_scope is not None:
            confusion[(case.expected_scope, pred.get("scope"))] += 1
        if wrong:
            mistakes.append({"id": case.id, "question": case.question, **wrong})
    return {
        "accuracy": {f: (c / n if n else None) for f, (c, n) in totals.items()},
        "n": {f: n for f, (_, n) in totals.items()},
        "scope_confusion": {f"{e} -> {g}": k for (e, g), k in sorted(confusion.items(), key=lambda x: (x[0][0], str(x[0][1])))},
        "mistakes": mistakes,
    }


async def run_classification(path: str | Path, concurrency: int = 2) -> dict:
    from app.services.query import classify_query

    cases = load_cases(path)
    sem = asyncio.Semaphore(concurrency)

    async def _one(case: ClassificationCase) -> dict:
        history = [{"role": t.role, "content": t.content} for t in case.history]
        async with sem:
            qc = await classify_query(case.question, history)
        return qc.to_dict()

    predictions = await asyncio.gather(*(_one(c) for c in cases))
    result = score_cases(cases, list(predictions))
    result["fallbacks"] = sum(1 for p in predictions if p.get("method") == "fallback")
    return result


def render(result: dict) -> str:
    lines = ["Query classification accuracy", ""]
    for f, acc in result["accuracy"].items():
        n = result["n"][f]
        lines.append(f"  {f:<12} {'—' if acc is None else f'{acc:.1%}'}  (n={n})")
    if result.get("fallbacks"):
        lines.append(f"\n  {result['fallbacks']} call(s) fell back to the default (LLM error or unparseable output)")
    lines += ["", "Scope (expected -> predicted):"]
    lines += [f"  {k:<26} {v}" for k, v in result["scope_confusion"].items()]
    if result["mistakes"]:
        lines += ["", "Mistakes:"]
        lines += [f"  {m['id']}: {m['question']!r} " + json.dumps({k: v for k, v in m.items() if k not in ('id', 'question')})
                  for m in result["mistakes"]]
    return "\n".join(lines)
