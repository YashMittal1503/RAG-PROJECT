"""
Judge calibration: check that the LLM judge agrees with a human before trusting it.

1. `python -m eval calibrate export <run_dir> -n 50` writes calibration.csv with
   the question, gold answer, system answer and the judge's label.
2. Fill in the `human_label` column (correct / partial / wrong) yourself,
   without looking at the judge's label first.
3. `python -m eval calibrate score <run_dir>/calibration.csv` reports agreement
   and Cohen's kappa. Rule of thumb: kappa >= 0.7 means the judge is usable.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

LABELS = ("correct", "partial", "wrong")


def cohens_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def export(run_dir: str | Path, n: int = 50, seed: int = 0) -> Path:
    run_dir = Path(run_dir)
    preds = {
        p["item_id"]: p
        for p in (json.loads(l) for l in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }
    scores = [json.loads(l) for l in (run_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    items = {i["id"]: i for i in (json.loads(l) for l in (run_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    graded = [s for s in scores if s.get("judge_label")]
    random.Random(seed).shuffle(graded)
    out = run_dir / "calibration.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "question", "answerable", "gold_answer", "system_answer", "judge_label", "human_label"])
        for s in graded[:n]:
            it = items[s["item_id"]]
            w.writerow([s["item_id"], it["question"], it["answerable"], it["answer"],
                        preds[s["item_id"]]["answer"], s["judge_label"], ""])
    return out


def score(csv_path: str | Path) -> dict:
    with Path(csv_path).open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("human_label", "").strip()]
    if not rows:
        raise ValueError("No rows have human_label filled in yet.")
    bad = [r["item_id"] for r in rows if r["human_label"].strip().lower() not in LABELS]
    if bad:
        raise ValueError(f"human_label must be one of {LABELS}; bad rows: {bad}")
    judge = [r["judge_label"].strip().lower() for r in rows]
    human = [r["human_label"].strip().lower() for r in rows]
    disagreements = [r["item_id"] for r, j, h in zip(rows, judge, human) if j != h]
    return {
        "n": len(rows),
        "agreement": sum(j == h for j, h in zip(judge, human)) / len(rows),
        "kappa": cohens_kappa(judge, human),
        "disagreements": disagreements,
    }
