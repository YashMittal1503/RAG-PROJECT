"""
Aggregation and reporting: per-category metrics with bootstrap 95% confidence
intervals, plus paired comparison of two runs on the same items.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable

from eval.schema import ItemScore

Metric = Callable[[ItemScore], float | None]

# Ordered metric registry: label -> extractor. None means "not applicable to this item".
METRICS: dict[str, Metric] = {
    "correctness (judge)": lambda s: s.judge_score,
    "answer match": lambda s: s.exact_match,
    "token F1": lambda s: s.token_f1,
    "evidence recall@5": lambda s: s.evidence_recall_at_k.get(5),
    "evidence recall@10": lambda s: s.evidence_recall_at_k.get(10),
    "MRR": lambda s: s.mrr,
    "nDCG@10": lambda s: s.ndcg_at_10,
    "citation precision": lambda s: s.citation_precision,
    "citation recall": lambda s: s.citation_recall,
    "correct refusal (unanswerable)": lambda s: s.correct_refusal,
    "false refusal (answerable)": lambda s: s.false_refusal,
    "route accuracy": lambda s: s.route_correct,
    "doc routing accuracy": lambda s: s.doc_correct,
    "error rate": lambda s: 1.0 if s.error else 0.0,
}
# Metrics where lower is better (affects comparison arrows).
LOWER_IS_BETTER = {"false refusal (answerable)", "error rate"}
HEADLINE = ["correctness (judge)", "evidence recall@10", "MRR", "citation precision",
            "correct refusal (unanswerable)", "false refusal (answerable)"]


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean."""
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    k = len(values)
    means = sorted(sum(rng.choices(values, k=k)) / k for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def summarize_metric(scores: list[ItemScore], metric: Metric) -> dict | None:
    vals = [v for v in (metric(s) for s in scores) if v is not None]
    if not vals:
        return None
    lo, hi = bootstrap_ci(vals)
    return {"mean": sum(vals) / len(vals), "ci95": [lo, hi], "n": len(vals)}


def _percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = min(len(vals) - 1, max(0, round(p * (len(vals) - 1))))
    return vals[idx]


def summarize(scores: list[ItemScore]) -> dict:
    by_cat: dict[str, list[ItemScore]] = defaultdict(list)
    for s in scores:
        by_cat[s.category].append(s)

    def block(group: list[ItemScore]) -> dict:
        out = {name: summarize_metric(group, fn) for name, fn in METRICS.items()}
        lat = [s.total_ms for s in group if not s.error]
        out["latency_ms"] = {"p50": _percentile(lat, 0.5), "p95": _percentile(lat, 0.95)} if lat else None
        out["tokens"] = {
            "prompt_mean": statistics.mean(s.prompt_tokens for s in group),
            "completion_mean": statistics.mean(s.completion_tokens for s in group),
        }
        out["n_items"] = len(group)
        return out

    return {"overall": block(scores), "by_category": {c: block(g) for c, g in sorted(by_cat.items())}}


def compare(a: list[ItemScore], b: list[ItemScore], n: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of run B vs run A on items both runs share, per metric."""
    a_by, b_by = {s.item_id: s for s in a}, {s.item_id: s for s in b}
    common = sorted(set(a_by) & set(b_by))
    result: dict[str, dict] = {"n_common_items": len(common), "metrics": {}}
    rng = random.Random(seed)
    for name, fn in METRICS.items():
        diffs = []
        for i in common:
            va, vb = fn(a_by[i]), fn(b_by[i])
            if va is not None and vb is not None:
                diffs.append(vb - va)
        if not diffs:
            continue
        k = len(diffs)
        means = sorted(sum(rng.choices(diffs, k=k)) / k for _ in range(n))
        lo, hi = means[int(0.025 * n)], means[int(0.975 * n) - 1]
        mean = sum(diffs) / k
        significant = lo > 0 or hi < 0
        better = None
        if significant:
            better = (mean < 0) if name in LOWER_IS_BETTER else (mean > 0)
        result["metrics"][name] = {
            "delta": mean, "ci95": [lo, hi], "n": k,
            "significant": significant, "b_better": better,
        }
    return result


# ── Rendering ─────────────────────────────────────────────────────────────

def _fmt(m: dict | None) -> str:
    if not m:
        return "—"
    lo, hi = m["ci95"]
    return f"{m['mean']:.3f} [{lo:.2f}, {hi:.2f}] (n={m['n']})"


def render_markdown(summary: dict, meta: dict) -> str:
    o = summary["overall"]
    lines = [
        f"# Eval report — {meta.get('system', '?')}",
        "",
        f"- Run: `{meta.get('run_id')}`",
        f"- Dataset: `{meta.get('dataset')}` (hash `{meta.get('dataset_hash')}`), {o['n_items']} items",
        f"- Corpus hash: `{meta.get('corpus_hash')}` · Judge: `{meta.get('judge_model') or 'disabled'}`",
        "",
        "Values are means with bootstrap 95% CIs; n = items the metric applies to.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name in METRICS:
        lines.append(f"| {name} | {_fmt(o[name])} |")
    if o["latency_ms"]:
        lines.append(f"| latency p50 / p95 (ms) | {o['latency_ms']['p50']:.0f} / {o['latency_ms']['p95']:.0f} |")
    lines.append(
        f"| tokens (prompt / completion, mean) | {o['tokens']['prompt_mean']:.0f} / {o['tokens']['completion_mean']:.0f} |"
    )

    lines += ["", "## By category", "", "| Category | n | " + " | ".join(HEADLINE) + " |",
              "|---|---|" + "---|" * len(HEADLINE)]
    for cat, blk in summary["by_category"].items():
        cells = [f"{blk[h]['mean']:.2f}" if blk.get(h) else "—" for h in HEADLINE]
        lines.append(f"| {cat} | {blk['n_items']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_comparison(cmp: dict, name_a: str, name_b: str) -> str:
    lines = [
        f"# {name_b} vs {name_a}",
        "",
        f"Paired over {cmp['n_common_items']} shared items. Δ = {name_b} − {name_a}, bootstrap 95% CI.",
        "",
        "| Metric | Δ | 95% CI | n | Verdict |",
        "|---|---|---|---|---|",
    ]
    for name, m in cmp["metrics"].items():
        verdict = "no significant difference"
        if m["significant"]:
            verdict = f"**{name_b} better**" if m["b_better"] else f"**{name_a} better**"
        lo, hi = m["ci95"]
        lines.append(f"| {name} | {m['delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {m['n']} | {verdict} |")
    return "\n".join(lines) + "\n"


def load_scores(run_dir: str | Path) -> list[ItemScore]:
    path = Path(run_dir) / "scores.jsonl"
    return [ItemScore.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_meta(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "meta.json").read_text(encoding="utf-8"))
