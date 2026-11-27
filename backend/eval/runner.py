"""Runs a system over a dataset, scores every item and writes a self-contained run directory."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.corpus import corpus_hash, ensure_files, load_manifest
from eval.dataset import dataset_hash, filter_items, load_dataset
from eval.report import render_markdown, summarize
from eval.schema import GoldItem, ItemScore, Prediction
from eval.scoring import score_item
from eval.systems.base import get_system

logger = logging.getLogger(__name__)
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.model_dump_json() + "\n")


async def score_predictions(
    items: list[GoldItem], preds: list[Prediction], judge=None, ragas=None
) -> list[ItemScore]:
    by_id = {p.item_id: p for p in preds}
    return list(await asyncio.gather(*(score_item(i, by_id[i.id], judge, ragas) for i in items)))


def build_scorers(
    use_judge: bool = True,
    use_ragas: bool = True,
    judge=None,
    ragas=None,
    provider: str | None = None,
    ragas_metrics: list[str] | None = None,
):
    """Judge and RAGAS scorer sharing one judge configuration (provider, model, key pool)."""
    if not use_judge:
        return None, None
    from eval.scoring.judge import Judge, JudgeConfig

    config = None
    if judge is None or (use_ragas and ragas is None):
        config = JudgeConfig.resolve(provider)
    if judge is None:
        judge = Judge(config=config)
    if use_ragas and ragas is None:
        from eval.scoring.ragas_metrics import METRIC_NAMES, RagasScorer

        ragas = RagasScorer(config=config, metrics=tuple(ragas_metrics or METRIC_NAMES))
    return judge, (ragas if use_ragas else None)


def _scorer_meta(judge, ragas) -> dict:
    return {
        "judge_model": judge.model if judge else None,
        "judge_provider": getattr(getattr(judge, "config", None), "provider", None),
        "ragas": ragas.config() if ragas else None,
    }


def write_outputs(run_dir: Path, items, preds, scores, meta: dict) -> dict:
    _write_jsonl(run_dir / "items.jsonl", items)
    _write_jsonl(run_dir / "predictions.jsonl", preds)
    _write_jsonl(run_dir / "scores.jsonl", scores)
    summary = summarize(scores)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(summary, meta), encoding="utf-8")
    try:  # the dashboard is a convenience: never fail a run because of it
        from eval.view import write_index, write_latest, write_report

        write_report(run_dir)
        write_index(run_dir.parent)
        if run_dir.parent.resolve() == (Path(__file__).resolve().parent / "runs"):
            write_latest(run_dir.parent)   # eval/report.html: newest fully scored run
    except Exception as e:
        logger.warning(f"Could not write the HTML report: {e}")
    return summary


async def run(
    system_name: str,
    dataset_path: str,
    corpus_path: str,
    split: str | None = None,
    categories: list[str] | None = None,
    limit: int | None = None,
    use_judge: bool = True,
    concurrency: int = 1,
    skip_index: bool = False,
    out_dir: str | Path | None = None,
    judge=None,
    use_ragas: bool = True,
    ragas=None,
    judge_provider: str | None = None,
    ragas_metrics: list[str] | None = None,
) -> Path:
    items = filter_items(load_dataset(dataset_path), split=split, categories=categories, limit=limit)
    if not items:
        raise ValueError("No items match the given filters.")
    docs = load_manifest(corpus_path)
    ensure_files(docs)
    system = get_system(system_name, docs)
    # Build scorers before running the system, so a missing key or package fails fast.
    judge, ragas = build_scorers(use_judge, use_ragas, judge, ragas, judge_provider, ragas_metrics)

    if not skip_index:
        logger.info(f"Indexing corpus for '{system.name}'...")
        await system.index()

    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{system.name}"
    run_dir = Path(out_dir or RUNS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def _one(item: GoldItem) -> Prediction:
        nonlocal done
        async with sem:
            pred = await system.answer(item)
        done += 1
        status = "ERROR " + pred.error if pred.error else f"{pred.total_ms:.0f}ms"
        logger.info(f"[{done}/{len(items)}] {item.id}: {status}")
        return pred

    t0 = time.perf_counter()
    preds = list(await asyncio.gather(*(_one(i) for i in items)))
    # Save predictions first: if scoring fails (quota, network), `rescore` can finish the job.
    _write_jsonl(run_dir / "items.jsonl", items)
    _write_jsonl(run_dir / "predictions.jsonl", preds)
    t_gen = time.perf_counter() - t0
    meta = {
        "run_id": run_id,
        "system": system.name,
        "system_config": system.config(),
        "dataset": str(dataset_path),
        "dataset_hash": dataset_hash(dataset_path),
        "corpus": str(corpus_path),
        "corpus_hash": corpus_hash(docs),
        "filters": {"split": split, "categories": categories, "limit": limit},
        **_scorer_meta(judge, ragas),
        "n_items": len(items),
        "n_errors": sum(1 for p in preds if p.error),
        "generation_time_s": round(t_gen, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    # Written before scoring too, so a crash mid-scoring still leaves the run's provenance for `rescore`.
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    scores = await score_predictions(items, preds, judge, ragas)
    meta["wall_time_s"] = round(time.perf_counter() - t0, 1)
    write_outputs(run_dir, items, preds, scores, meta)
    return run_dir


async def rescore(
    run_dir: str | Path,
    use_judge: bool = True,
    judge=None,
    use_ragas: bool = True,
    ragas=None,
    judge_provider: str | None = None,
    ragas_metrics: list[str] | None = None,
) -> Path:
    """Re-score saved predictions (e.g. new judge, new metric) without re-running the system."""
    run_dir = Path(run_dir)
    items = [GoldItem.model_validate_json(l) for l in (run_dir / "items.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    preds = [Prediction.model_validate_json(l) for l in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    judge, ragas = build_scorers(use_judge, use_ragas, judge, ragas, judge_provider, ragas_metrics)
    scores = await score_predictions(items, preds, judge, ragas)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"run_id": run_dir.name}
    meta.update(_scorer_meta(judge, ragas))
    meta["rescored_at"] = datetime.now(timezone.utc).isoformat()
    write_outputs(run_dir, items, preds, scores, meta)
    return run_dir
