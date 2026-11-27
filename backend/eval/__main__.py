"""
Command-line entry point. Run from the backend/ directory:

  python -m eval validate eval/data/smoke.jsonl --corpus eval/data/smoke_corpus.yaml
  python -m eval index    --corpus eval/data/smoke_corpus.yaml [--force]
  python -m eval run      --system vanilla --dataset eval/data/smoke.jsonl --corpus eval/data/smoke_corpus.yaml
  python -m eval rescore  eval/runs/<run_id>
  python -m eval compare  eval/runs/<run_a> eval/runs/<run_b>
  python -m eval classify [--dataset eval/data/query_classification.jsonl]
  python -m eval draft    --corpus eval/data/corpus.yaml --existing eval/data/golden.jsonl
  python -m eval view     eval/runs/<run_id> [eval/runs/<other_run_id>]
  python -m eval calibrate export eval/runs/<run_id> -n 50
  python -m eval calibrate score  eval/runs/<run_id>/calibration.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def _cmd_validate(args) -> int:
    from eval.corpus import doc_texts, ensure_files, load_manifest
    from eval.dataset import load_dataset, validate_against_corpus

    items = load_dataset(args.dataset)
    docs = load_manifest(args.corpus)
    ensure_files(docs)
    problems = validate_against_corpus(items, doc_texts(docs), {d.id for d in docs if d.is_tabular})
    cats: dict[str, int] = {}
    for i in items:
        cats[i.category] = cats.get(i.category, 0) + 1
    print(f"{len(items)} items, {len(docs)} documents")
    for c, n in sorted(cats.items()):
        print(f"  {c:<24} {n}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK: all items valid and all evidence quotes found in the corpus.")
    return 0


def _cmd_index(args) -> int:
    from eval.corpus import index_corpus, load_manifest

    result = asyncio.run(index_corpus(load_manifest(args.corpus), force=args.force))
    print(json.dumps(result, indent=2))
    return 0


def _cmd_run(args) -> int:
    from eval.runner import run

    run_dir = asyncio.run(run(
        system_name=args.system,
        dataset_path=args.dataset,
        corpus_path=args.corpus,
        split=args.split,
        categories=args.category,
        limit=args.limit,
        use_judge=not args.no_judge,
        concurrency=args.concurrency,
        skip_index=args.skip_index,
        out_dir=args.out,
        use_ragas=not args.no_ragas,
        judge_provider=args.judge_provider,
        ragas_metrics=args.ragas_metrics,
    ))
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    print(f"Run saved to {run_dir}")
    return 0


def _cmd_rescore(args) -> int:
    from eval.runner import rescore

    run_dir = asyncio.run(rescore(
        args.run_dir, use_judge=not args.no_judge, use_ragas=not args.no_ragas,
        judge_provider=args.judge_provider, ragas_metrics=args.ragas_metrics,
    ))
    print((run_dir / "report.md").read_text(encoding="utf-8"))
    return 0


def _cmd_compare(args) -> int:
    from eval.report import compare, load_meta, load_scores, render_comparison

    a, b = Path(args.run_a), Path(args.run_b)
    ma, mb = load_meta(a), load_meta(b)
    if ma.get("dataset_hash") != mb.get("dataset_hash"):
        print("WARNING: runs used different dataset versions; only shared item ids are compared.\n")
    if ma.get("judge_model") != mb.get("judge_model"):
        print("WARNING: runs used different judges; correctness deltas may reflect the judge, not the system.\n")
    cmp = compare(load_scores(a), load_scores(b))
    print(render_comparison(cmp, f"{ma['system']} ({a.name})", f"{mb['system']} ({b.name})"))
    return 0


def _cmd_draft(args) -> int:
    from eval.draft import draft

    out, n, plan = asyncio.run(draft(args.corpus, args.out, per_doc=args.per_doc, provider=args.judge_provider,
                                     force=args.force, existing_path=args.existing))
    print(f"{'document':<20} {'tokens':>8} {'target':>7} {'existing':>9} {'drafted':>8}")
    for doc_id, p in plan.items():
        print(f"{doc_id:<20} {p['tokens']:>8} {p['target']:>7} {p['existing']:>9} {p['draft']:>8}")
    print(f"\nWrote {n} draft item(s) to {out}. Review them before copying into a golden set.")
    return 0


def _cmd_classify(args) -> int:
    from eval.classify import render, run_classification

    result = asyncio.run(run_classification(args.dataset, concurrency=args.concurrency))
    print(render(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved to {args.out}")
    return 0


def _cmd_view(args) -> int:
    from eval.view import write_comparison, write_index, write_report

    runs_dir = Path(args.runs_dir)
    if args.runs:
        out = write_report(args.runs[0]) if len(args.runs) == 1 else write_comparison(args.runs[0], args.runs[1])
        runs_dir = Path(args.runs[-1]).parent
        print(f"Wrote {out}")
    if runs_dir.exists():
        print(f"Wrote {write_index(runs_dir)}")
    elif not args.runs:
        raise FileNotFoundError(f"no runs directory at {runs_dir}")
    if args.open and args.runs:
        import webbrowser

        webbrowser.open(Path(out).resolve().as_uri())
    return 0


def _cmd_calibrate(args) -> int:
    from eval import calibrate

    if args.action == "export":
        print(f"Wrote {calibrate.export(args.path, n=args.n)}: fill in the human_label column.")
        return 0
    res = calibrate.score(args.path)
    print(f"n={res['n']}  agreement={res['agreement']:.2%}  Cohen's kappa={res['kappa']:.3f}")
    print("Judge looks usable." if res["kappa"] >= 0.7 else "Kappa < 0.7: improve the judge prompt/model before trusting it.")
    if res["disagreements"]:
        print("Disagreements:", ", ".join(res["disagreements"]))
    return 0


def _judge_args(s) -> None:
    from eval.scoring.judge import PROVIDERS
    from eval.scoring.ragas_metrics import METRIC_NAMES

    s.add_argument("--judge-provider", choices=sorted(PROVIDERS), help="judge LLM provider (default: EVAL_JUDGE_PROVIDER or gemini)")
    s.add_argument("--no-ragas", action="store_true", help="skip RAGAS metrics (keeps judge correctness)")
    s.add_argument("--ragas-metrics", type=lambda v: [m.strip() for m in v.split(",") if m.strip()],
                   help=f"comma-separated subset of {','.join(METRIC_NAMES)} (default: all)")


def _utf8_console() -> None:
    """Reports contain characters like '—' and 'Δ'; don't crash on Windows cp1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _utf8_console()
    p = argparse.ArgumentParser(prog="python -m eval", description="DocTalk RAG evaluation harness")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("validate", help="check a dataset's schema and that every evidence quote is in the corpus")
    s.add_argument("dataset")
    s.add_argument("--corpus", default=str(DATA / "smoke_corpus.yaml"))
    s.set_defaults(fn=_cmd_validate)

    s = sub.add_parser("index", help="ingest the corpus into the isolated eval tenant")
    s.add_argument("--corpus", default=str(DATA / "smoke_corpus.yaml"))
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=_cmd_index)

    s = sub.add_parser("run", help="run a system over a dataset and score it")
    s.add_argument("--system", default="vanilla")
    s.add_argument("--dataset", default=str(DATA / "smoke.jsonl"))
    s.add_argument("--corpus", default=str(DATA / "smoke_corpus.yaml"))
    s.add_argument("--split", choices=["dev", "holdout"])
    s.add_argument("--category", action="append", help="repeatable")
    s.add_argument("--limit", type=int)
    s.add_argument("--no-judge", action="store_true", help="skip every LLM-judged metric (offline metrics only)")
    _judge_args(s)
    s.add_argument("--concurrency", type=int, default=1)
    s.add_argument("--skip-index", action="store_true")
    s.add_argument("--out", help="runs directory (default eval/runs)")
    s.set_defaults(fn=_cmd_run)

    s = sub.add_parser("rescore", help="re-score saved predictions without re-running the system")
    s.add_argument("run_dir")
    s.add_argument("--no-judge", action="store_true")
    _judge_args(s)
    s.set_defaults(fn=_cmd_rescore)

    s = sub.add_parser("compare", help="paired comparison of two runs")
    s.add_argument("run_a")
    s.add_argument("run_b")
    s.set_defaults(fn=_cmd_compare)

    s = sub.add_parser("draft", help="LLM-draft candidate items from the corpus for human review")
    s.add_argument("--corpus", default=str(DATA / "smoke_corpus.yaml"))
    s.add_argument("--out", default=str(DATA / "drafts.jsonl"))
    s.add_argument("--existing", help="golden set whose fact items count toward each document's target")
    s.add_argument("--per-doc", type=int, help="fixed count per document (default: proportional to its length)")
    s.add_argument("--judge-provider")
    s.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    s.set_defaults(fn=_cmd_draft)

    s = sub.add_parser("classify", help="accuracy of the query classifier (intent / scope / aggregation)")
    s.add_argument("--dataset", default=str(DATA / "query_classification.jsonl"))
    s.add_argument("--concurrency", type=int, default=2)
    s.add_argument("--out", help="also save the full result as JSON")
    s.set_defaults(fn=_cmd_classify)

    s = sub.add_parser("view", help="HTML dashboard for a run, a comparison of two runs, and the runs index")
    s.add_argument("runs", nargs="*", help="one run dir (report) or two (compare the second against the first)")
    s.add_argument("--runs-dir", default=str(Path(__file__).resolve().parent / "runs"), help="where the index is written")
    s.add_argument("--open", action="store_true", help="open the page in the browser")
    s.set_defaults(fn=_cmd_view)

    s = sub.add_parser("calibrate", help="check judge agreement with human labels")
    s.add_argument("action", choices=["export", "score"])
    s.add_argument("path", help="run dir (export) or calibration.csv (score)")
    s.add_argument("-n", type=int, default=50)
    s.set_defaults(fn=_cmd_calibrate)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        return args.fn(args)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
