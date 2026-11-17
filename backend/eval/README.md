# Eval harness

Scores any RAG architecture on the same frozen dataset, layer by layer, so changes
and architectures can be compared fairly. It replaces `ragas_eval.py`, which used
single-chunk synthetic questions graded by a 3B model.

Run all commands from `backend/`.

## Quick start

```bash
pip install -r requirements.txt
python -m eval validate eval/data/smoke.jsonl     # schema + evidence quotes exist in the corpus
python -m eval run --system vanilla               # smoke dataset, indexes the smoke corpus first
```

`run` needs the backend's `.env` (Groq, Qdrant) plus a judge key: `GEMINI_API_KEY`,
or `EVAL_JUDGE_API_KEY`. Use `--no-judge` for offline metrics only.

Each run writes `eval/runs/<timestamp>_<system>/`:

| File | Contents |
|---|---|
| `report.md` | Overall and per-category metrics with 95% CIs |
| `summary.json` | Same, machine-readable |
| `predictions.jsonl` | Answers, retrieved contexts, citations, route, timings |
| `scores.jsonl` | Every metric for every item |
| `items.jsonl` | The exact items evaluated |
| `meta.json` | System config, dataset hash, corpus hash, judge model |

## Commands

| Command | What it does |
|---|---|
| `python -m eval validate <dataset> --corpus <manifest>` | Check items and that every evidence quote is in its document |
| `python -m eval index --corpus <manifest> [--force]` | Ingest the corpus into the isolated eval tenant (skipped if unchanged) |
| `python -m eval run --system vanilla --dataset ... --corpus ... [--split dev] [--category X] [--limit N] [--no-judge]` | Run and score |
| `python -m eval rescore <run_dir>` | Re-score saved predictions (e.g. new judge) without re-running the system |
| `python -m eval compare <run_a> <run_b>` | Paired bootstrap comparison on shared items |
| `python -m eval calibrate export <run_dir> -n 50` | Export a CSV to hand-label |
| `python -m eval calibrate score <csv>` | Judge-vs-human agreement and Cohen's kappa |

## Isolation

The eval uses its own tenant id (`EVAL_USER_ID`, default `00000000-0000-4000-8000-0000000e7a11`)
and therefore its own Qdrant collection. It never reads or writes real users' data,
Supabase Storage or Postgres. Don't set `EVAL_USER_ID` to a real user's id: `index`
recreates that collection.

## Metrics

| Layer | Metrics | Notes |
|---|---|---|
| Answer | Judge correctness (correct 1 / partial 0.5 / wrong 0), answer match, token F1 | Judge: Gemini (`EVAL_JUDGE_MODEL`, default `gemini-flash-latest`), cached in `runs/.judge_cache` |
| Retrieval | Evidence recall@1/3/5/10, MRR, nDCG@10 | Matched on **text** (fuzzy quote containment), so chunks, graph passages and tree nodes compare fairly |
| Citations | Precision, recall vs gold evidence pages | Only items whose evidence has page numbers |
| Abstention | Correct refusal (unanswerable items), false refusal (answerable items) | Pattern-based refusal detection in `scoring/abstention.py` |
| Routing | Route accuracy, document-choice accuracy | Uses `expected_route` / `expected_doc` |
| Cost | p50/p95 latency, prompt/completion tokens | Tokens are approximate (generation step only) |

Calibrate the judge before trusting it: hand-label about 50 answers. Kappa ≥ 0.7 means it's usable.

## Dataset format (JSONL, one item per line)

```json
{"id": "gatsby-017", "question": "What is the narrator's name?", "docs": ["gatsby"],
 "answer": "Nick Carraway", "acceptable": ["Nick", "Carraway"],
 "evidence": [{"doc": "gatsby", "page": 12, "quote": "exact text from the book"}],
 "answerable": true, "category": "implicit_inference", "hops": 1, "needs_inference": true,
 "expected_route": "rag", "expected_doc": "gatsby", "difficulty": "hard", "split": "dev",
 "history": [], "tags": ["narrator"]}
```

- **Categories:** `single_fact`, `implicit_inference`, `character_relationship`, `multi_hop`, `temporal`,
  `global_summary`, `unanswerable`, `multi_turn`, `routing`, `tabular_sql`, `table_figure`, `other`.
- **Unanswerable and false-premise items:** `"answerable": false`, `"category": "unanswerable"`, no answer or evidence.
- **Chitchat:** `"expected_route": "chitchat"` and no answer.
- **Splits:** keep about 30% as `holdout` and never look at it while tuning prompts.

## Corpus manifest

See `data/corpus.example.yaml`. Each document has an `id` (used by items), a `path`, and
optionally a `url` (downloaded if missing) and a `sha256` (verified). PDF and TXT are supported.

## Adding an architecture

Subclass `RAGSystem` in `systems/<name>.py`. Implement `index()` and `answer()` returning a
`Prediction` that includes the contexts given to the generator, then register it in
`systems/base.py:get_system`. `systems/vanilla.py` is the reference implementation: it
calls the same functions as the chat endpoint.

## Known gaps

- `vanilla`: the lead-chunk lookup inside `retrieve_chunks` needs Postgres. Without it,
  that step logs a warning and is skipped, so overview questions get slightly less context than in production.
- Spreadsheet / Text-to-SQL items aren't run yet (the corpus loader handles PDF/TXT only).
- Not yet implemented: claim-level faithfulness (RAGChecker-style) and summary key-point recall.
- The smoke dataset only proves the harness works. The real golden set (about 280 items) comes next.
