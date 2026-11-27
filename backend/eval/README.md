# Eval harness

Scores any RAG architecture on the same frozen dataset, layer by layer, so changes
and architectures can be compared fairly. It is the project's only evaluation system
(the old `ragas_eval.py` was removed; see "History" at the end).

Run all commands from `backend/`.

## Quick start

```bash
pip install -r requirements-eval.txt                # app requirements + ragas
python -m eval validate eval/data/smoke.jsonl     # schema + evidence quotes exist in the corpus
python -m eval run --system vanilla               # smoke dataset, indexes the smoke corpus first
```

`run` needs the backend's `.env` (Groq, Qdrant) plus keys for the judge provider.
Use `--no-judge` for offline metrics only, or `--no-ragas` to keep judge correctness but
skip RAGAS (the most expensive part: up to 10 judge calls per item).

### Judge

One judge configuration drives every LLM-judged metric (correctness, key-point recall and RAGAS):

| Setting | Default | Notes |
|---|---|---|
| `--judge-provider` / `EVAL_JUDGE_PROVIDER` | `gemini` | `nvidia`, `gemini`, `mistral` or `groq`. Prefer a provider other than the generator's (Groq). `nvidia` is free at ~40 RPM and is the best fit for full runs; Gemini's free tier allows only ~20 requests/day on some models |
| `EVAL_JUDGE_MODEL` | `gemini-flash-latest`, `mistral-small-latest`, `openai/gpt-oss-120b` | per provider |
| keys | `<PROVIDER>_API_KEY`, `_2`, `_3`, `<PROVIDER>_API_KEYS` | the same variables the app uses; `EVAL_JUDGE_API_KEY` overrides. Calls rotate across all keys |
| `EVAL_JUDGE_BASE_URL` | provider's OpenAI-compatible endpoint | |

Judge verdicts are cached in `runs/.judge_cache` and RAGAS calls in `runs/.ragas_cache`,
so `rescore` on unchanged predictions is free. If scoring fails midway (quota), the
predictions are already saved: run `rescore <run_dir>` to finish.

Each run writes `eval/runs/<timestamp>_<system>/`:

| File | Contents |
|---|---|
| `report.md` | Overall, per-category and per-document metrics with 95% CIs |
| `summary.json` | Same, machine-readable |
| `predictions.jsonl` | Answers, retrieved contexts, citations, route, timings |
| `scores.jsonl` | Every metric for every item |
| `items.jsonl` | The exact items evaluated |
| `meta.json` | System config, dataset hash, corpus hash, judge provider and model, RAGAS version and settings |

## Commands

| Command | What it does |
|---|---|
| `python -m eval validate <dataset> --corpus <manifest>` | Check items and that every evidence quote is in its document |
| `python -m eval index --corpus <manifest> [--force]` | Ingest the corpus into the isolated eval tenant (skipped if unchanged) |
| `python -m eval run --system vanilla --dataset ... --corpus ... [--split dev] [--category X] [--limit N] [--no-judge] [--no-ragas] [--ragas-metrics a,b] [--judge-provider P]` | Run and score |
| `python -m eval rescore <run_dir> [same scoring flags]` | Re-score saved predictions (new judge, new metric) without re-running the system |
| `python -m eval classify [--dataset eval/data/query_classification.jsonl]` | Accuracy of the query classifier (intent, scope, aggregation): one LLM call per question, with a confusion table and the mistakes. Use it while tuning the classifier prompt |
| `python -m eval draft --corpus <manifest> --existing eval/data/golden.jsonl` | LLM-draft fact questions for you to review. Count per document is proportional to its length (1 per 4,000 tokens, 4–30), minus facts already in `--existing` |
| `python -m eval view <run> [<run_b>]` | HTML dashboard: score cards, all metrics, by category/document, clickable items; two runs = comparison. `run`/`rescore` write `report.html` and `runs/index.html` automatically |
| `python -m eval compare <run_a> <run_b>` | Paired bootstrap comparison on shared items |
| `python -m eval calibrate export <run_dir> -n 50` | Export a CSV to hand-label |
| `python -m eval calibrate score <csv>` | Judge-vs-human agreement and Cohen's kappa |

## Isolation

The eval uses its own tenant id (`EVAL_USER_ID`, default `00000000-0000-4000-8000-0000000e7a11`)
and therefore its own Qdrant collection and its own DuckDB file (`data/tabular/<eval user id>.duckdb`,
for spreadsheets). It never reads or writes real users' data, Supabase Storage or Postgres.
Don't set `EVAL_USER_ID` to a real user's id: `index` recreates that collection and drops
DuckDB tables that aren't in the corpus.

## Metrics

| Layer | Metrics | Notes |
|---|---|---|
| Answer | Judge correctness (correct 1 / partial 0.5 / wrong 0), answer match, token F1 | Judge: Gemini (`EVAL_JUDGE_MODEL`, default `gemini-flash-latest`), cached in `runs/.judge_cache` |
| RAGAS (generation) | Faithfulness, answer relevancy | Faithfulness: share of the answer's statements supported by the contexts the generator saw (unsupported ones are listed per item in `scores.jsonl`). Relevancy: does the answer address the question. Both skip refusals |
| RAGAS (retrieval) | Context precision@5, context recall | Judged against the gold answer, so answerable items only. Complements the text-match evidence recall / MRR below |
| Completeness | Key-point recall: share of an item's `key_points` the answer covers | Only items with `key_points` (summaries, multi-part answers) |
| Text-to-SQL | SQL executed, SQL result match (execution accuracy vs `gold_sql_rows`) | Ignores row order (unless `ordered`), column order, column names and float noise; extra columns allowed |
| Retrieval | Evidence recall@1/3/5/10, MRR, nDCG@10 | Matched on **text** (fuzzy quote containment), so chunks, graph passages and tree nodes compare fairly |
| Citations | Precision, recall vs gold evidence pages | Only items whose evidence has page numbers |
| Abstention | Correct refusal (unanswerable items), false refusal (answerable items) | Pattern-based refusal detection in `scoring/abstention.py` |
| Routing | Route accuracy, document-choice accuracy, query scope accuracy, aggregation detection accuracy | Uses `expected_route` / `expected_doc` / `expected_scope` / `expected_aggregation` |
| Cost | p50/p95 latency, prompt/completion tokens | Tokens are approximate (generation step only) |

Calibrate the judge before trusting it: hand-label about 50 answers. Kappa ≥ 0.7 means it's usable.
(Calibration covers correctness labels only; spot-check faithfulness by reading `unsupported_claims`.)

RAGAS sees every context the generator saw (context precision is capped at the top 5,
one judge call each). Answer relevancy embeds with the app's own local model, so it
makes no embedding API calls.

Reading correctness and faithfulness together:

| Correct? | Faithful? | Meaning |
|---|---|---|
| yes | high | Working as intended |
| no | high | The generator faithfully used the wrong contexts: a retrieval problem |
| no | low | The generator made things up: a generation/prompt problem |
| yes | low | Right by luck or from the model's own knowledge, not from your documents |

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
- **Spreadsheets:** `"expected_route": "sql"`, `"category": "tabular_sql"`, and `gold_sql_rows` (the rows a correct
  query returns). Optional: `gold_sql` (a reference query, for humans; the test suite runs the smoke ones) and
  `"ordered": true` for ORDER BY / top-N questions. No evidence needed.
  ```json
  {"id": "sales-004", "question": "What is the total revenue per region?", "docs": ["sales"],
   "answer": "North 1.2M, South 0.9M", "gold_sql_rows": [["North", 1200000], ["South", 900000]],
   "expected_route": "sql", "expected_doc": "sales", "category": "tabular_sql"}
  ```
- **Query classification:** optional `expected_scope` (`global`, `section`, `metadata`, `targeted`) and `expected_aggregation` (true/false) are scored against what `classify_query` returned. Spreadsheet questions about the whole dataset use `"expected_route": "tabular_overview"`.
- **Summaries:** add `key_points`, a list of facts a complete answer must contain, to measure completeness.
- **Splits:** keep about 30% as `holdout` and never look at it while tuning prompts.

## Corpus manifest

See `data/corpus.example.yaml`. Each document has an `id` (used by items), a `path`, and
optionally a `url` (downloaded if missing) and a `sha256` (verified). PDF, TXT, CSV and XLSX are supported.
Spreadsheets go through the app's Text-to-SQL path: each sheet becomes a DuckDB table and isn't chunked or embedded.

The real corpus is `data/corpus.yaml`, pointing at `eval_files/` (6 PDFs: a paper, a handbook,
an annual report, a slide deck, two novels; 3 spreadsheets). Both are gitignored because the
files are copyrighted or private. Its golden set is `data/golden.jsonl`:

    python -m eval validate eval/data/golden.jsonl --corpus eval/data/corpus.yaml
    python -m eval run --dataset eval/data/golden.jsonl --corpus eval/data/corpus.yaml

Spreadsheet `gold_sql_rows` were computed with pandas from the source files, not by an LLM.
Novel evidence quotes are kept to short phrases.

## Adding an architecture

Subclass `RAGSystem` in `systems/<name>.py`. Implement `index()` and `answer()` returning a
`Prediction` that includes the contexts given to the generator, then register it in
`systems/base.py:get_system`. `systems/vanilla.py` is the reference implementation: it
calls the same functions as the chat endpoint.

## Known gaps

- `calibrate` checks correctness labels only, not RAGAS scores or key-point recall.
- The smoke dataset only proves the harness works; `data/golden.jsonl` is the real set. It has no `holdout` split yet.

## History

`ragas_eval.py` (removed) generated one question per chunk with `ministral-3b-2512`, used
that model's answer as ground truth, and graded with the same model on the top 3 contexts.
Its last run (Oct 31, 2026; 23 questions over 9 personal documents) scored faithfulness 0.81,
answer relevancy 0.87, context precision 0.87, context recall 0.93, factual correctness 0.42.
Those numbers aren't comparable with this harness: different questions, unverified ground
truth, a much weaker judge, and contexts cut to 3 (which counts claims drawn from contexts
4-10 as hallucinations). What it did well lives on here: multi-provider judge with key
rotation, per-document breakdown, re-scoring without regenerating, and question drafting
(now `draft`, with evidence checks and human review).
