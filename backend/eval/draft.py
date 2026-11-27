"""
Draft candidate golden items from the corpus, for a human to review.

The old ragas_eval.py generated one question per chunk and used the LLM's answer as
ground truth without review. This keeps the useful part (fast candidate generation)
and fixes the rest:
  - chunks are spread evenly across each document instead of the first few;
  - each candidate must include a verbatim evidence quote, and candidates whose quote
    isn't actually in the document are dropped;
  - output goes to a separate drafts file, tagged "needs_review". Nothing becomes
    part of a golden set until you check it and copy it over.

  - the number of questions per document is proportional to its content (one fact per
    TOKENS_PER_FACT tokens, clamped), and `--existing` subtracts what a dataset already has.

  python -m eval draft --corpus eval/data/corpus.yaml --existing eval/data/golden.jsonl --out eval/data/drafts.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from eval.corpus import CorpusDoc, doc_texts, ensure_files, load_manifest, read_pages
from eval.schema import GoldItem
from eval.scoring.judge import JudgeConfig, _json_object
from eval.scoring.retrieval import quote_in_context

logger = logging.getLogger(__name__)

DRAFT_SYSTEM_PROMPT = """You write test questions for a document question-answering system.
Given an EXCERPT from a document, write ONE question a real reader might ask that the excerpt answers.

Rules:
- The question must make sense without seeing the excerpt (name people, places or things; no "in this passage").
- Prefer questions that need understanding, not just copying one phrase. Avoid yes/no questions.
- "answer" is short and precise (a phrase or one sentence), using only the excerpt.
- "evidence" is ONE short phrase (at most 15 words) copied EXACTLY, character for character, from the excerpt, that supports the answer.
- "category" is one of: single_fact, implicit_inference, character_relationship, temporal, multi_hop, other.

Respond with JSON only: {"question": "...", "answer": "...", "evidence": "...", "category": "..."}"""

CATEGORIES = {"single_fact", "implicit_inference", "character_relationship", "temporal", "multi_hop", "other"}


# How many fact questions each document should have, proportional to its content:
# one per TOKENS_PER_FACT tokens of extracted text, clamped to [MIN_FACTS, MAX_FACTS].
# A 6-page deck gets the minimum, a 300-page novel the maximum.
TOKENS_PER_FACT = 4000
MIN_FACTS = 4
MAX_FACTS = 30
# Review usually rejects about a third of drafts, so draft a few more than the gap.
OVERSAMPLE = 1.5
FACT_CATEGORIES = {"single_fact", "implicit_inference", "character_relationship", "temporal", "multi_hop"}


def fact_target(tokens: int) -> int:
    """Fact questions a document of `tokens` tokens should have in the golden set."""
    return max(MIN_FACTS, min(MAX_FACTS, round(tokens / TOKENS_PER_FACT)))


def existing_fact_counts(items) -> dict[str, int]:
    """Answerable, targeted fact items per document already in a dataset."""
    counts: dict[str, int] = {}
    for it in items:
        if not it.answerable or it.category not in FACT_CATEGORIES or it.expected_route != "rag":
            continue
        if it.expected_scope not in (None, "targeted"):
            continue
        doc = it.expected_doc or (it.docs[0] if len(it.docs) == 1 else None)
        if doc:
            counts[doc] = counts.get(doc, 0) + 1
    return counts


def plan_drafts(doc_tokens: dict[str, int], existing: dict[str, int] | None = None,
                oversample: float = OVERSAMPLE) -> dict[str, dict]:
    """Per document: its token count, the target fact count, what exists, and how many to draft."""
    existing = existing or {}
    plan = {}
    for doc, tokens in doc_tokens.items():
        target = fact_target(tokens)
        have = existing.get(doc, 0)
        gap = max(0, target - have)
        plan[doc] = {"tokens": tokens, "target": target, "existing": have,
                     "draft": 0 if gap == 0 else max(1, round(gap * oversample))}
    return plan


def pick_chunks(chunks: list, n: int) -> list:
    """n chunks spread evenly over the document (start, middle, end), skipping tiny ones."""
    usable = [c for c in chunks if len(c.content.split()) >= 60] or chunks
    if n >= len(usable):
        return usable
    step = len(usable) / n
    return [usable[int(i * step + step / 2)] for i in range(n)]


async def _draft_one(client, model: str, doc: CorpusDoc, chunk, full_text: str) -> dict | None:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Document: {doc.title}\n\nEXCERPT:\n{chunk.content[:3000]}"},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    data = _json_object(resp.choices[0].message.content or "")
    q, a, ev = (str(data.get(k, "")).strip() for k in ("question", "answer", "evidence"))
    if not (q and a and ev):
        return None
    if not quote_in_context(ev, full_text):
        logger.info(f"Dropped a {doc.id} draft: evidence quote not found in the document")
        return None
    category = str(data.get("category", "other")).strip()
    return {
        "question": q,
        "docs": [doc.id],
        "answer": a,
        "evidence": [{"doc": doc.id, "page": chunk.page_number, "quote": ev}],
        "category": category if category in CATEGORIES else "other",
        "expected_doc": doc.id,
        "split": "dev",
        "tags": ["draft", "needs_review"],
        "notes": f"LLM draft from chunk {chunk.chunk_index}. Verify the answer and evidence before use.",
    }


async def draft(
    corpus_path: str | Path,
    out_path: str | Path,
    per_doc: int | None = None,
    provider: str | None = None,
    force: bool = False,
    existing_path: str | Path | None = None,
) -> tuple[Path, int, dict]:
    """Draft fact questions. By default the count per document follows its size (fact_target),
    minus fact items already in `existing_path`; `per_doc` forces a fixed count instead."""
    from app.services.chunking import chunk_text, count_tokens
    from app.services.parsing import PageText
    from eval.dataset import load_dataset

    out_path = Path(out_path)
    if out_path.exists() and not force:
        raise ValueError(f"{out_path} exists; pass --force to overwrite it.")
    docs = [d for d in load_manifest(corpus_path) if not d.is_tabular]  # SQL gold rows need a human
    ensure_files(docs)
    texts = doc_texts(docs)

    existing = existing_fact_counts(load_dataset(existing_path)) if existing_path else {}
    plan = plan_drafts({d.id: count_tokens(texts[d.id]) for d in docs}, existing)
    if per_doc is not None:
        for p in plan.values():
            p["draft"] = per_doc
    for doc_id, p in plan.items():
        logger.info(f"{doc_id}: {p['tokens']} tokens -> target {p['target']} facts, "
                    f"{p['existing']} existing, drafting {p['draft']}")

    jobs = []
    for doc in docs:
        n = plan[doc.id]["draft"]
        if not n:
            continue
        pages = [PageText(page_number=n_, text=t) for n_, t in read_pages(doc)]
        for chunk in pick_chunks(chunk_text(pages, filename=doc.filename), n):
            jobs.append((doc, chunk))
    if not jobs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("// Nothing to draft: every document already has its target number of fact questions.\n", encoding="utf-8")
        return out_path, 0, plan
    config = JudgeConfig.resolve(provider)
    clients = config.async_clients(max_retries=8)  # free tiers 429 often; back off instead of dropping drafts

    sem = asyncio.Semaphore(2)

    async def _run(i: int, doc: CorpusDoc, chunk):
        async with sem:
            try:
                return await _draft_one(clients[i % len(clients)], config.model, doc, chunk, texts[doc.id])
            except Exception as e:
                logger.warning(f"Draft failed for {doc.id} chunk {chunk.chunk_index}: {e}")
                return None

    results = await asyncio.gather(*(_run(i, d, c) for i, (d, c) in enumerate(jobs)))
    lines = ["// LLM-drafted candidates. Review each one, fix it or delete it, then copy it into a golden set."]
    counts: dict[str, int] = {}
    for r in results:
        if r is None:
            continue
        counts[r["docs"][0]] = counts.get(r["docs"][0], 0) + 1
        r = {"id": f"draft-{r['docs'][0]}-{counts[r['docs'][0]]:03d}", **r}
        GoldItem.model_validate(r)  # never write an item the loader would reject
        lines.append(json.dumps(r, ensure_ascii=False))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path, len(lines) - 1, plan
