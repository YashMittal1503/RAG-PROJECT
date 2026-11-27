"""
RAGAS metrics on the harness's own predictions (ragas 0.4 "collections" API).

  faithfulness       share of the answer's statements supported by the contexts the
                     generator saw. Catches hallucination. Needs no gold answer.
  answer_relevancy   does the answer address the question? (RAGAS generates questions
                     from the answer and compares them with the real one by embedding)
  context_precision  are the relevant contexts ranked high? Judged against the gold
                     answer, over the top CONTEXT_PRECISION_TOP_K contexts.
  context_recall     share of the gold answer's statements that the contexts support.

What applies to which item:
  - Nothing runs for errors, empty answers, or predictions without contexts (chitchat,
    clarification, "no documents").
  - faithfulness and answer_relevancy skip refusals: a refusal makes no claims and
    isn't meant to address the question.
  - context_precision and context_recall need a gold answer, so only answerable items.
  - All four always see ALL contexts the generator saw, except context_precision's
    top-k cap. (The old ragas_eval.py cut contexts to 3 for every metric, which marked
    claims drawn from contexts 4-10 as hallucinations.)

The judge LLM, keys and provider come from JudgeConfig (judge.py), so RAGAS and the
correctness judge use the same model. Each LLM call is cached on disk under
runs/.ragas_cache, keyed by (ragas version, model, prompt, output schema), so
`rescore` on unchanged predictions costs nothing. Calls rotate across the key pool
and retry with backoff on rate limits.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

from eval.schema import GoldItem, Prediction
from eval.scoring.judge import retry_wait

logger = logging.getLogger(__name__)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
CONTEXT_PRECISION_TOP_K = 5
ANSWER_RELEVANCY_STRICTNESS = 2     # questions generated per answer (RAGAS default 3; 2 halves the noise of 1)
JUDGE_MAX_TOKENS = 4096             # statement lists for long answers overflow the 1024 default


def _ragas():
    try:
        import ragas  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "RAGAS metrics need the eval extras: pip install -r requirements-eval.txt (or run with --no-ragas)."
        ) from e
    import ragas

    return ragas


def applicable_metrics(item: GoldItem, pred: Prediction, refused: bool, enabled: tuple[str, ...]) -> list[str]:
    """Which RAGAS metrics make sense for this (item, prediction)."""
    if pred.error or not pred.answer.strip() or not pred.contexts:
        return []
    out = []
    if not refused:
        out += ["faithfulness", "answer_relevancy"]
    if item.answerable and item.answer.strip():
        out += ["context_precision", "context_recall"]
    return [m for m in out if m in enabled]


def _user_input(item: GoldItem) -> str:
    """The question as RAGAS sees it; multi-turn items carry their history so follow-ups make sense."""
    if not item.history:
        return item.question
    history = "\n".join(f"{t.role}: {t.content}" for t in item.history)
    return f"Conversation so far:\n{history}\n\nQuestion: {item.question}"


def _clean(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def _build_llm_class():
    from ragas.llms.base import InstructorBaseRagasLLM

    class RotatingCachedLLM(InstructorBaseRagasLLM):
        """Round-robins RAGAS structured calls across a key pool, with retries and a disk cache."""

        def __init__(self, llms: list[Any], model: str, cache_dir: Path, version: str, max_retries: int = 6,
                     concurrency: int = 2):
            self.llms = llms
            self.model = model
            self.cache_dir = cache_dir
            self.version = version
            self.max_retries = max_retries
            self._next = 0
            self._sem = asyncio.Semaphore(concurrency)
            cache_dir.mkdir(parents=True, exist_ok=True)

        def _path(self, prompt: str, response_model: type) -> Path:
            schema = json.dumps(response_model.model_json_schema(), sort_keys=True)
            key = json.dumps([self.version, self.model, prompt, schema], ensure_ascii=False)
            return self.cache_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"

        def _pick(self) -> Any:
            llm = self.llms[self._next % len(self.llms)]
            self._next += 1
            return llm

        async def agenerate(self, prompt, response_model):
            path = self._path(prompt, response_model)
            if path.exists():
                return response_model.model_validate_json(path.read_text(encoding="utf-8"))
            last: Exception | None = None
            async with self._sem:
                for attempt in range(self.max_retries):
                    try:
                        result = await self._pick().agenerate(prompt, response_model)
                        path.write_text(result.model_dump_json(), encoding="utf-8")
                        return result
                    except Exception as e:  # 429s, timeouts, malformed structured output
                        last = e
                        wait = retry_wait(e, attempt)
                        logger.warning(f"RAGAS judge call failed (attempt {attempt + 1}): {e}; retrying in {wait}s")
                        await asyncio.sleep(wait)
            raise RuntimeError(f"RAGAS judge call failed after {self.max_retries} attempts: {last}")

        def generate(self, prompt, response_model):
            return asyncio.run(self.agenerate(prompt, response_model))

    return RotatingCachedLLM


def _build_embeddings_class():
    from ragas.embeddings.base import BaseRagasEmbedding

    class AppEmbeddings(BaseRagasEmbedding):
        """The app's own local embedding model (no API calls), for answer_relevancy."""

        def embed_text(self, text: str, **kwargs) -> list[float]:
            from app.services.embedding import embed_query

            return list(embed_query(text))

        def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
            from app.services.embedding import embed_texts

            return [list(v) for v in embed_texts(texts)]

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return await asyncio.to_thread(self.embed_text, text)

        async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_texts, texts)

    return AppEmbeddings


class RagasScorer:
    """Scores (item, prediction) pairs with RAGAS. Pass `llm` / `embeddings` to inject fakes in tests."""

    def __init__(
        self,
        config=None,
        metrics: tuple[str, ...] = METRIC_NAMES,
        cache_dir: str | Path | None = None,
        llm: Any | None = None,
        embeddings: Any | None = None,
        concurrency: int = 2,
    ):
        unknown = set(metrics) - set(METRIC_NAMES)
        if unknown:
            raise ValueError(f"Unknown RAGAS metric(s) {sorted(unknown)}. Available: {list(METRIC_NAMES)}")
        ragas = _ragas()
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )

        self.version = ragas.__version__
        self.enabled = tuple(metrics)
        cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1] / "runs" / ".ragas_cache"

        if llm is None:
            from eval.scoring.judge import JudgeConfig
            from ragas.llms import llm_factory

            config = config or JudgeConfig.resolve()
            inner = [
                llm_factory(config.model, provider="openai", client=c, max_tokens=JUDGE_MAX_TOKENS, temperature=0.0)
                for c in config.async_clients()
            ]
            llm = _build_llm_class()(inner, config.model, cache_dir, self.version, concurrency=concurrency)
            self.model = config.model
        else:
            self.model = getattr(llm, "model", "injected")
        embeddings = embeddings if embeddings is not None else _build_embeddings_class()()

        self._metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings, strictness=ANSWER_RELEVANCY_STRICTNESS),
            "context_precision": ContextPrecisionWithReference(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }

    def config(self) -> dict:
        return {"ragas_version": self.version, "model": self.model, "metrics": list(self.enabled),
                "context_precision_top_k": CONTEXT_PRECISION_TOP_K,
                "answer_relevancy_strictness": ANSWER_RELEVANCY_STRICTNESS}

    async def _faithfulness(self, question: str, answer: str, contexts: list[str]) -> tuple[float | None, list[str]]:
        """Score plus the statements judged unsupported (for debugging). Uses Faithfulness's
        own two steps so the statements and verdicts are the exact ones behind the score."""
        m = self._metrics["faithfulness"]
        statements = await m._create_statements(question, answer)
        if not statements:
            return None, []
        verdicts = await m._create_verdicts(statements, "\n".join(contexts))
        unsupported = [s.statement for s in verdicts.statements if not s.verdict]
        return _clean(m._compute_score(verdicts)), unsupported

    async def score(self, item: GoldItem, pred: Prediction, refused: bool) -> dict:
        """{"ragas": {metric: score|None}, "unsupported_claims": [...]} for the applicable metrics."""
        names = applicable_metrics(item, pred, refused, self.enabled)
        out: dict = {"ragas": {}, "unsupported_claims": []}
        if not names:
            return out
        question, answer = _user_input(item), pred.answer
        contexts = [c.text for c in pred.contexts]

        async def run(name: str):
            try:
                if name == "faithfulness":
                    value, unsupported = await self._faithfulness(question, answer, contexts)
                    out["unsupported_claims"] = unsupported
                    return name, value
                if name == "answer_relevancy":
                    r = await self._metrics[name].ascore(user_input=question, response=answer)
                elif name == "context_precision":
                    r = await self._metrics[name].ascore(
                        user_input=question, reference=item.answer,
                        retrieved_contexts=contexts[:CONTEXT_PRECISION_TOP_K])
                else:
                    r = await self._metrics[name].ascore(
                        user_input=question, retrieved_contexts=contexts, reference=item.answer)
                return name, _clean(r.value)
            except Exception as e:  # one metric failing must not lose the others
                logger.warning(f"RAGAS {name} failed for {item.id}: {e}")
                return name, None

        for name, value in await asyncio.gather(*(run(n) for n in names)):
            out["ragas"][name] = value
        return out
