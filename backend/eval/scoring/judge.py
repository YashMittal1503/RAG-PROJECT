"""
LLM-as-judge: answer correctness against the gold answer, and key-point recall
for summary-style items. RAGAS metrics (ragas_metrics.py) use the same judge
configuration, so every LLM-judged number in a run comes from one model.

Provider, model and keys come from JudgeConfig.resolve():
  EVAL_JUDGE_PROVIDER   gemini (default) | mistral | groq
  EVAL_JUDGE_MODEL      overrides the provider's default model
  EVAL_JUDGE_API_KEY    overrides the provider's key pool (otherwise <PROVIDER>_API_KEY,
                        <PROVIDER>_API_KEY_2/_3 and comma-separated <PROVIDER>_API_KEYS)
  EVAL_JUDGE_BASE_URL   overrides the provider's OpenAI-compatible endpoint
Prefer a judge from a different provider than the generator (Groq): a model grading
its own family's answers tends to be lenient.

Requests rotate across all keys in the pool. Verdicts are cached on disk, keyed by
(judge model, prompt version, prompt), so re-running a report or re-scoring an
unchanged prediction costs nothing.

Calibrate the judge against your own labels (`python -m eval calibrate`) before
trusting it: aim for Cohen's kappa >= 0.7.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from eval.schema import GoldItem

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
KEY_POINTS_PROMPT_VERSION = "k1"
MIN_JUDGE_TOKENS = 2048

def retry_wait(err: Exception, attempt: int) -> float:
    """Seconds to wait before retrying a judge call: long for rate limits (free tiers reset per
    minute), short exponential backoff for anything else."""
    msg = str(err).lower()
    if getattr(err, "status_code", None) == 429 or "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return min(15 * (attempt + 1), 60)
    return min(2 ** attempt, 30)

LABEL_SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# provider -> (OpenAI-compatible base URL, default judge model, env prefix for keys)
PROVIDERS: dict[str, tuple[str, str, str]] = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-flash-latest", "GEMINI"),
    "mistral": ("https://api.mistral.ai/v1", "mistral-small-latest", "MISTRAL"),
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ"),
    # Free (~40 RPM) and separate from the generator's Groq budget: the recommended judge.
    "nvidia": ("https://integrate.api.nvidia.com/v1", "nvidia/nemotron-3-ultra-550b-a55b", "NVIDIA"),
}
DEFAULT_PROVIDER = "gemini"
GEMINI_BASE_URL = PROVIDERS["gemini"][0]
DEFAULT_JUDGE_MODEL = PROVIDERS["gemini"][1]

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial grader for a document question-answering system.
You compare a SYSTEM ANSWER to a GOLD ANSWER written by a human who read the source document.

Grade with exactly one label:
- "correct": the system answer states the gold answer (or an accepted alias), or something semantically equivalent, and does not contradict it. Extra correct detail is fine. Wording, formatting and citations do not matter.
- "partial": the system answer contains part of the gold answer but misses an essential part, or hedges between the right answer and a wrong one.
- "wrong": the system answer is missing, contradicts the gold answer, answers a different question, or refuses when an answer exists.

If the question is marked UNANSWERABLE (the documents do not contain the answer, or the question has a false premise):
- "correct": the system clearly says the information is not in the documents, or corrects the false premise.
- "partial": the system expresses uncertainty but still offers a speculative answer.
- "wrong": the system confidently answers as if the information existed.

Judge only against the gold answer. Do not use your own knowledge of the book or topic to reward or penalize.
Respond with JSON only: {"label": "correct" | "partial" | "wrong", "reason": "<one sentence>"}"""

KEY_POINTS_SYSTEM_PROMPT = """You check how completely an AI answer covers a list of KEY POINTS a good answer must include.

For each key point, decide whether the ANSWER conveys it (same meaning; wording may differ; partial mentions that lose the essential fact do not count).
Judge only against the key points; do not reward extra content and do not use your own knowledge.

Respond with JSON only, one entry per key point, in the given order:
{"points": [{"index": <1-based index>, "covered": true | false}]}"""


@dataclass
class Verdict:
    label: str
    score: float
    reason: str
    cached: bool = False


def _load_dotenv() -> None:
    """Load backend/.env if python-dotenv is available, without overriding real env vars."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def _env_keys(prefix: str) -> list[str]:
    keys = [os.environ.get(f"{prefix}_API_KEY{suffix}", "").strip() for suffix in ("", "_2", "_3")]
    keys += [k.strip() for k in (os.environ.get(f"{prefix}_API_KEYS") or "").split(",")]
    return list(dict.fromkeys(k for k in keys if k))


@dataclass
class JudgeConfig:
    """Which LLM judges, and with which keys. Shared by Judge and the RAGAS scorer."""

    provider: str
    model: str
    base_url: str
    keys: list[str] = field(default_factory=list, repr=False)

    @classmethod
    def resolve(cls, provider: str | None = None, model: str | None = None) -> "JudgeConfig":
        _load_dotenv()
        provider = (provider or os.environ.get("EVAL_JUDGE_PROVIDER") or DEFAULT_PROVIDER).lower().strip()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown judge provider '{provider}'. Choose from {sorted(PROVIDERS)}.")
        base_url, default_model, prefix = PROVIDERS[provider]
        override = os.environ.get("EVAL_JUDGE_API_KEY", "").strip()
        keys = [override] if override else _env_keys(prefix)
        if not keys:
            raise RuntimeError(
                f"No judge API key for '{provider}'. Set EVAL_JUDGE_API_KEY or {prefix}_API_KEY (backend/.env works)."
            )
        if provider == "groq":
            logger.warning("Judge provider is Groq, the same provider as the generator: expect a lenient judge.")
        return cls(
            provider=provider,
            model=model or os.environ.get("EVAL_JUDGE_MODEL") or default_model,
            base_url=os.environ.get("EVAL_JUDGE_BASE_URL") or base_url,
            keys=keys,
        )

    def async_clients(self, max_retries: int = 0) -> list[Any]:
        from openai import AsyncOpenAI

        return [AsyncOpenAI(api_key=k, base_url=self.base_url, max_retries=max_retries, timeout=90) for k in self.keys]


def build_user_prompt(item: GoldItem, answer: str) -> str:
    gold = item.answer if item.answerable else "(UNANSWERABLE — the documents do not contain this information)"
    aliases = ", ".join(item.acceptable) if item.acceptable else "none"
    history = ""
    if item.history:
        history = "CONVERSATION SO FAR:\n" + "\n".join(f"{t.role}: {t.content}" for t in item.history) + "\n\n"
    return (
        f"{history}QUESTION: {item.question}\n"
        f"ANSWERABLE: {'yes' if item.answerable else 'no'}\n"
        f"GOLD ANSWER: {gold}\n"
        f"ACCEPTED ALIASES: {aliases}\n\n"
        f"SYSTEM ANSWER:\n{answer.strip() or '(empty)'}"
    )


def build_key_points_prompt(question: str, answer: str, key_points: list[str]) -> str:
    points = "\n".join(f"{i}. {p}" for i, p in enumerate(key_points, start=1))
    return f"QUESTION: {question}\n\nKEY POINTS:\n{points}\n\nANSWER:\n{answer.strip()}"


def _json_object(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in judge output: {raw[:200]!r}")
    return json.loads(m.group(0))


def parse_verdict(raw: str) -> tuple[str, str]:
    """Extract (label, reason) from the judge's reply; tolerant of code fences and prose."""
    text = raw.strip()
    try:
        data = _json_object(text)
        label = str(data.get("label", "")).strip().lower()
        if label in LABEL_SCORES:
            return label, str(data.get("reason", "")).strip()
    except (ValueError, json.JSONDecodeError):
        pass
    m = re.search(r"\b(correct|partial|wrong)\b", text.lower())
    if m:
        return m.group(1), text[:200]
    raise ValueError(f"Unparseable judge output: {raw[:200]!r}")


def parse_key_points(raw: str, n: int) -> list[bool]:
    """Covered flag per key point, in order. Missing entries count as not covered."""
    data = _json_object(raw)
    covered = [False] * n
    for p in data.get("points", []) or []:
        try:
            idx = int(p.get("index")) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n:
            covered[idx] = p.get("covered") is True or str(p.get("covered")).lower() == "true"
    return covered


class Judge:
    """Grades answers with an LLM. Pass `client` to inject a fake in tests."""

    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | Path | None = None,
        client: Any | None = None,
        concurrency: int = 2,
        max_retries: int = 6,
        config: JudgeConfig | None = None,
        provider: str | None = None,
    ):
        if client is not None:
            self.config = config or JudgeConfig(provider="test", model=model or DEFAULT_JUDGE_MODEL, base_url="")
            self._clients = [client]
        else:
            self.config = config or JudgeConfig.resolve(provider, model)
            self._clients = self.config.async_clients()
        self.model = self.config.model
        self.cache_dir = Path(cache_dir) if cache_dir else RUNS_DIR / ".judge_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._next = 0
        self._sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries

    def _client(self) -> Any:
        c = self._clients[self._next % len(self._clients)]
        self._next += 1
        return c

    def _cache_path(self, version: str, system_prompt: str, user_prompt: str) -> Path:
        payload = json.dumps([self.model, version, system_prompt, user_prompt], ensure_ascii=False)
        return self.cache_dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"

    async def _cached_call(
        self,
        *,
        version: str,
        system_prompt: str,
        user_prompt: str,
        parse: Callable[[str], dict],
        max_tokens: int,
        item_id: str,
    ) -> tuple[dict, bool]:
        """One judge call with on-disk cache, key rotation and retries. Returns (parsed, cached)."""
        path = self._cache_path(version, system_prompt, user_prompt)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), True

        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        last_err: Exception | None = None
        async with self._sem:
            for attempt in range(self.max_retries):
                try:
                    resp = await self._client().chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0,
                        # Reasoning models (Nemotron, gpt-oss) think before answering; a tight
                        # budget ends mid-thought with empty content, so leave room for both.
                        max_tokens=max(max_tokens, MIN_JUDGE_TOKENS),
                    )
                    msg = resp.choices[0].message
                    raw = msg.content or ""
                    if not raw.strip():  # some providers put the whole reply in a reasoning field
                        raw = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
                    parsed = parse(raw)
                    path.write_text(json.dumps({**parsed, "model": self.model}, ensure_ascii=False), encoding="utf-8")
                    return parsed, False
                except Exception as e:  # rate limits, transient network, unparseable output
                    last_err = e
                    wait = retry_wait(e, attempt)
                    logger.warning(f"Judge call failed for {item_id} (attempt {attempt + 1}): {e}; retrying in {wait}s")
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Judge failed for {item_id}: {last_err}")

    async def grade(self, item: GoldItem, answer: str) -> Verdict:
        """Answer correctness against the gold answer."""

        def _parse(raw: str) -> dict:
            label, reason = parse_verdict(raw)
            return {"label": label, "reason": reason}

        data, cached = await self._cached_call(
            version=PROMPT_VERSION,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=build_user_prompt(item, answer),
            parse=_parse,
            max_tokens=300,
            item_id=item.id,
        )
        return Verdict(data["label"], LABEL_SCORES[data["label"]], data.get("reason", ""), cached=cached)

    async def key_point_recall(self, item: GoldItem, answer: str) -> dict:
        """Share of the item's gold key points the answer covers."""
        n = len(item.key_points)
        data, _ = await self._cached_call(
            version=KEY_POINTS_PROMPT_VERSION,
            system_prompt=KEY_POINTS_SYSTEM_PROMPT,
            user_prompt=build_key_points_prompt(item.question, answer, item.key_points),
            parse=lambda raw: {"covered": parse_key_points(raw, n)},
            max_tokens=600,
            item_id=item.id,
        )
        covered = data["covered"]
        return {
            "key_point_recall": sum(covered) / n,
            "missed_key_points": [p for p, c in zip(item.key_points, covered) if not c],
        }
