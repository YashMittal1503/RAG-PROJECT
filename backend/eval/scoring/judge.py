"""
LLM-as-judge for answer correctness against the gold answer.

Uses Gemini through its OpenAI-compatible endpoint so the judge comes from a
different provider than the Groq generator. Verdicts are cached on disk, keyed
by (judge model, prompt version, item, prediction), so re-running a report or
re-scoring an unchanged prediction costs nothing.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.schema import GoldItem

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_JUDGE_MODEL = "gemini-flash-latest"
LABEL_SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

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


def _judge_keys() -> list[str]:
    _load_dotenv()
    keys: list[str] = []
    for name in ("EVAL_JUDGE_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        if os.environ.get(name):
            keys.append(os.environ[name].strip())
    for k in (os.environ.get("GEMINI_API_KEYS") or "").split(","):
        if k.strip():
            keys.append(k.strip())
    return list(dict.fromkeys(keys))


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


def parse_verdict(raw: str) -> tuple[str, str]:
    """Extract (label, reason) from the judge's reply; tolerant of code fences and prose."""
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            label = str(data.get("label", "")).strip().lower()
            if label in LABEL_SCORES:
                return label, str(data.get("reason", "")).strip()
        except json.JSONDecodeError:
            pass
    m = re.search(r"\b(correct|partial|wrong)\b", text.lower())
    if m:
        return m.group(1), text[:200]
    raise ValueError(f"Unparseable judge output: {raw[:200]!r}")


class Judge:
    """Grades answers with an LLM. Pass `client` to inject a fake in tests."""

    def __init__(
        self,
        model: str | None = None,
        cache_dir: str | Path | None = None,
        client: Any | None = None,
        concurrency: int = 4,
        max_retries: int = 4,
    ):
        self.model = model or os.environ.get("EVAL_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[1] / "runs" / ".judge_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._clients = [client] if client is not None else self._make_clients()
        self._next = 0
        self._sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries

    def _make_clients(self) -> list[Any]:
        keys = _judge_keys()
        if not keys:
            raise RuntimeError(
                "No judge API key. Set EVAL_JUDGE_API_KEY or GEMINI_API_KEY (backend/.env works)."
            )
        from openai import AsyncOpenAI

        base_url = os.environ.get("EVAL_JUDGE_BASE_URL", GEMINI_BASE_URL)
        return [AsyncOpenAI(api_key=k, base_url=base_url, max_retries=0) for k in keys]

    def _client(self) -> Any:
        c = self._clients[self._next % len(self._clients)]
        self._next += 1
        return c

    def _cache_path(self, item: GoldItem, answer: str) -> Path:
        payload = json.dumps(
            [self.model, PROMPT_VERSION, JUDGE_SYSTEM_PROMPT, build_user_prompt(item, answer)],
            ensure_ascii=False,
        )
        return self.cache_dir / f"{hashlib.sha256(payload.encode()).hexdigest()}.json"

    async def grade(self, item: GoldItem, answer: str) -> Verdict:
        path = self._cache_path(item, answer)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Verdict(data["label"], LABEL_SCORES[data["label"]], data.get("reason", ""), cached=True)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item, answer)},
        ]
        last_err: Exception | None = None
        async with self._sem:
            for attempt in range(self.max_retries):
                try:
                    resp = await self._client().chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=300,
                    )
                    label, reason = parse_verdict(resp.choices[0].message.content or "")
                    path.write_text(
                        json.dumps({"label": label, "reason": reason, "model": self.model}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    return Verdict(label, LABEL_SCORES[label], reason)
                except Exception as e:  # rate limits, transient network, unparseable output
                    last_err = e
                    wait = 2 ** attempt
                    logger.warning(f"Judge call failed for {item.id} (attempt {attempt + 1}): {e}; retrying in {wait}s")
                    await asyncio.sleep(wait)
        raise RuntimeError(f"Judge failed for {item.id}: {last_err}")
