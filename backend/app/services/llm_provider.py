"""
Resilient Multi-Key & Cross-Provider LLM Orchestration Service.

Features:
- Multi-tier provider failover: Groq (Tier 1) -> Google Gemini (Tier 2) -> OpenRouter (Tier 3)
- Multi-key rotation per tier: Supports at least 3 API keys per provider tier
- Round-robin load distribution across healthy keys
- 429 / TPM / RPM rate-limit quarantine with automatic cooldown
- Automatic intra-tier key rotation before escalating across providers
- Standardized streaming and non-streaming responses with Logfire telemetry and key masking
"""

from dataclasses import dataclass
import logging
import time
from typing import Any, AsyncGenerator, Callable, List, Optional

from groq import AsyncGroq, APIError as GroqAPIError, APIStatusError as GroqAPIStatusError, RateLimitError as GroqRateLimitError
import httpx
import logfire
from openai import AsyncOpenAI, APIError as OpenAIAPIError, APIStatusError as OpenAIAPIStatusError, RateLimitError as OpenAIRateLimitError

from app.config import settings

logger = logging.getLogger(__name__)


def mask_key(key: str) -> str:
    """Mask sensitive API key for safe logging and metrics tracing (e.g. gsk_...3a4f)."""
    if not key:
        return "<none>"
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


@dataclass
class KeySlot:
    """Represents a single API key, its client, and its health status."""
    key: str
    masked_key: str
    client: Any  # AsyncGroq or AsyncOpenAI
    cooldown_until: float = 0.0
    failure_count: int = 0


class ProviderKeyPool:
    """
    Manages a pool of API keys for a single provider tier.
    Provides round-robin rotation, health checks, and 429 quarantine cooldowns.
    """

    def __init__(self, provider: str, keys: List[str], client_factory: Callable[[str], Any]):
        self.provider = provider
        self.client_factory = client_factory
        self.slots: List[KeySlot] = [
            KeySlot(
                key=k,
                masked_key=mask_key(k),
                client=client_factory(k),
            )
            for k in keys if k
        ]
        self._index: int = 0

    @property
    def key_count(self) -> int:
        return len(self.slots)

    def is_available(self) -> bool:
        return len(self.slots) > 0

    def get_primary_client(self) -> Optional[Any]:
        """Return the primary (first) client or None if no keys configured."""
        if not self.slots:
            return None
        return self.slots[0].client

    def get_rotation_candidates(self) -> List[KeySlot]:
        """
        Return all key slots ordered by round-robin priority.
        Active (non-cooling) keys come first.
        If all keys are on cooldown, returns them ordered by soonest recovery.
        """
        if not self.slots:
            return []

        now = time.time()
        n = len(self.slots)

        # Start from current round-robin index
        ordered = [self.slots[(self._index + i) % n] for i in range(n)]
        self._index = (self._index + 1) % n

        active = [s for s in ordered if s.cooldown_until <= now]
        cooling = [s for s in ordered if s.cooldown_until > now]
        cooling.sort(key=lambda s: s.cooldown_until)

        return active + cooling

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 60.0):
        """Quarantine a key that encountered 429 or TPM limits."""
        now = time.time()
        for slot in self.slots:
            if slot.key == key:
                slot.failure_count += 1
                slot.cooldown_until = now + cooldown_seconds
                logger.warning(
                    f"Provider [{self.provider.upper()}] key '{slot.masked_key}' hit rate limit (429/TPM). "
                    f"Quarantined for {cooldown_seconds:.0f}s (failure count: {slot.failure_count})."
                )
                break

    def mark_success(self, key: str):
        """Reset failure count upon successful execution."""
        for slot in self.slots:
            if slot.key == key:
                slot.failure_count = 0
                break


def _make_groq_client(key: str) -> AsyncGroq:
    return AsyncGroq(api_key=key)


def _make_gemini_client(key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def _make_openrouter_client(key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/YashMittal1503/RAG-PROJECT",
            "X-Title": "RAG-Document-QA",
        },
    )


# Provider singleton pools and legacy test client references
_groq_pool: Optional[ProviderKeyPool] = None
_gemini_pool: Optional[ProviderKeyPool] = None
_openrouter_pool: Optional[ProviderKeyPool] = None
_gemini_client: Optional[AsyncOpenAI] = None
_openrouter_client: Optional[AsyncOpenAI] = None


def get_groq_pool() -> ProviderKeyPool:
    global _groq_pool
    keys = settings.get_groq_keys()
    if _groq_pool is None or [s.key for s in _groq_pool.slots] != keys:
        _groq_pool = ProviderKeyPool("groq", keys, _make_groq_client)
    return _groq_pool


def get_gemini_pool() -> ProviderKeyPool:
    global _gemini_pool, _gemini_client
    if _gemini_client is not None:
        return ProviderKeyPool("gemini", ["mock-gemini-key"], lambda k: _gemini_client)
    keys = settings.get_gemini_keys()
    if _gemini_pool is None or [s.key for s in _gemini_pool.slots] != keys:
        _gemini_pool = ProviderKeyPool("gemini", keys, _make_gemini_client)
    return _gemini_pool


def get_openrouter_pool() -> ProviderKeyPool:
    global _openrouter_pool, _openrouter_client
    if _openrouter_client is not None:
        return ProviderKeyPool("openrouter", ["mock-openrouter-key"], lambda k: _openrouter_client)
    keys = settings.get_openrouter_keys()
    if _openrouter_pool is None or [s.key for s in _openrouter_pool.slots] != keys:
        _openrouter_pool = ProviderKeyPool("openrouter", keys, _make_openrouter_client)
    return _openrouter_pool


def reset_pools():
    """Reset singleton pools (primarily used in tests)."""
    global _groq_pool, _gemini_pool, _openrouter_pool, _gemini_client, _openrouter_client
    _groq_pool = None
    _gemini_pool = None
    _openrouter_pool = None
    _gemini_client = None
    _openrouter_client = None


def get_groq_client() -> AsyncGroq:
    """Get the primary or first active async Groq client."""
    client = get_groq_pool().get_primary_client()
    if client is not None:
        return client
    return _make_groq_client(settings.groq_api_key)


def get_gemini_client() -> Optional[AsyncOpenAI]:
    """Get the primary or first active async Gemini client."""
    return get_gemini_pool().get_primary_client()


def get_openrouter_client() -> Optional[AsyncOpenAI]:
    """Get the primary or first active async OpenRouter client."""
    return get_openrouter_pool().get_primary_client()


@dataclass
class LLMTarget:
    """Represents a specific provider, model, client, and optional key pool."""
    provider: str  # "groq" | "gemini" | "openrouter"
    model: str
    client: Any
    key: str = ""
    masked_key: str = ""
    pool: Optional[ProviderKeyPool] = None


def get_generation_targets() -> List[LLMTarget]:
    """
    Build the ordered fallback chain for answer generation:
    1. Groq configured model (e.g. openai/gpt-oss-120b) + Groq alternatives
    2. Google Gemini (e.g. gemini-1.5-flash)
    3. OpenRouter (e.g. meta-llama/llama-3.3-70b-instruct)
    """
    targets: List[LLMTarget] = []
    groq_pool = get_groq_pool()
    gemini_pool = get_gemini_pool()
    openrouter_pool = get_openrouter_pool()

    # Tier 1: Groq models
    if groq_pool.is_available():
        groq_models = [
            settings.groq_model,
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-20b",
            "groq/compound-mini",
        ]
        seen_groq = set()
        for m in groq_models:
            if m not in seen_groq:
                seen_groq.add(m)
                targets.append(LLMTarget(
                    provider="groq",
                    model=m,
                    client=groq_pool.get_primary_client(),
                    pool=groq_pool,
                ))

    # Tier 2: Google Gemini (if configured)
    if gemini_pool.is_available():
        targets.append(LLMTarget(
            provider="gemini",
            model=settings.gemini_model,
            client=gemini_pool.get_primary_client(),
            pool=gemini_pool,
        ))
        if settings.gemini_model != "gemini-2.0-flash":
            targets.append(LLMTarget(
                provider="gemini",
                model="gemini-2.0-flash",
                client=gemini_pool.get_primary_client(),
                pool=gemini_pool,
            ))

    # Tier 3: OpenRouter (if configured)
    if openrouter_pool.is_available():
        targets.append(LLMTarget(
            provider="openrouter",
            model=settings.openrouter_model,
            client=openrouter_pool.get_primary_client(),
            pool=openrouter_pool,
        ))

    return targets


def get_fast_targets() -> List[LLMTarget]:
    """
    Build the ordered fallback chain for fast utility tasks (intent, rewrite, SQL generation):
    Prioritizes low-latency, zero-preamble models.
    """
    targets: List[LLMTarget] = []
    groq_pool = get_groq_pool()
    gemini_pool = get_gemini_pool()
    openrouter_pool = get_openrouter_pool()

    # Tier 1: Groq fast models
    if groq_pool.is_available():
        fast_groq = [
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "groq/compound-mini",
        ]
        for m in fast_groq:
            targets.append(LLMTarget(
                provider="groq",
                model=m,
                client=groq_pool.get_primary_client(),
                pool=groq_pool,
            ))

    # Tier 2: Gemini Flash
    if gemini_pool.is_available():
        targets.append(LLMTarget(
            provider="gemini",
            model="gemini-1.5-flash",
            client=gemini_pool.get_primary_client(),
            pool=gemini_pool,
        ))

    # Tier 3: OpenRouter
    if openrouter_pool.is_available():
        targets.append(LLMTarget(
            provider="openrouter",
            model=settings.openrouter_model,
            client=openrouter_pool.get_primary_client(),
            pool=openrouter_pool,
        ))

    return targets


def is_recoverable_error(err: Exception) -> bool:
    """
    Determine if an error is transient, rate-limited, or provider-side recoverable.
    """
    status_code = getattr(err, "status_code", None)
    err_msg = str(err).lower()

    if isinstance(err, (
        GroqRateLimitError, GroqAPIStatusError, GroqAPIError,
        OpenAIRateLimitError, OpenAIAPIStatusError, OpenAIAPIError,
        httpx.TimeoutException, httpx.NetworkError,
    )):
        return True

    recoverable_terms = [
        "rate limit", "tokens per minute", "tpm", "requests per minute", "rpm",
        "quota", "capacity", "overloaded", "too large", "decommissioned",
        "timeout", "timed out", "connection", "service unavailable",
    ]
    if any(term in err_msg for term in recoverable_terms):
        return True

    if status_code in (408, 413, 429, 500, 502, 503, 504):
        return True

    return False


async def call_llm_with_cross_provider_fallback(
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.0,
    fast: bool = False,
) -> Any:
    """
    Execute a non-streaming LLM call with automated multi-key rotation and cross-provider fallback.
    Rotates through healthy keys within the current provider tier before failing over to the next tier.
    """
    targets = get_fast_targets() if fast else get_generation_targets()
    last_err = None

    for i, target in enumerate(targets):
        # Resolve key candidates: if a pool is attached, rotate through its keys
        if target.pool and target.pool.is_available():
            slots = target.pool.get_rotation_candidates()
        else:
            slots = [KeySlot(
                key=getattr(target, "key", ""),
                masked_key=mask_key(getattr(target, "key", "")),
                client=target.client,
            )]

        for slot in slots:
            try:
                logger.info(
                    f"Invoking LLM [{target.provider.upper()}] (Key: {slot.masked_key}) "
                    f"with model: {target.model}"
                )
                with logfire.span(
                    "llm.call",
                    provider=target.provider,
                    model=target.model,
                    key_id=slot.masked_key,
                    attempt=i + 1,
                    fast=fast,
                    max_tokens=max_tokens,
                ) as span:
                    response = await slot.client.chat.completions.create(
                        model=target.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                    usage = getattr(response, "usage", None)
                    if usage:
                        span.set_attribute("prompt_tokens", getattr(usage, "prompt_tokens", 0))
                        span.set_attribute("completion_tokens", getattr(usage, "completion_tokens", 0))
                        span.set_attribute("total_tokens", getattr(usage, "total_tokens", 0))

                    if target.pool:
                        target.pool.mark_success(slot.key)
                    return response

            except Exception as err:
                last_err = err
                if is_recoverable_error(err):
                    if target.pool:
                        target.pool.mark_rate_limited(slot.key)
                    logger.warning(
                        f"Provider [{target.provider.upper()}] key '{slot.masked_key}' failed on '{target.model}' "
                        f"({type(err).__name__}: {err}). Rotating to next key/model..."
                    )
                    continue
                else:
                    logger.error(
                        f"Unrecoverable error on [{target.provider.upper()}] '{target.model}' "
                        f"(key: {slot.masked_key}): {err}"
                    )
                    raise

    logger.error("All fallback targets across all providers and keys exhausted for non-streaming call.")
    if last_err:
        raise last_err
    raise RuntimeError("All LLM providers and keys in fallback chain failed.")


async def stream_llm_with_cross_provider_fallback(
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.1,
    fast: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Execute a streaming LLM call with automated multi-key rotation and cross-provider fallback.
    If a key fails before stream emission starts, it automatically catches the error,
    rotates to the next healthy key in the pool, and falls back across tiers if needed.
    """
    targets = get_fast_targets() if fast else get_generation_targets()
    last_err = None

    for i, target in enumerate(targets):
        if target.pool and target.pool.is_available():
            slots = target.pool.get_rotation_candidates()
        else:
            slots = [KeySlot(
                key=getattr(target, "key", ""),
                masked_key=mask_key(getattr(target, "key", "")),
                client=target.client,
            )]

        for slot in slots:
            started = False
            try:
                logger.info(
                    f"Starting LLM stream [{target.provider.upper()}] (Key: {slot.masked_key}) "
                    f"with model: {target.model}"
                )
                with logfire.span(
                    "llm.stream",
                    provider=target.provider,
                    model=target.model,
                    key_id=slot.masked_key,
                    attempt=i + 1,
                    fast=fast,
                    max_tokens=max_tokens,
                ) as span:
                    stream = await slot.client.chat.completions.create(
                        model=target.model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                    )
                    token_count = 0
                    async for chunk in stream:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            started = True
                            token_count += 1
                            yield content
                    span.set_attribute("streamed_tokens", token_count)
                    if target.pool:
                        target.pool.mark_success(slot.key)
                    return  # Stream completed successfully

            except Exception as err:
                last_err = err
                # If stream failed before emitting tokens, rotate to next key/model
                if is_recoverable_error(err) and not started:
                    if target.pool:
                        target.pool.mark_rate_limited(slot.key)
                    logger.warning(
                        f"Provider [{target.provider.upper()}] key '{slot.masked_key}' stream failed before start "
                        f"({type(err).__name__}: {err}). Rotating to next key/model..."
                    )
                    continue
                else:
                    logger.error(
                        f"Error during stream on [{target.provider.upper()}] '{target.model}' "
                        f"(key: {slot.masked_key}): {err}"
                    )
                    raise

    logger.error("All fallback targets across all providers and keys exhausted for streaming.")
    if last_err:
        raise last_err
    raise RuntimeError("All LLM providers and keys in fallback chain failed.")
