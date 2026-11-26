"""
Resilient Multi-Key & Cross-Provider LLM Orchestration Service.

Features:
- Multi-tier provider failover: Groq -> NVIDIA NIM -> Mistral -> Google Gemini -> OpenRouter
- Multi-key rotation per tier: Supports at least 3 API keys per provider tier
- Round-robin load distribution across healthy keys
- 429 / TPM / RPM rate-limit quarantine with automatic cooldown, per (key, model): free tiers
  such as Groq limit each model separately, so a limit on one model doesn't block the others
- Models that no longer exist (404 / decommissioned) are skipped without quarantining the key
- Keys that are not cooling down are tried across every tier before any cooling key is retried
- Automatic intra-tier key rotation before escalating across providers
- Standardized streaming and non-streaming responses with Logfire telemetry and key masking
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import re
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


# ── Provider quirks and error classification (ported from the llm-key-pool router) ────────────────

# max_request_tokens: largest request one call may carry (Groq free tier: 8K tokens/minute per model, so a
# bigger request can never succeed and must not cool keys). daily_reset: when daily quotas reset.
PROVIDER_LIMITS: dict = {
    "groq": {"max_request_tokens": 8000},
    "gemini": {"daily_reset": "America/Los_Angeles"},   # Gemini daily quotas reset at midnight Pacific
    "openrouter": {"daily_reset": "UTC"},
}
DAILY_FALLBACK_SECONDS = 3600.0   # daily limit, no reset time known: look again in an hour
DAILY_MIN_SECONDS = 60.0
RATE_FALLBACK_SECONDS = 60.0      # per-minute limit, no hint given
LAST_RESORT_MAX_WAIT = 120.0      # a key cooling longer than this is not worth a "last resort" call

_TRUE = {"1", "true", "yes", "on"}
# "(?<![\w/.-])tpm(?![\w/-])", not "\btpm\b": a hyphen is a word boundary, so "\brpm\b" matches ".../rpm-docs".
_RATE_RE = re.compile(r"rate[ _-]?limit|too many requests|per[ _-]?minute|(?<![\w/.-])[rt]pm(?![\w/-])")
_DAILY_RE = re.compile(r"per[ _-]?day|(?<![\w/.-])[rt]pd(?![\w/-])|daily")
_MINUTE_RE = re.compile(r"per[ _-]?minute|(?<![\w/.-])[rt]pm(?![\w/-])")
# Gemini says "exceeded your current quota" for per-minute *and* per-day limits; the retry hint tells them apart.
_QUOTA_TERMS = ("quota exceeded", "resource_exhausted", "exceeded your current quota")
_TOO_LARGE_TERMS = ("request too large", "too large", "context length", "context_length_exceeded", "maximum context")
_LIMIT_RE = re.compile(r"\blimit[:\s]+(\d{3,})", re.IGNORECASE)
_ACCOUNT_RE = re.compile(r"organization\s+[`'\"]?(org_[A-Za-z0-9]+)")
_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")
_TRY_AGAIN_RE = re.compile(r"(?:try again in|retry after|retry in|please retry in)\s+([\d\.hms]+)", re.IGNORECASE)
_RETRY_DELAY_RE = re.compile(r"retrydelay['\"]?\s*[:=]\s*['\"]?([\d\.]+)s", re.IGNORECASE)


def _shared_quota_setting(provider: str) -> bool:
    """<PROVIDER>_SHARED_QUOTA=true from the environment or .env: all of that provider's keys share one quota."""
    name = f"{provider}_shared_quota"
    value = os.environ.get(name.upper()) or (settings.model_extra or {}).get(name) or ""
    return str(value).strip().lower() in _TRUE


def _status(err: BaseException) -> Optional[int]:
    code = getattr(err, "status_code", None)
    if code is None:
        code = getattr(getattr(err, "response", None), "status_code", None)
    return code if isinstance(code, int) else None


def parse_duration(text: str) -> Optional[float]:
    """Seconds in "1h2m3.5s" / "7m4.2s" / "500ms" / "31" (plain seconds)."""
    text = text.strip().lower().rstrip(".")
    try:
        return float(text)
    except ValueError:
        pass
    m = _DURATION_RE.match(text)
    if not m or not any(m.groups()):
        return None
    h, mi, sec, ms = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mi * 60 + sec + ms / 1000


def retry_after_seconds(err: BaseException) -> Optional[float]:
    """Wait time the provider gave: Retry-After / x-ratelimit-reset-* headers (smallest), else the
    "try again in 7m4s" / retryDelay text in the message. None when nothing is given."""
    headers = getattr(getattr(err, "response", None), "headers", None) or {}
    found = []
    for name in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        try:
            value = headers.get(name)
        except Exception:
            value = None
        if value:
            seconds = parse_duration(str(value))
            if seconds is not None:
                found.append(seconds)
    if found:
        return min(found)
    msg = str(err)
    for rx in (_TRY_AGAIN_RE, _RETRY_DELAY_RE):
        m = rx.search(msg)
        if m:
            seconds = parse_duration(m.group(1))
            if seconds is not None:
                return seconds
    return None


def account_of(err: BaseException) -> Optional[str]:
    """The provider account an error names, e.g. Groq's "in organization `org_01...`"."""
    m = _ACCOUNT_RE.search(str(err))
    return m.group(1) if m else None


def token_limit_of(err: BaseException) -> Optional[int]:
    """The token ceiling a too-large error names, e.g. Groq's "(TPM): Limit 8000, Requested 14210"."""
    m = _LIMIT_RE.search(str(err))
    return int(m.group(1)) if m else None


def classify_limit_error(err: BaseException) -> Optional[str]:
    """ "too_large" | "daily" | "rate" for quota-style failures, else None (handled by the older checks).

    413 is checked before 429: Groq's 413 says "Request too large ... tokens per minute (TPM)" and no key can
    serve that request, so it must skip the model rather than cool every key down. rpm/tpm/rpd/tpd match as
    whole words only (a 404 linking to ".../rpm-docs" is not a rate limit).
    """
    status = _status(err)
    msg = str(err).lower()
    if status == 413 or (status != 429 and any(t in msg for t in _TOO_LARGE_TERMS)):
        return "too_large"
    names = {c.__name__ for c in type(err).__mro__}
    if status == 429 or "RateLimitError" in names or _RATE_RE.search(msg) or any(t in msg for t in _QUOTA_TERMS):
        if _DAILY_RE.search(msg):
            return "daily"
        if _MINUTE_RE.search(msg):
            return "rate"
        if any(t in msg for t in _QUOTA_TERMS):
            wait = retry_after_seconds(err)
            return "rate" if wait is not None and wait <= 120 else "daily"
        return "rate"
    return None


def seconds_until_reset(tz_name: str, now: datetime | None = None) -> float:
    """Seconds until the next midnight in `tz_name` ("UTC", "America/Los_Angeles")."""
    try:
        from zoneinfo import ZoneInfo

        tz = timezone.utc if tz_name.upper() == "UTC" else ZoneInfo(tz_name)
    except Exception:  # no tz database on this machine: Pacific standard time is the only non-UTC reset
        tz = timezone(timedelta(hours=-8)) if "los_angeles" in tz_name.lower() else timezone.utc
    local = (now or datetime.now(timezone.utc)).astimezone(tz)
    midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight - local).total_seconds()


def estimate_tokens(messages: Any) -> int:
    """Rough prompt size (~4 characters per token)."""
    return len(json.dumps(messages, default=str, ensure_ascii=False)) // 4


@dataclass
class KeySlot:
    """Represents a single API key, its client, and its health status."""
    key: str
    masked_key: str
    client: Any  # AsyncGroq or AsyncOpenAI
    cooldown_until: float = 0.0          # key-wide cooldown
    failure_count: int = 0
    model_cooldowns: dict = None         # model -> cooldown_until, for per-model limits
    account: Optional[str] = None        # provider account the key belongs to (Groq names its org in 429s)

    def __post_init__(self):
        if self.model_cooldowns is None:
            self.model_cooldowns = {}

    def ready_at(self, model: Optional[str] = None) -> float:
        """When this key can be used again (for `model`, if given)."""
        return max(self.cooldown_until, self.model_cooldowns.get(model, 0.0) if model else 0.0)


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
        spec = PROVIDER_LIMITS.get(provider, {})
        self.shared_quota: bool = _shared_quota_setting(provider)   # all keys draw on ONE account's quota
        self.daily_reset: Optional[str] = spec.get("daily_reset")   # time zone whose midnight resets daily quotas
        self.max_request_tokens: Optional[int] = spec.get("max_request_tokens")
        self.learned_limits: dict = {}   # model -> token ceiling named by a 413 ("Limit 8000")

    def request_limit(self, model: str) -> Optional[int]:
        """Largest request (prompt + max output tokens) `model` can take on this provider, if known."""
        return self.learned_limits.get(model) or self.max_request_tokens

    def cool_down_quota(self, key: str, model: Optional[str], seconds: float, reason: str, err: Exception | None = None):
        """Cool down a quota hit: this key, plus every key on the same account.

        Keys share an account when `shared_quota` is set, or when the provider names the same account for them
        (Groq: "in organization `org_...`"). Six keys on one Groq org share one 200K tokens/day budget, so
        finding that out by spending one request per key just burns six 429s every minute.
        """
        hit = next((sl for sl in self.slots if sl.key == key), None)
        if hit is None:
            return
        account = account_of(err) if err is not None else None
        if account:
            hit.account = account
        same = [sl for sl in self.slots
                if sl is hit or self.shared_quota or (hit.account and sl.account == hit.account)]
        now = time.time()
        for sl in same:
            until = now + seconds
            if model:
                sl.model_cooldowns[model] = max(sl.model_cooldowns.get(model, 0.0), until)
            else:
                sl.cooldown_until = max(sl.cooldown_until, until)
        hit.failure_count += 1
        logger.warning(
            f"Provider [{self.provider.upper()}] key '{hit.masked_key}' {reason}{f' on {model}' if model else ''}. "
            f"Cooling {len(same)} key(s) for {seconds:.0f}s."
        )

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

    def get_rotation_candidates(self, model: Optional[str] = None) -> List[KeySlot]:
        """
        Return all key slots ordered by round-robin priority.
        Active (non-cooling) keys come first; cooldowns for `model` count when it's given.
        If all keys are on cooldown, returns them ordered by soonest recovery.
        """
        if not self.slots:
            return []

        now = time.time()
        n = len(self.slots)

        # Start from current round-robin index
        ordered = [self.slots[(self._index + i) % n] for i in range(n)]
        self._index = (self._index + 1) % n

        active = [s for s in ordered if s.ready_at(model) <= now]
        cooling = [s for s in ordered if s.ready_at(model) > now]
        cooling.sort(key=lambda s: s.ready_at(model))

        return active + cooling

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 60.0, model: Optional[str] = None):
        """Quarantine a key that hit 429 / TPM limits: for `model` only when given, else for every model."""
        now = time.time()
        for slot in self.slots:
            if slot.key == key:
                slot.failure_count += 1
                if model:
                    slot.model_cooldowns[model] = now + cooldown_seconds
                else:
                    slot.cooldown_until = now + cooldown_seconds
                logger.warning(
                    f"Provider [{self.provider.upper()}] key '{slot.masked_key}' hit rate limit (429/TPM)"
                    f"{f' on {model}' if model else ''}. "
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
    return AsyncGroq(api_key=key, max_retries=0)


def _make_gemini_client(key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_retries=0,
    )


def _make_openrouter_client(key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        max_retries=0,
        default_headers={
            "HTTP-Referer": "https://github.com/YashMittal1503/RAG-PROJECT",
            "X-Title": "RAG-Document-QA",
        },
    )


def _make_mistral_client(key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=key,
        base_url="https://api.mistral.ai/v1",
        max_retries=0,
    )


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _make_nvidia_client(key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=key, base_url=NVIDIA_BASE_URL, max_retries=0)


# Provider singleton pools and legacy test client references
_groq_pool: Optional[ProviderKeyPool] = None
_gemini_pool: Optional[ProviderKeyPool] = None
_openrouter_pool: Optional[ProviderKeyPool] = None
_mistral_pool: Optional[ProviderKeyPool] = None
_nvidia_pool: Optional[ProviderKeyPool] = None
_gemini_client: Optional[AsyncOpenAI] = None
_openrouter_client: Optional[AsyncOpenAI] = None
_mistral_client: Optional[AsyncOpenAI] = None


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


def get_mistral_pool() -> ProviderKeyPool:
    global _mistral_pool, _mistral_client
    if _mistral_client is not None:
        return ProviderKeyPool("mistral", ["mock-mistral-key"], lambda k: _mistral_client)
    keys = settings.get_mistral_keys()
    if _mistral_pool is None or [s.key for s in _mistral_pool.slots] != keys:
        _mistral_pool = ProviderKeyPool("mistral", keys, _make_mistral_client)
    return _mistral_pool


def get_nvidia_pool() -> ProviderKeyPool:
    global _nvidia_pool
    keys = settings.get_nvidia_keys()
    if _nvidia_pool is None or [s.key for s in _nvidia_pool.slots] != keys:
        _nvidia_pool = ProviderKeyPool("nvidia", keys, _make_nvidia_client)
    return _nvidia_pool


def reset_pools():
    """Reset singleton pools (primarily used in tests)."""
    global _groq_pool, _gemini_pool, _openrouter_pool, _mistral_pool, _nvidia_pool
    global _gemini_client, _openrouter_client, _mistral_client
    _nvidia_pool = None
    _groq_pool = None
    _gemini_pool = None
    _openrouter_pool = None
    _mistral_pool = None
    _gemini_client = None
    _openrouter_client = None
    _mistral_client = None



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


# Groq free-plan limits are per model (30 RPM, 1,000 RPD, 8K TPM, 200K TPD each), so every
# model here adds capacity. groq/compound-mini was shut down on 2026-09-21; gpt-oss-20b was
# removed earlier because it stalled streams for ~6s.
GROQ_GENERATION_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
GROQ_FAST_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
GROQ_EXCLUDED_MODELS = ("groq/compound", "groq/compound-mini", "openai/gpt-oss-20b")


def _nvidia_targets(pool: ProviderKeyPool, fast: bool) -> List[LLMTarget]:
    """NVIDIA NIM models (configured model first, then the backup), if keys are configured."""
    if not pool.is_available():
        return []
    first = settings.nvidia_fast_model if fast else settings.nvidia_model
    models = [m for m in dict.fromkeys([first, settings.nvidia_model, settings.nvidia_backup_model]) if m]
    return [LLMTarget(provider="nvidia", model=m, client=pool.get_primary_client(), pool=pool) for m in models]


def get_generation_targets() -> List[LLMTarget]:
    """
    Build the ordered fallback chain for answer generation:
    1. Groq configured model + alternatives (LPUs; limits are per model)
    2. NVIDIA NIM (free, ~40 RPM, large open models)
    3. Mistral AI (e.g. ministral-3b-2512)
    4. Google Gemini (e.g. gemini-flash-latest)
    5. OpenRouter (e.g. meta-llama/llama-3.3-70b-instruct)
    """
    targets: List[LLMTarget] = []
    groq_pool = get_groq_pool()
    nvidia_pool = get_nvidia_pool()
    mistral_pool = get_mistral_pool()
    gemini_pool = get_gemini_pool()
    openrouter_pool = get_openrouter_pool()

    # Tier 1: Groq models
    if groq_pool.is_available():
        candidates = [settings.groq_model, *GROQ_GENERATION_MODELS]
        # Exclude known non-chat or malformed models
        groq_models = [m for m in candidates if m and m not in GROQ_EXCLUDED_MODELS]
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

    # Tier 2: NVIDIA NIM (if configured)
    targets += _nvidia_targets(nvidia_pool, fast=False)

    # Tier 3: Mistral AI (if configured)
    if mistral_pool.is_available():
        targets.append(LLMTarget(
            provider="mistral",
            model=settings.mistral_model or "ministral-3b-2512",
            client=mistral_pool.get_primary_client(),
            pool=mistral_pool,
        ))

    # Tier 3: Google Gemini (if configured)
    if gemini_pool.is_available():
        primary_model = settings.gemini_model
        if primary_model in ("gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"):
            primary_model = "gemini-flash-latest"
        targets.append(LLMTarget(
            provider="gemini",
            model=primary_model,
            client=gemini_pool.get_primary_client(),
            pool=gemini_pool,
        ))
        if primary_model != "gemini-3.6-flash":
            targets.append(LLMTarget(
                provider="gemini",
                model="gemini-3.6-flash",
                client=gemini_pool.get_primary_client(),
                pool=gemini_pool,
            ))

    # Tier 4: OpenRouter (if configured)
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
    nvidia_pool = get_nvidia_pool()
    mistral_pool = get_mistral_pool()
    gemini_pool = get_gemini_pool()
    openrouter_pool = get_openrouter_pool()

    # Tier 1: Groq fast models
    if groq_pool.is_available():
        for m in GROQ_FAST_MODELS:
            targets.append(LLMTarget(
                provider="groq",
                model=m,
                client=groq_pool.get_primary_client(),
                pool=groq_pool,
            ))

    # Tier 2: NVIDIA NIM
    targets += _nvidia_targets(nvidia_pool, fast=True)

    # Tier 3: Mistral AI fast models
    if mistral_pool.is_available():
        targets.append(LLMTarget(
            provider="mistral",
            model=settings.mistral_model or "ministral-3b-2512",
            client=mistral_pool.get_primary_client(),
            pool=mistral_pool,
        ))

    # Tier 3: Gemini Flash
    if gemini_pool.is_available():
        fast_model = settings.gemini_model
        if fast_model in ("gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"):
            fast_model = "gemini-flash-latest"
        targets.append(LLMTarget(
            provider="gemini",
            model=fast_model,
            client=gemini_pool.get_primary_client(),
            pool=gemini_pool,
        ))

    # Tier 4: OpenRouter
    if openrouter_pool.is_available():
        targets.append(LLMTarget(
            provider="openrouter",
            model=settings.openrouter_model,
            client=openrouter_pool.get_primary_client(),
            pool=openrouter_pool,
        ))

    return targets



def is_model_unavailable(err: Exception) -> bool:
    """The model doesn't exist (any more) on this provider: skip it, but the key is fine."""
    if getattr(err, "status_code", None) in (404, 410):   # 410 Gone: model reached end of life
        return True
    msg = str(err).lower()
    return any(t in msg for t in ("model_not_found", "decommissioned", "does not exist", "no longer supported", "unknown model", "end of life"))


def _attempt_plan(targets: List[LLMTarget]) -> List[tuple]:
    """
    Ordered (target_index, target, slot) attempts: every ready key of every tier first, in
    tier order, then keys still cooling down (soonest recovery first) as a last resort.
    Trying a cooling key before a healthy key on the next tier only burns another 429.
    """
    ready, cooling = [], []
    now = time.time()
    for i, target in enumerate(targets):
        if target.pool and target.pool.is_available():
            for slot in target.pool.get_rotation_candidates(target.model):
                (ready if slot.ready_at(target.model) <= now else cooling).append((i, target, slot))
        else:
            key = getattr(target, "key", "")
            ready.append((i, target, KeySlot(key=key, masked_key=mask_key(key), client=target.client)))
    cooling.sort(key=lambda a: a[2].ready_at(a[1].model))
    return ready + cooling


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


# Reasoning models spend completion tokens thinking before they answer. A caller asking for 60 tokens
# (a routing label, a rewritten query) gets an empty reply from them, which callers then read as
# "no decision" and silently fall back. Give these models enough room; short-reply models are unaffected.
REASONING_MODEL_MARKERS = ("gpt-oss", "nemotron", "minimax", "deepseek-r1", "-thinking", "qwq")
REASONING_MIN_TOKENS = 1024


def effective_max_tokens(model: str, max_tokens: int) -> int:
    """max_tokens to send to `model`: raised to REASONING_MIN_TOKENS for reasoning models."""
    m = model.lower()
    if any(k in m for k in REASONING_MODEL_MARKERS):
        return max(max_tokens, REASONING_MIN_TOKENS)
    return max_tokens


def _too_big(target: LLMTarget, est_tokens: int) -> bool:
    """The request can't fit this model on this provider (e.g. over Groq's 8K tokens/minute)."""
    limit = target.pool.request_limit(target.model) if target.pool else None
    return bool(limit and est_tokens > limit)


def _skip_attempt(i: int, target: LLMTarget, slot: KeySlot, skip: set, ready_at_plan: set) -> bool:
    """Whether to pass over this planned attempt without a call.

    A key that was ready when the plan was made but got cooled since (a shared-quota 429 earlier in this
    same call) is skipped; so is a "last resort" cooling key that won't recover for minutes.
    """
    if i in skip:
        return True
    remaining = slot.ready_at(target.model) - time.time()
    if remaining <= 0:
        return False
    return (i, slot.key) in ready_at_plan or remaining > LAST_RESORT_MAX_WAIT


def _plan_with_ready(targets: List[LLMTarget]) -> tuple:
    plan = _attempt_plan(targets)
    now = time.time()
    ready = {(i, sl.key) for i, t, sl in plan if sl.ready_at(t.model) <= now}
    return plan, ready


def _handle_failure(i: int, target: LLMTarget, slot: KeySlot, err: Exception, skip: set) -> bool:
    """React to a failed attempt. True: try the next attempt. False: unrecoverable, re-raise."""
    kind = classify_limit_error(err)
    pool = target.pool
    where = f"[{target.provider.upper()}] '{target.model}' (key {slot.masked_key})"
    if kind == "too_large":
        skip.add(i)   # no key can serve this request on this model; remember the ceiling it named
        limit = token_limit_of(err)
        if pool and limit and limit < (pool.request_limit(target.model) or 10**9):
            pool.learned_limits[target.model] = limit
        logger.warning(f"Request too large for {where}{f' (limit {limit})' if limit else ''}. Skipping that model.")
        return True
    if is_model_unavailable(err):
        skip.add(i)
        logger.warning(f"Model unavailable on {where} ({err}). Skipping it.")
        return True
    if kind in ("rate", "daily"):
        wait = retry_after_seconds(err)
        if kind == "daily":
            if pool and pool.daily_reset:
                seconds = max(wait or 0.0, seconds_until_reset(pool.daily_reset), DAILY_MIN_SECONDS)
            else:
                seconds = max(wait + 1, DAILY_MIN_SECONDS) if wait is not None else DAILY_FALLBACK_SECONDS
        else:
            seconds = wait + 1 if wait is not None else RATE_FALLBACK_SECONDS
        if pool:
            pool.cool_down_quota(slot.key, target.model, seconds,
                                 "hit its daily limit" if kind == "daily" else "hit a rate limit", err)
        return True
    if is_recoverable_error(err):
        if pool:
            pool.mark_rate_limited(slot.key, model=target.model)
        logger.warning(f"{where} failed ({type(err).__name__}: {err}). Rotating to next key/model...")
        return True
    logger.error(f"Unrecoverable error on {where}: {err}")
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
    est = estimate_tokens(messages)
    # target indexes to pass over: model gone, or the request can't fit it (skipped without a call)
    skip: set = {i for i, t in enumerate(targets) if _too_big(t, est + effective_max_tokens(t.model, max_tokens))}
    plan, ready_at_plan = _plan_with_ready(targets)

    for i, target, slot in plan:
            if _skip_attempt(i, target, slot, skip, ready_at_plan):
                continue
            try:
                logger.info(
                    f"Invoking LLM [{target.provider.upper()}] (Key: {slot.masked_key}) "
                    f"with model: {target.model}"
                )
                with logfire.span(
                    "🤖 LLM Call | {provider}/{model}",
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
                        max_tokens=effective_max_tokens(target.model, max_tokens),
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
                if _handle_failure(i, target, slot, err, skip):
                    continue
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
    est = estimate_tokens(messages)
    skip: set = {i for i, t in enumerate(targets) if _too_big(t, est + effective_max_tokens(t.model, max_tokens))}
    plan, ready_at_plan = _plan_with_ready(targets)

    for i, target, slot in plan:
            if _skip_attempt(i, target, slot, skip, ready_at_plan):
                continue
            started = False
            try:
                logger.info(
                    f"Starting LLM stream [{target.provider.upper()}] (Key: {slot.masked_key}) "
                    f"with model: {target.model}"
                )
                with logfire.span(
                    "🤖 LLM Stream | {provider}/{model}",
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
                        max_tokens=effective_max_tokens(target.model, max_tokens),
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
                # Once tokens have reached the caller, switching models would splice two answers: re-raise.
                if not started and _handle_failure(i, target, slot, err, skip):
                    continue
                logger.error(f"Error during stream on [{target.provider.upper()}] '{target.model}' (key: {slot.masked_key}): {err}")
                raise

    logger.error("All fallback targets across all providers and keys exhausted for streaming.")
    if last_err:
        raise last_err
    raise RuntimeError("All LLM providers and keys in fallback chain failed.")
