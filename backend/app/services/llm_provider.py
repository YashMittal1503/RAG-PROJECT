"""
Resilient Cross-Provider LLM Orchestration Service.

Provides automated multi-tier failover across distinct cloud AI providers:
- Tier 1 (Primary): Groq (sub-second streaming via LPUs)
- Tier 2 (Secondary): Google Gemini (generous free quotas, massive context)
- Tier 3 (Tertiary): OpenRouter (universal fail-safe routing to Llama, DeepSeek, Mistral)

Standardizes non-streaming and streaming calls across providers with OpenTelemetry / Logfire tracing.
"""

from dataclasses import dataclass
import logging
from typing import Any, AsyncGenerator, List, Optional

from groq import AsyncGroq, APIError as GroqAPIError, APIStatusError as GroqAPIStatusError, RateLimitError as GroqRateLimitError
import httpx
import logfire
from openai import AsyncOpenAI, APIError as OpenAIAPIError, APIStatusError as OpenAIAPIStatusError, RateLimitError as OpenAIRateLimitError

from app.config import settings

logger = logging.getLogger(__name__)

# Provider client singletons
_groq_client: Optional[AsyncGroq] = None
_gemini_client: Optional[AsyncOpenAI] = None
_openrouter_client: Optional[AsyncOpenAI] = None


def get_groq_client() -> AsyncGroq:
    """Get or create the async Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


def get_gemini_client() -> Optional[AsyncOpenAI]:
    """
    Get or create the async Gemini client using Google's OpenAI-compatible endpoint.
    Returns None if GEMINI_API_KEY is not configured.
    """
    global _gemini_client
    if not settings.gemini_api_key:
        return None
    if _gemini_client is None:
        _gemini_client = AsyncOpenAI(
            api_key=settings.gemini_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return _gemini_client


def get_openrouter_client() -> Optional[AsyncOpenAI]:
    """
    Get or create the async OpenRouter client using its OpenAI-compatible endpoint.
    Returns None if OPENROUTER_API_KEY is not configured.
    """
    global _openrouter_client
    if not settings.openrouter_api_key:
        return None
    if _openrouter_client is None:
        _openrouter_client = AsyncOpenAI(
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/YashMittal1503/RAG-PROJECT",
                "X-Title": "RAG-Document-QA",
            },
        )
    return _openrouter_client


@dataclass
class LLMTarget:
    """Represents a specific provider, model, and client target."""
    provider: str  # "groq" | "gemini" | "openrouter"
    model: str
    client: Any


def get_generation_targets() -> List[LLMTarget]:
    """
    Build the ordered fallback chain for answer generation:
    1. Groq configured model (e.g. openai/gpt-oss-120b) + Groq alternatives
    2. Google Gemini (e.g. gemini-1.5-flash)
    3. OpenRouter (e.g. meta-llama/llama-3.3-70b-instruct)
    """
    targets: List[LLMTarget] = []
    groq = get_groq_client()

    # Tier 1: Groq models
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
            targets.append(LLMTarget(provider="groq", model=m, client=groq))

    # Tier 2: Google Gemini (if configured)
    gemini = get_gemini_client()
    if gemini:
        targets.append(LLMTarget(provider="gemini", model=settings.gemini_model, client=gemini))
        if settings.gemini_model != "gemini-2.0-flash":
            targets.append(LLMTarget(provider="gemini", model="gemini-2.0-flash", client=gemini))

    # Tier 3: OpenRouter (if configured)
    openrouter = get_openrouter_client()
    if openrouter:
        targets.append(LLMTarget(provider="openrouter", model=settings.openrouter_model, client=openrouter))

    return targets


def get_fast_targets() -> List[LLMTarget]:
    """
    Build the ordered fallback chain for fast utility tasks (intent, rewrite, SQL generation):
    Prioritizes low-latency, zero-preamble models.
    """
    targets: List[LLMTarget] = []
    groq = get_groq_client()

    # Tier 1: Groq fast models
    fast_groq = [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "groq/compound-mini",
    ]
    for m in fast_groq:
        targets.append(LLMTarget(provider="groq", model=m, client=groq))

    # Tier 2: Gemini Flash
    gemini = get_gemini_client()
    if gemini:
        targets.append(LLMTarget(provider="gemini", model="gemini-1.5-flash", client=gemini))

    # Tier 3: OpenRouter
    openrouter = get_openrouter_client()
    if openrouter:
        targets.append(LLMTarget(provider="openrouter", model=settings.openrouter_model, client=openrouter))

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
    Execute a non-streaming LLM call with automated cross-provider fallback.
    Tries each target in the multi-provider chain.
    """
    targets = get_fast_targets() if fast else get_generation_targets()
    last_err = None

    for i, target in enumerate(targets):
        try:
            logger.info(f"Invoking LLM [{target.provider.upper()}] with model: {target.model}")
            with logfire.span(
                "llm.call",
                provider=target.provider,
                model=target.model,
                attempt=i + 1,
                fast=fast,
                max_tokens=max_tokens,
            ) as span:
                response = await target.client.chat.completions.create(
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
                return response

        except Exception as err:
            last_err = err
            if is_recoverable_error(err):
                logger.warning(
                    f"Provider [{target.provider.upper()}] model '{target.model}' failed "
                    f"({type(err).__name__}: {err}). Auto-switching to next target in fallback chain..."
                )
                continue
            else:
                logger.error(f"Unrecoverable error on [{target.provider.upper()}] '{target.model}': {err}")
                raise

    logger.error("All fallback targets across all providers exhausted for non-streaming call.")
    if last_err:
        raise last_err
    raise RuntimeError("All LLM providers in fallback chain failed.")


async def stream_llm_with_cross_provider_fallback(
    messages: list[dict],
    max_tokens: int,
    temperature: float = 0.1,
    fast: bool = False,
) -> AsyncGenerator[str, None]:
    """
    Execute a streaming LLM call with automated cross-provider fallback.
    If a provider fails or hits rate limits before the stream yields,
    it automatically catches the error and switches to the next available provider.
    """
    targets = get_fast_targets() if fast else get_generation_targets()
    last_err = None

    for i, target in enumerate(targets):
        started = False
        try:
            logger.info(f"Starting LLM stream [{target.provider.upper()}] with model: {target.model}")
            with logfire.span(
                "llm.stream",
                provider=target.provider,
                model=target.model,
                attempt=i + 1,
                fast=fast,
                max_tokens=max_tokens,
            ) as span:
                stream = await target.client.chat.completions.create(
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
                return  # Stream completed successfully

        except Exception as err:
            last_err = err
            # If the stream failed before emitting tokens, we can safely fail over to the next provider
            if is_recoverable_error(err) and not started:
                logger.warning(
                    f"Provider [{target.provider.upper()}] stream failed before start "
                    f"({type(err).__name__}: {err}). Auto-switching to next target in fallback chain..."
                )
                continue
            else:
                logger.error(f"Error during stream on [{target.provider.upper()}] '{target.model}': {err}")
                raise

    logger.error("All fallback targets across all providers exhausted for streaming.")
    if last_err:
        raise last_err
    raise RuntimeError("All LLM providers in fallback chain failed.")
