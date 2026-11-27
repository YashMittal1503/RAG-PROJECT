"""
Tests for the NVIDIA NIM tier, per-model cooldowns, skipping unavailable models, and the
"ready keys on every tier before any cooling key" attempt order.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import NotFoundError, RateLimitError

import app.services.llm_provider as lp
from app.services.llm_provider import (
    LLMTarget, ProviderKeyPool, _attempt_plan, call_llm_with_cross_provider_fallback,
    is_model_unavailable, stream_llm_with_cross_provider_fallback,
)


def _resp(text="ok"):
    msg = MagicMock(content=text, reasoning=None)
    return MagicMock(choices=[MagicMock(message=msg)], usage=None)


def _err(cls, status):
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


def _pool(provider, keys, client):
    return ProviderKeyPool(provider, keys, lambda k: client)


def test_rate_limit_on_one_model_leaves_the_others_usable():
    pool = _pool("groq", ["k1"], MagicMock())
    pool.mark_rate_limited("k1", cooldown_seconds=60, model="openai/gpt-oss-120b")
    slot = pool.slots[0]
    assert slot.ready_at("openai/gpt-oss-120b") > 0
    assert slot.ready_at("qwen/qwen3.8-27b") == 0          # separate per-model budget on Groq
    pool.mark_rate_limited("k1", cooldown_seconds=60)        # key-wide still works
    assert slot.ready_at("qwen/qwen3.8-27b") > 0


def test_attempt_plan_tries_ready_keys_on_every_tier_before_cooling_ones():
    groq, nvidia = _pool("groq", ["g1"], MagicMock()), _pool("nvidia", ["n1"], MagicMock())
    groq.mark_rate_limited("g1", 60, model="m-groq")
    targets = [LLMTarget("groq", "m-groq", None, pool=groq), LLMTarget("nvidia", "m-nv", None, pool=nvidia)]
    order = [(t.provider, s.key) for _, t, s in _attempt_plan(targets)]
    assert order == [("nvidia", "n1"), ("groq", "g1")]       # cooling Groq key is the last resort


@pytest.mark.anyio
async def test_unavailable_model_is_skipped_without_quarantining_the_key():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[_err(NotFoundError, 404), _resp("from qwen")])
    pool = _pool("groq", ["g1"], client)
    targets = [LLMTarget("groq", "groq/compound-mini", client, pool=pool), LLMTarget("groq", "qwen/qwen3.8-27b", client, pool=pool)]
    with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
        r = await call_llm_with_cross_provider_fallback([{"role": "user", "content": "hi"}], max_tokens=10)
    assert r.choices[0].message.content == "from qwen"
    slot = pool.slots[0]
    assert slot.cooldown_until == 0 and slot.model_cooldowns == {}   # the key was never quarantined


@pytest.mark.anyio
async def test_groq_429_on_one_model_falls_to_its_other_model_then_nvidia():
    groq_client, nv_client = MagicMock(), MagicMock()
    groq_client.chat.completions.create = AsyncMock(side_effect=[_err(RateLimitError, 429), _err(RateLimitError, 429)])
    nv_client.chat.completions.create = AsyncMock(return_value=_resp("from nvidia"))
    groq, nvidia = _pool("groq", ["g1"], groq_client), _pool("nvidia", ["n1"], nv_client)
    targets = [LLMTarget("groq", "openai/gpt-oss-120b", groq_client, pool=groq),
               LLMTarget("groq", "qwen/qwen3.8-27b", groq_client, pool=groq),
               LLMTarget("nvidia", "openai/gpt-oss-120b", nv_client, pool=nvidia)]
    with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
        r = await call_llm_with_cross_provider_fallback([{"role": "user", "content": "hi"}], max_tokens=10)
    assert r.choices[0].message.content == "from nvidia"
    called = [c.kwargs["model"] for c in groq_client.chat.completions.create.call_args_list]
    assert called == ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]     # second Groq model was tried, not skipped
    assert set(groq.slots[0].model_cooldowns) == {"openai/gpt-oss-120b", "qwen/qwen3.8-27b"}


@pytest.mark.anyio
async def test_stream_skips_unavailable_model():
    async def gen():
        for t in ["a", "b"]:
            yield MagicMock(choices=[MagicMock(delta=MagicMock(content=t))])

    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[_err(NotFoundError, 404), gen()])
    pool = _pool("nvidia", ["n1"], client)
    targets = [LLMTarget("nvidia", "bad/model", client, pool=pool), LLMTarget("nvidia", "good/model", client, pool=pool)]
    with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
        out = "".join([t async for t in stream_llm_with_cross_provider_fallback([{"role": "user", "content": "x"}], 10)])
    assert out == "ab"


def test_nvidia_tier_sits_after_groq_and_decommissioned_models_are_gone(monkeypatch):
    lp.reset_pools()
    monkeypatch.setattr(lp.settings, "nvidia_api_key", "nvapi-test")
    monkeypatch.setattr(lp.settings, "mistral_api_key", None)
    monkeypatch.setattr(lp.settings, "mistral_api_keys", None)
    try:
        for chain in (lp.get_generation_targets(), lp.get_fast_targets()):
            providers = [t.provider for t in chain]
            assert providers.index("nvidia") > max(i for i, p in enumerate(providers) if p == "groq")
            assert not {t.model for t in chain} & {"groq/compound-mini", "openai/gpt-oss-20b"}
            nv = [t.model for t in chain if t.provider == "nvidia"]
            assert nv[0] == lp.settings.nvidia_model and lp.settings.nvidia_backup_model in nv
    finally:
        lp.reset_pools()


def test_is_model_unavailable():
    assert is_model_unavailable(_err(NotFoundError, 404))
    assert is_model_unavailable(Exception("Error code: 410 - The model has reached its end of life"))
    assert is_model_unavailable(Exception("The model `x` has been decommissioned"))
    assert not is_model_unavailable(_err(RateLimitError, 429))


def test_numbered_keys_beyond_three_are_picked_up(monkeypatch):
    from app.config import Settings

    monkeypatch.setenv("GROQ_API_KEY_5", "g5")
    monkeypatch.setenv("GROQ_API_KEY_4", "g4")
    s = Settings(_env_file=None, supabase_url="x", supabase_key="x", database_url="x",
                 qdrant_url="x", qdrant_api_key="x", groq_api_key="g1")
    assert s.get_groq_keys() == ["g1", "g4", "g5"]


def test_reasoning_models_get_a_token_floor():
    from app.services.llm_provider import REASONING_MIN_TOKENS, effective_max_tokens

    assert effective_max_tokens("nvidia/nemotron-3-ultra-550b-a55b", 60) == REASONING_MIN_TOKENS
    assert effective_max_tokens("openai/gpt-oss-120b", 60) == REASONING_MIN_TOKENS
    assert effective_max_tokens("openai/gpt-oss-120b", 4000) == 4000         # never lowered
    assert effective_max_tokens("qwen/qwen3.8-27b", 60) == 60                 # short-reply model untouched


@pytest.mark.anyio
async def test_call_sends_raised_max_tokens_to_reasoning_model():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_resp("CATEGORY: SPECIFIC"))
    pool = _pool("nvidia", ["n1"], client)
    targets = [LLMTarget("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", client, pool=pool)]
    with patch("app.services.llm_provider.get_fast_targets", return_value=targets):
        await call_llm_with_cross_provider_fallback([{"role": "user", "content": "x"}], max_tokens=60, fast=True)
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == lp.REASONING_MIN_TOKENS


# ── Ported from the llm-key-pool router: quota-aware error handling ──────

GROQ_TPD = ("Error code: 429 - {'error': {'message': 'Rate limit reached for model `qwen/qwen3.8-27b` in organization "
            "`org_01abc` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199480, Requested 1352. "
            "Please try again in 5m59.424s.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}")


def test_error_classification():
    from app.services.llm_provider import classify_limit_error, retry_after_seconds, token_limit_of

    assert classify_limit_error(Exception(GROQ_TPD)) == "daily"
    assert abs(retry_after_seconds(Exception(GROQ_TPD)) - 359.424) < 0.01
    too_large = Exception("Error code: 413 - Request too large for model `x` on tokens per minute (TPM): Limit 8000, Requested 14210")
    assert classify_limit_error(too_large) == "too_large" and token_limit_of(too_large) == 8000
    assert classify_limit_error(_err(RateLimitError, 429)) == "rate"
    assert classify_limit_error(Exception("404 page not found, see https://x/rpm-docs")) is None   # whole words only
    # Gemini words per-minute and per-day hits the same way; the retry hint tells them apart
    assert classify_limit_error(Exception("429 You exceeded your current quota. retryDelay: '31s'")) == "rate"
    assert classify_limit_error(Exception("429 You exceeded your current quota.")) == "daily"


@pytest.mark.anyio
async def test_daily_limit_on_one_org_key_cools_every_key_of_that_org():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[Exception(GROQ_TPD), _resp("from nvidia")])
    groq = _pool("groq", ["g1", "g2", "g3"], client)
    for sl in groq.slots[1:]:
        sl.account = "org_01abc"           # learned from earlier errors
    nvidia = _pool("nvidia", ["n1"], client)
    targets = [LLMTarget("groq", "qwen/qwen3.8-27b", client, pool=groq), LLMTarget("nvidia", "m", client, pool=nvidia)]
    with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
        r = await call_llm_with_cross_provider_fallback([{"role": "user", "content": "hi"}], max_tokens=10)
    assert r.choices[0].message.content == "from nvidia"
    assert client.chat.completions.create.call_count == 2               # g2 and g3 were not tried
    for sl in groq.slots:
        assert sl.ready_at("qwen/qwen3.8-27b") - lp.time.time() > 300    # ~6 min, from "try again in 5m59s"


def test_shared_quota_setting_cools_all_keys(monkeypatch):
    monkeypatch.setenv("MISTRAL_SHARED_QUOTA", "true")
    pool = _pool("mistral", ["a", "b"], MagicMock())
    assert pool.shared_quota
    pool.cool_down_quota("a", "m", 30, "hit a rate limit")
    assert all(sl.ready_at("m") > lp.time.time() for sl in pool.slots)


@pytest.mark.anyio
async def test_too_large_skips_the_model_without_cooling_keys_and_oversized_requests_skip_groq():
    client = MagicMock()
    too_large = _err(RateLimitError, 413)
    client.chat.completions.create = AsyncMock(side_effect=[too_large, _resp("big model")])
    groq = _pool("groq", ["g1", "g2"], client)
    other = _pool("nvidia", ["n1"], client)
    targets = [LLMTarget("groq", "qwen/qwen3.8-27b", client, pool=groq), LLMTarget("nvidia", "m", client, pool=other)]
    with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
        await call_llm_with_cross_provider_fallback([{"role": "user", "content": "hi"}], max_tokens=10)
    assert client.chat.completions.create.call_count == 2              # g2 not burned on the same request
    assert all(sl.ready_at("qwen/qwen3.8-27b") == 0 for sl in groq.slots)

    client.chat.completions.create = AsyncMock(return_value=_resp("nvidia"))
    huge = [{"role": "user", "content": "x" * 40_000}]                  # ~10K tokens > Groq's 8K/min
    with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
        await call_llm_with_cross_provider_fallback(huge, max_tokens=10)
    assert client.chat.completions.create.call_args.kwargs["model"] == "m"
    assert client.chat.completions.create.call_count == 1               # Groq skipped without a call


def test_daily_quota_reset_times():
    from datetime import datetime, timezone

    from app.services.llm_provider import seconds_until_reset

    noon_utc = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    assert seconds_until_reset("UTC", noon_utc) == 12 * 3600
    assert seconds_until_reset("America/Los_Angeles", noon_utc) == 20 * 3600   # 04:00 PST -> midnight


def test_full_chain_order_and_missing_providers_are_skipped(monkeypatch):
    """Groq -> NVIDIA -> Mistral -> Gemini -> OpenRouter, from configured keys only (not the local .env)."""
    from app.config import Settings

    def chain(configured):
        lp.reset_pools()
        for p in ("groq", "nvidia", "mistral", "gemini", "openrouter"):
            monkeypatch.setattr(Settings, f"get_{p}_keys", lambda self, p=p: [f"{p}-k1"] if p in configured else [])
        try:
            return {kind: list(dict.fromkeys(t.provider for t in targets))
                    for kind, targets in (("generation", lp.get_generation_targets()), ("fast", lp.get_fast_targets()))}
        finally:
            lp.reset_pools()

    everything = chain({"groq", "nvidia", "mistral", "gemini", "openrouter"})
    assert everything["generation"] == ["groq", "nvidia", "mistral", "gemini", "openrouter"]
    assert everything["fast"][0] == "groq" and set(everything["fast"]) == set(everything["generation"])
    assert chain({"groq", "openrouter"})["generation"] == ["groq", "openrouter"]
