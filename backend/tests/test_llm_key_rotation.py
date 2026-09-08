"""
Unit tests for Multi-Key Pool Rotation per Provider Tier.

Verifies:
1. Multi-key configuration parsing (numbered keys + comma-separated strings).
2. Key masking utility for secure observability.
3. Round-robin candidate ordering and rotation.
4. Intra-tier key failover: when Key 1 hits 429 TPM, Key 2 immediately answers.
5. Cross-tier failover when all keys of a provider are quarantined.
6. Streaming key rotation before first token emission.
7. Cooldown recovery once timer elapses.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.config import Settings
from app.services.llm_provider import (
    KeySlot,
    LLMTarget,
    ProviderKeyPool,
    mask_key,
    call_llm_with_cross_provider_fallback,
    stream_llm_with_cross_provider_fallback,
    reset_pools,
)


def _make_dummy_response(content: str = "Test response"):
    msg = MagicMock()
    msg.content = content
    msg.reasoning = None

    choice = MagicMock()
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.total_tokens = 30

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_chunk(content: str):
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


class TestKeyConfigParsing:
    """Test parsing and deduplication of API keys from Settings."""

    def test_mask_key(self):
        assert mask_key("gsk_1234567890abcdef") == "gsk_...cdef"
        assert mask_key("AIzaSyDummySecretKey123") == "AIza...y123"
        assert mask_key("") == "<none>"
        assert mask_key("short") == "***"

    def test_numbered_keys_parsed(self):
        cfg = Settings(
            _env_file=None,
            supabase_url="https://mock.supabase.co",
            supabase_key="mock-key",
            database_url="postgresql+asyncpg://mock",
            qdrant_url="https://mock.qdrant.io",
            qdrant_api_key="mock-key",
            groq_api_key="gsk_key_1",
            groq_api_key_2="gsk_key_2",
            groq_api_key_3="gsk_key_3",
            gemini_api_key="AIza_gemini_1",
            gemini_api_key_2="AIza_gemini_2",
            openrouter_api_key="sk-or-v1-1",
            openrouter_api_key_2="sk-or-v1-2",
            openrouter_api_key_3="sk-or-v1-3",
        )
        assert cfg.get_groq_keys() == ["gsk_key_1", "gsk_key_2", "gsk_key_3"]
        assert cfg.get_gemini_keys() == ["AIza_gemini_1", "AIza_gemini_2"]
        assert cfg.get_openrouter_keys() == ["sk-or-v1-1", "sk-or-v1-2", "sk-or-v1-3"]

    def test_comma_separated_keys_parsed(self):
        cfg = Settings(
            _env_file=None,
            supabase_url="https://mock.supabase.co",
            supabase_key="mock-key",
            database_url="postgresql+asyncpg://mock",
            qdrant_url="https://mock.qdrant.io",
            qdrant_api_key="mock-key",
            groq_api_key="gsk_key_1",
            groq_api_keys="gsk_key_a, gsk_key_b, gsk_key_c",
        )
        keys = cfg.get_groq_keys()
        assert keys == ["gsk_key_a", "gsk_key_b", "gsk_key_c", "gsk_key_1"]


class TestProviderKeyPool:
    """Test ProviderKeyPool round-robin, cooldown quarantine, and recovery."""

    def test_round_robin_rotation(self):
        keys = ["key-A", "key-B", "key-C"]
        pool = ProviderKeyPool("groq", keys, lambda k: MagicMock(name=k))

        # First query: [A, B, C]
        c1 = [s.key for s in pool.get_rotation_candidates()]
        assert c1 == ["key-A", "key-B", "key-C"]

        # Second query: [B, C, A]
        c2 = [s.key for s in pool.get_rotation_candidates()]
        assert c2 == ["key-B", "key-C", "key-A"]

        # Third query: [C, A, B]
        c3 = [s.key for s in pool.get_rotation_candidates()]
        assert c3 == ["key-C", "key-A", "key-B"]

    def test_cooldown_quarantine_reorders_keys(self):
        keys = ["key-A", "key-B", "key-C"]
        pool = ProviderKeyPool("groq", keys, lambda k: MagicMock(name=k))

        # Mark key-A as rate limited for 60s
        pool.mark_rate_limited("key-A", cooldown_seconds=60.0)

        # Candidates should prioritize active keys (B, C), putting A at the end
        candidates = pool.get_rotation_candidates()
        keys_order = [s.key for s in candidates]
        assert keys_order[0] in ("key-B", "key-C")
        assert keys_order[1] in ("key-B", "key-C")
        assert keys_order[2] == "key-A"

    def test_cooldown_recovery(self):
        keys = ["key-A", "key-B"]
        pool = ProviderKeyPool("groq", keys, lambda k: MagicMock(name=k))

        # Mark key-A on cooldown for 0.05 seconds
        pool.mark_rate_limited("key-A", cooldown_seconds=0.05)
        assert pool.slots[0].cooldown_until > time.time()

        time.sleep(0.06)

        # After sleep, key-A should be considered active again
        candidates = pool.get_rotation_candidates()
        active = [s for s in candidates if s.cooldown_until <= time.time()]
        assert len(active) == 2


class TestIntraTierKeyFailover:
    """Test automated key rotation within a single provider before cross-tier failover."""

    @pytest.mark.anyio
    async def test_groq_key1_429_rotates_to_groq_key2(self):
        reset_pools()

        mock_client_1 = MagicMock()
        rate_limit_err = Exception("Rate limit reached: 429 TPM limit exceeded")
        setattr(rate_limit_err, "status_code", 429)
        mock_client_1.chat.completions.create = AsyncMock(side_effect=rate_limit_err)

        mock_client_2 = MagicMock()
        mock_client_2.chat.completions.create = AsyncMock(return_value=_make_dummy_response("Key 2 Success"))

        mock_gemini_client = MagicMock()

        def factory(key: str):
            if key == "groq-key-1":
                return mock_client_1
            elif key == "groq-key-2":
                return mock_client_2
            return MagicMock()

        groq_pool = ProviderKeyPool("groq", ["groq-key-1", "groq-key-2"], factory)
        gemini_pool = ProviderKeyPool("gemini", ["gemini-key-1"], lambda k: mock_gemini_client)

        target = LLMTarget(provider="groq", model="test-model", client=mock_client_1, pool=groq_pool)
        gemini_target = LLMTarget(provider="gemini", model="gemini-1.5-flash", client=mock_gemini_client, pool=gemini_pool)

        with patch("app.services.llm_provider.get_generation_targets", return_value=[target, gemini_target]):
            resp = await call_llm_with_cross_provider_fallback(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )
            assert resp.choices[0].message.content == "Key 2 Success"
            mock_client_1.chat.completions.create.assert_awaited_once()
            mock_client_2.chat.completions.create.assert_awaited_once()
            # Crucial: Gemini should NOT have been called because Key 2 resolved it!
            mock_gemini_client.chat.completions.create.assert_not_called()

    @pytest.mark.anyio
    async def test_groq_all_keys_429_escalates_to_gemini(self):
        reset_pools()

        mock_groq_1 = MagicMock()
        mock_groq_1.chat.completions.create = AsyncMock(side_effect=Exception("TPM limit"))
        mock_groq_2 = MagicMock()
        mock_groq_2.chat.completions.create = AsyncMock(side_effect=Exception("TPM limit"))
        mock_groq_3 = MagicMock()
        mock_groq_3.chat.completions.create = AsyncMock(side_effect=Exception("TPM limit"))

        mock_gemini = MagicMock()
        mock_gemini.chat.completions.create = AsyncMock(return_value=_make_dummy_response("Gemini fallback"))

        def groq_factory(key: str):
            if key == "k1":
                return mock_groq_1
            elif key == "k2":
                return mock_groq_2
            return mock_groq_3

        groq_pool = ProviderKeyPool("groq", ["k1", "k2", "k3"], groq_factory)
        gemini_pool = ProviderKeyPool("gemini", ["gem-1"], lambda k: mock_gemini)

        targets = [
            LLMTarget(provider="groq", model="groq-model", client=mock_groq_1, pool=groq_pool),
            LLMTarget(provider="gemini", model="gemini-flash", client=mock_gemini, pool=gemini_pool),
        ]

        with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
            resp = await call_llm_with_cross_provider_fallback(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            )
            assert resp.choices[0].message.content == "Gemini fallback"
            mock_groq_1.chat.completions.create.assert_awaited_once()
            mock_groq_2.chat.completions.create.assert_awaited_once()
            mock_groq_3.chat.completions.create.assert_awaited_once()
            mock_gemini.chat.completions.create.assert_awaited_once()


class TestStreamingKeyRotation:
    """Test streaming key rotation before token emission."""

    @pytest.mark.anyio
    async def test_stream_failover_from_key1_to_key2(self):
        reset_pools()

        mock_client_1 = MagicMock()
        err = Exception("Rate limit reached: 429")
        setattr(err, "status_code", 429)
        mock_client_1.chat.completions.create = AsyncMock(side_effect=err)

        async def mock_stream_2():
            yield _make_chunk("Rotated ")
            yield _make_chunk("stream")

        mock_client_2 = MagicMock()
        mock_client_2.chat.completions.create = AsyncMock(return_value=mock_stream_2())

        def factory(key: str):
            return mock_client_1 if key == "k1" else mock_client_2

        pool = ProviderKeyPool("groq", ["k1", "k2"], factory)
        target = LLMTarget(provider="groq", model="test-model", client=mock_client_1, pool=pool)

        with patch("app.services.llm_provider.get_generation_targets", return_value=[target]):
            tokens = []
            async for token in stream_llm_with_cross_provider_fallback(
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=100,
            ):
                tokens.append(token)

            assert "".join(tokens) == "Rotated stream"
            mock_client_1.chat.completions.create.assert_awaited_once()
            mock_client_2.chat.completions.create.assert_awaited_once()
