"""
Unit tests for Resilient Cross-Provider LLM Fallback Chain.

Verifies:
1. Provider target resolution (Groq -> Gemini -> OpenRouter).
2. Transient and rate-limit error classification (is_recoverable_error).
3. Automated failover from Groq to Gemini and OpenRouter for non-streaming calls.
4. Immediate escalation on unrecoverable errors (e.g., 401 unauthorized).
5. Seamless failover during streaming calls before first token emission.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from groq import APIError as GroqAPIError, RateLimitError as GroqRateLimitError
from openai import APIError as OpenAIAPIError, RateLimitError as OpenAIRateLimitError
import httpx

from app.services.llm_provider import (
    LLMTarget,
    get_generation_targets,
    get_fast_targets,
    is_recoverable_error,
    call_llm_with_cross_provider_fallback,
    stream_llm_with_cross_provider_fallback,
)


def _make_dummy_response(content: str = "Test response"):
    """Helper to create an OpenAI/Groq-compatible ChatCompletion response."""
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
    """Helper to create a streaming ChatCompletionChunk."""
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


class TestLLMTargetBuilding:
    """Test resolution of multi-tier provider chains."""

    def test_default_targets_groq_only(self):
        with patch("app.services.llm_provider.settings.gemini_api_key", None), \
             patch("app.services.llm_provider.settings.openrouter_api_key", None), \
             patch("app.services.llm_provider._gemini_client", None), \
             patch("app.services.llm_provider._openrouter_client", None):
            targets = get_generation_targets()
            assert len(targets) > 0
            assert all(t.provider == "groq" for t in targets)

    def test_targets_with_gemini_and_openrouter(self):
        with patch("app.services.llm_provider.settings.gemini_api_key", "mock-gemini-key"), \
             patch("app.services.llm_provider.settings.openrouter_api_key", "mock-openrouter-key"), \
             patch("app.services.llm_provider._gemini_client", MagicMock()), \
             patch("app.services.llm_provider._openrouter_client", MagicMock()):
            targets = get_generation_targets()
            providers = [t.provider for t in targets]
            assert "groq" in providers
            assert "gemini" in providers
            assert "openrouter" in providers

            # Check tier order: Groq first, then Gemini, then OpenRouter
            first_groq_idx = providers.index("groq")
            first_gemini_idx = providers.index("gemini")
            first_openrouter_idx = providers.index("openrouter")
            assert first_groq_idx < first_gemini_idx < first_openrouter_idx

    def test_fast_targets_tiering(self):
        with patch("app.services.llm_provider.settings.gemini_api_key", "mock-gemini-key"), \
             patch("app.services.llm_provider.settings.openrouter_api_key", "mock-openrouter-key"), \
             patch("app.services.llm_provider._gemini_client", MagicMock()), \
             patch("app.services.llm_provider._openrouter_client", MagicMock()):
            fast_targets = get_fast_targets()
            assert len(fast_targets) > 0
            providers = [t.provider for t in fast_targets]
            assert "groq" in providers
            assert "gemini" in providers
            assert "openrouter" in providers


class TestRecoverableErrors:
    """Test recoverable vs unrecoverable error classification."""

    def test_detects_rate_limit_and_tpm_errors(self):
        err1 = Exception("Rate limit reached: TPM limit exceeded")
        err2 = Exception("tokens per minute limit reached for model")
        err3 = Exception("Service is temporarily overloaded")
        assert is_recoverable_error(err1) is True
        assert is_recoverable_error(err2) is True
        assert is_recoverable_error(err3) is True

    def test_detects_http_status_codes(self):
        err429 = Exception("Rate limited")
        setattr(err429, "status_code", 429)

        err503 = Exception("Service unavailable")
        setattr(err503, "status_code", 503)

        assert is_recoverable_error(err429) is True
        assert is_recoverable_error(err503) is True

    def test_network_and_timeout_errors(self):
        req = httpx.Request("POST", "https://api.groq.com")
        timeout_err = httpx.ReadTimeout("Timed out", request=req)
        net_err = httpx.NetworkError("Connection refused", request=req)
        assert is_recoverable_error(timeout_err) is True
        assert is_recoverable_error(net_err) is True

    def test_unrecoverable_errors(self):
        err_auth = Exception("Invalid API key provided")
        setattr(err_auth, "status_code", 401)

        err_bad_req = Exception("Invalid model parameter")
        setattr(err_bad_req, "status_code", 400)

        assert is_recoverable_error(err_auth) is False
        assert is_recoverable_error(err_bad_req) is False


class TestCrossProviderNonStreamingFallback:
    """Test non-streaming fallback across providers."""

    @pytest.mark.anyio
    async def test_primary_groq_success(self):
        mock_groq_client = MagicMock()
        expected = _make_dummy_response("Primary Groq response")
        mock_groq_client.chat.completions.create = AsyncMock(return_value=expected)

        target = LLMTarget(provider="groq", model="test-groq", client=mock_groq_client)

        with patch("app.services.llm_provider.get_generation_targets", return_value=[target]):
            resp = await call_llm_with_cross_provider_fallback(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            )
            assert resp.choices[0].message.content == "Primary Groq response"
            mock_groq_client.chat.completions.create.assert_awaited_once()

    @pytest.mark.anyio
    async def test_fallback_from_groq_to_gemini(self):
        mock_groq = MagicMock()
        rate_limit_err = Exception("Rate limit exceeded: 429 TPM")
        setattr(rate_limit_err, "status_code", 429)
        mock_groq.chat.completions.create = AsyncMock(side_effect=rate_limit_err)

        mock_gemini = MagicMock()
        expected = _make_dummy_response("Gemini fallback answer")
        mock_gemini.chat.completions.create = AsyncMock(return_value=expected)

        targets = [
            LLMTarget(provider="groq", model="groq-model-1", client=mock_groq),
            LLMTarget(provider="gemini", model="gemini-1.5-flash", client=mock_gemini),
        ]

        with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
            resp = await call_llm_with_cross_provider_fallback(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            )
            assert resp.choices[0].message.content == "Gemini fallback answer"
            mock_groq.chat.completions.create.assert_awaited_once()
            mock_gemini.chat.completions.create.assert_awaited_once()

    @pytest.mark.anyio
    async def test_multi_tier_fallback_to_openrouter(self):
        mock_groq = MagicMock()
        mock_groq.chat.completions.create = AsyncMock(side_effect=Exception("Groq TPM limit reached"))

        mock_gemini = MagicMock()
        mock_gemini.chat.completions.create = AsyncMock(side_effect=Exception("Gemini quota exhausted"))

        mock_openrouter = MagicMock()
        expected = _make_dummy_response("OpenRouter backup response")
        mock_openrouter.chat.completions.create = AsyncMock(return_value=expected)

        targets = [
            LLMTarget(provider="groq", model="groq-1", client=mock_groq),
            LLMTarget(provider="gemini", model="gemini-flash", client=mock_gemini),
            LLMTarget(provider="openrouter", model="openrouter-llama", client=mock_openrouter),
        ]

        with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
            resp = await call_llm_with_cross_provider_fallback(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            )
            assert resp.choices[0].message.content == "OpenRouter backup response"
            mock_groq.chat.completions.create.assert_awaited_once()
            mock_gemini.chat.completions.create.assert_awaited_once()
            mock_openrouter.chat.completions.create.assert_awaited_once()

    @pytest.mark.anyio
    async def test_unrecoverable_error_aborts_immediately(self):
        mock_groq = MagicMock()
        auth_err = Exception("Invalid API key")
        setattr(auth_err, "status_code", 401)
        mock_groq.chat.completions.create = AsyncMock(side_effect=auth_err)

        mock_gemini = MagicMock()
        mock_gemini.chat.completions.create = AsyncMock()

        targets = [
            LLMTarget(provider="groq", model="groq-1", client=mock_groq),
            LLMTarget(provider="gemini", model="gemini-flash", client=mock_gemini),
        ]

        with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
            with pytest.raises(Exception) as exc_info:
                await call_llm_with_cross_provider_fallback(
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=100,
                )
            assert "Invalid API key" in str(exc_info.value)
            mock_gemini.chat.completions.create.assert_not_called()


class TestCrossProviderStreamingFallback:
    """Test streaming fallback across providers."""

    @pytest.mark.anyio
    async def test_streaming_success(self):
        async def mock_stream():
            yield _make_chunk("Hello")
            yield _make_chunk(" world")

        mock_groq = MagicMock()
        mock_groq.chat.completions.create = AsyncMock(return_value=mock_stream())

        targets = [LLMTarget(provider="groq", model="groq-1", client=mock_groq)]

        with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
            tokens = []
            async for token in stream_llm_with_cross_provider_fallback(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            ):
                tokens.append(token)

            assert "".join(tokens) == "Hello world"

    @pytest.mark.anyio
    async def test_streaming_failover_before_start(self):
        mock_groq = MagicMock()
        rate_limit_err = Exception("Rate limit reached: 429")
        setattr(rate_limit_err, "status_code", 429)
        mock_groq.chat.completions.create = AsyncMock(side_effect=rate_limit_err)

        async def mock_gemini_stream():
            yield _make_chunk("Fallback")
            yield _make_chunk(" stream")

        mock_gemini = MagicMock()
        mock_gemini.chat.completions.create = AsyncMock(return_value=mock_gemini_stream())

        targets = [
            LLMTarget(provider="groq", model="groq-1", client=mock_groq),
            LLMTarget(provider="gemini", model="gemini-flash", client=mock_gemini),
        ]

        with patch("app.services.llm_provider.get_generation_targets", return_value=targets):
            tokens = []
            async for token in stream_llm_with_cross_provider_fallback(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=100,
            ):
                tokens.append(token)

            assert "".join(tokens) == "Fallback stream"
