"""
Application configuration loaded from environment variables.

Uses Pydantic Settings to validate and type-check all config values at startup.
If any required env var is missing, the app fails fast with a clear error.
"""

import os
import re

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application configuration, loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",  # numbered keys beyond _3 (GROQ_API_KEY_4, _5, ...) land here; see _numbered_keys
    )

    def _numbered_keys(self, provider: str) -> list[str]:
        """Any <PROVIDER>_API_KEY_<n> from .env or the environment, in numeric order (no fixed upper limit)."""
        pat = re.compile(rf"^{provider}_api_key_(\d+)$", re.I)
        found: dict[int, str] = {}
        for source in ((self.model_extra or {}), os.environ):
            for name, value in source.items():
                m = pat.match(name)
                if m and isinstance(value, str) and value.strip():
                    found.setdefault(int(m.group(1)), value)
        return [found[n] for n in sorted(found)]

    # ── Supabase ──────────────────────────────────────────────────────────
    supabase_url: str
    supabase_key: str                # Service-role key (server-side only)
    supabase_jwt_secret: str | None = None  # Optional, using JWKS for verification

    # ── Database (Supabase Postgres) ──────────────────────────────────────
    database_url: str                # e.g. postgresql+asyncpg://...

    # ── Qdrant Cloud ──────────────────────────────────────────────────────
    qdrant_url: str
    qdrant_api_key: str

    # ── Groq LLM (Tier 1 Primary — sub-second LPUs) ──────────────────────
    groq_api_key: str
    groq_api_key_2: str | None = None
    groq_api_key_3: str | None = None
    groq_api_keys: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

    # ── Embedding & Reranker ──────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    enable_reranker: bool = True
    embedding_threads: int = 2
    enable_onnx_arena: bool = False
    embed_batch_size: int = 8

    # ── Tabular Data Store (DuckDB) ───────────────────────────────────────
    duckdb_memory_limit: str = "256MB"
    duckdb_threads: int = 2

    # ── Upload limits ─────────────────────────────────────────────────────
    max_file_size_mb: int = 20
    max_batch_size: int = 10

    # ── Frontend URL (for CORS) ───────────────────────────────────────────
    frontend_url: str = "http://localhost:3000"

    # ── Observability (Logfire / OpenTelemetry) ───────────────────────────
    logfire_token: str | None = None

    # ── Multi-Provider LLM Fallbacks (Google Gemini & OpenRouter) ─────────
    gemini_api_key: str | None = None
    gemini_api_key_2: str | None = None
    gemini_api_key_3: str | None = None
    gemini_api_keys: str | None = None
    gemini_model: str = "gemini-flash-latest"

    openrouter_api_key: str | None = None
    openrouter_api_key_2: str | None = None
    openrouter_api_key_3: str | None = None
    openrouter_api_keys: str | None = None
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"

    # ── Mistral AI (High TPM & RAGAS Judge: https://console.mistral.ai) ────
    mistral_api_key: str | None = None
    mistral_api_key_2: str | None = None
    mistral_api_key_3: str | None = None
    mistral_api_keys: str | None = None
    mistral_model: str = "ministral-3b-2512"

    # ── NVIDIA NIM (free, ~40 RPM, OpenAI-compatible: https://build.nvidia.com) ──
    nvidia_api_key: str | None = None
    nvidia_api_key_2: str | None = None
    nvidia_api_key_3: str | None = None
    nvidia_api_keys: str | None = None
    nvidia_model: str = "nvidia/nemotron-3-ultra-550b-a55b"   # answer generation
    nvidia_fast_model: str = "nvidia/nemotron-3-ultra-550b-a55b"  # intent / rewrite / SQL
    nvidia_backup_model: str = "minimaxai/minimax-m3.1-flash"  # tried if the first is unavailable or limited
    # (checked 2026-11-23: NVIDIA retired gpt-oss-120b and qwen3.5-397b; several others time out on the free tier)

    def get_nvidia_keys(self) -> list[str]:
        """Collect all configured NVIDIA NIM API keys."""
        return _parse_keys(
            self.nvidia_api_key,
            self.nvidia_api_key_2,
            self.nvidia_api_key_3,
            *self._numbered_keys("nvidia"),
            comma_separated=self.nvidia_api_keys,
        )

    def get_groq_keys(self) -> list[str]:
        """Collect all configured Groq API keys."""
        return _parse_keys(
            self.groq_api_key,
            self.groq_api_key_2,
            self.groq_api_key_3,
            *self._numbered_keys("groq"),
            comma_separated=self.groq_api_keys,
        )

    def get_gemini_keys(self) -> list[str]:
        """Collect all configured Google Gemini API keys."""
        return _parse_keys(
            self.gemini_api_key,
            self.gemini_api_key_2,
            self.gemini_api_key_3,
            *self._numbered_keys("gemini"),
            comma_separated=self.gemini_api_keys,
        )

    def get_openrouter_keys(self) -> list[str]:
        """Collect all configured OpenRouter API keys."""
        return _parse_keys(
            self.openrouter_api_key,
            self.openrouter_api_key_2,
            self.openrouter_api_key_3,
            *self._numbered_keys("openrouter"),
            comma_separated=self.openrouter_api_keys,
        )

    def get_mistral_keys(self) -> list[str]:
        """Collect all configured Mistral API keys."""
        return _parse_keys(
            self.mistral_api_key,
            self.mistral_api_key_2,
            self.mistral_api_key_3,
            *self._numbered_keys("mistral"),
            comma_separated=self.mistral_api_keys,
        )


def _parse_keys(*individual: str | None, comma_separated: str | None = None) -> list[str]:
    """Helper to parse and deduplicate individual and comma-separated API keys."""
    keys: list[str] = []
    if comma_separated:
        for k in comma_separated.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    for k in individual:
        if k:
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    return keys


# Singleton — imported throughout the app
settings = Settings()
