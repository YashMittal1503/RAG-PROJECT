"""
Application configuration loaded from environment variables.

Uses Pydantic Settings to validate and type-check all config values at startup.
If any required env var is missing, the app fails fast with a clear error.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application configuration, loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

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

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ── SVD-RAG (Singular Value Decomposition Tree-Organized RAG) ────────
    enable_svd_rag: bool = True
    svd_tau: float = 0.95
    svd_cluster_size: int = 6
    svd_min_chunks: int = 4

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

    def get_groq_keys(self) -> list[str]:
        """Collect all configured Groq API keys."""
        return _parse_keys(
            self.groq_api_key,
            self.groq_api_key_2,
            self.groq_api_key_3,
            comma_separated=self.groq_api_keys,
        )

    def get_gemini_keys(self) -> list[str]:
        """Collect all configured Google Gemini API keys."""
        return _parse_keys(
            self.gemini_api_key,
            self.gemini_api_key_2,
            self.gemini_api_key_3,
            comma_separated=self.gemini_api_keys,
        )

    def get_openrouter_keys(self) -> list[str]:
        """Collect all configured OpenRouter API keys."""
        return _parse_keys(
            self.openrouter_api_key,
            self.openrouter_api_key_2,
            self.openrouter_api_key_3,
            comma_separated=self.openrouter_api_keys,
        )

    def get_mistral_keys(self) -> list[str]:
        """Collect all configured Mistral API keys."""
        return _parse_keys(
            self.mistral_api_key,
            self.mistral_api_key_2,
            self.mistral_api_key_3,
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
