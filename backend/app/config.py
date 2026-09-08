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

    # ── Groq LLM ──────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "groq/compound"

    # ── Embedding ─────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # ── Upload limits ─────────────────────────────────────────────────────
    max_file_size_mb: int = 20
    max_batch_size: int = 10

    # ── Frontend URL (for CORS) ───────────────────────────────────────────
    frontend_url: str = "http://localhost:3000"


# Singleton — imported throughout the app
settings = Settings()
