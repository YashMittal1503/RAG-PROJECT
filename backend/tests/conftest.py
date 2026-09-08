"""
Test configuration and fixtures for pytest.
"""

import os
import pytest

# Provide mock environment variables for unit tests so Settings initializes cleanly
# both in local test runners and in GitHub Actions CI where real secrets are absent.
os.environ.setdefault("SUPABASE_URL", "https://mock.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "mock-supabase-key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://mock:mock@localhost:5432/mock")
os.environ.setdefault("QDRANT_URL", "https://mock.qdrant.io:6333")
os.environ.setdefault("QDRANT_API_KEY", "mock-qdrant-key")
os.environ.setdefault("GROQ_API_KEY", "mock-groq-key")
