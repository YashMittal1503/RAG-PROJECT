"""
Async SQLAlchemy engine and session factory.

Provides:
- `engine`:  the async engine connected to Supabase Postgres
- `AsyncSessionLocal`:  a session factory for creating async sessions
- `get_db()`:  a FastAPI dependency that yields a session per request
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

logger = logging.getLogger(__name__)

# Create the async engine.
# pool_pre_ping=False eliminates the ~500ms cross-ocean ping on every checkout;
# the db_heartbeat_task keeps connections active, and pool_recycle=300 cleans stale sockets.
# statement_cache_size=0 avoids prepared statement cache conflicts with Supabase connection poolers.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=False,
    pool_size=5,
    max_overflow=5,
    pool_recycle=300,
    pool_timeout=30,
    connect_args={
        "statement_cache_size": 0,
        "command_timeout": 60,
    },
)

# Session factory.
# expire_on_commit=False prevents lazy-load errors after commit —
# once we commit, we don't want SQLAlchemy to expire the in-memory
# attributes, because we often return them in the same request.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db():
    """
    FastAPI dependency that yields an async database session.
    The session is automatically closed when the request finishes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def warm_up_db() -> None:
    """
    Warm up the database connection pool on application startup.
    Establishes the initial connection (which takes ~7s on remote Supabase)
    so user requests execute immediately.
    """
    try:
        logger.info("Warming up database connection pool...")
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        logger.info("Database connection pool warmed up successfully.")
    except Exception as e:
        logger.warning(f"Database warm-up failed (will connect on demand): {e}")


async def db_heartbeat_task() -> None:
    """
    Periodic heartbeat task to keep the database connection pool warm
    and prevent Supabase / intermediate proxies from terminating idle connections.
    """
    while True:
        try:
            await asyncio.sleep(45)
            async with AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Database heartbeat ping warning: {e}")

