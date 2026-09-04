"""
Async SQLAlchemy engine and session factory.

Provides:
- `engine`:  the async engine connected to Supabase Postgres
- `AsyncSessionLocal`:  a session factory for creating async sessions
- `get_db()`:  a FastAPI dependency that yields a session per request
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# Create the async engine.
# pool_pre_ping=True ensures stale connections are detected and recycled.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
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
