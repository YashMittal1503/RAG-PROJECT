"""
FastAPI application entry point.

Assembles the app with:
- Lifespan events (startup/shutdown) for model loading and client cleanup
- CORS middleware for frontend communication
- All API routers
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import asyncio
from app.auth import warm_up_auth
from app.config import settings
from app.database import engine, warm_up_db, db_heartbeat_task
from app.routers import chat, documents, health
from app.services import embedding, vector_store, reranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — runs on startup and shutdown.

    Startup:
    - Loads the FastEmbed embedding model into memory (once).
    - Loads the FlashRank cross-encoder reranker model (once).
    - Pre-warms database connection pool (eliminates 7s cold start).
    - Pre-warms Supabase Auth JWKS keys.
    - Starts background DB heartbeat task to keep connections hot.

    Shutdown:
    - Cancels heartbeat task.
    - Closes Qdrant client connection and SQLAlchemy engine.
    """
    # ── Startup ───────────────────────────────────────────────────────
    logger.info("Starting up — pre-warming services...")
    embedding.init_model()
    reranker.init_ranker()
    warm_up_auth()
    await warm_up_db()
    heartbeat_task = asyncio.create_task(db_heartbeat_task())
    logger.info("Startup complete — all services pre-warmed and ready.")

    yield  # App is running

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Shutting down...")
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

    client = await vector_store.get_client()
    await client.close()
    await engine.dispose()
    logger.info("Shutdown complete.")


# ── Create the FastAPI app ────────────────────────────────────────────────

app = FastAPI(
    title="RAG Chatbot API",
    description="Document Q&A with streaming answers and source citations",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────
# Allow the frontend origin. In production, this should be the Vercel URL.
# We also allow localhost for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_url,
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-Title"],
)

# ── Mount routers ─────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint — redirects to the API docs."""
    return {"message": "RAG Chatbot API", "docs": "/docs"}
