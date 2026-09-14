"""
FastAPI application entry point.

Assembles the app with:
- Lifespan events (startup/shutdown) for model loading and client cleanup
- CORS middleware for frontend communication
- All API routers
"""

import logging
from contextlib import asynccontextmanager

import sys

# Ensure UTF-8 console output on Windows so emoji span logs print cleanly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time
from fastapi import FastAPI, Request
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
    embedding.init_sparse_model()
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

import logfire

def _logfire_scrub_callback(match: logfire.ScrubMatch):
    """
    Ensure chat sessions, session IDs, and URL paths are never redacted in traces,
    while preserving scrubbing for actual secrets (passwords, tokens, API keys).
    """
    matched_word = match.pattern_match.group(0).lower()
    if matched_word == "session":
        return match.value
    return None

# ── Configure Logfire Observability ──────────────────────────────────────────
if settings.logfire_token:
    logfire.configure(
        token=settings.logfire_token,
        service_name="docuchat-api",
        service_version="1.0.0",
        environment="development" if "localhost" in settings.frontend_url else "production",
        inspect_arguments=True,
        scrubbing=logfire.ScrubbingOptions(callback=_logfire_scrub_callback),
    )
    logger.info("Logfire cloud observability enabled (service=docuchat-api).")
else:
    logfire.configure(
        send_to_logfire=False,
        service_name="docuchat-api",
        service_version="1.0.0",
        inspect_arguments=True,
        scrubbing=logfire.ScrubbingOptions(callback=_logfire_scrub_callback),
    )
    logger.info("Logfire local structured tracing enabled (no token provided).")

# Route standard Python application logs to Logfire for unified trace logs (excluding noisy uvicorn access)
try:
    logfire_handler = logfire.LogfireLoggingHandler()
    logging.getLogger("app").addHandler(logfire_handler)
except Exception as log_err:
    logger.warning(f"Could not attach LogfireLoggingHandler: {log_err}")

# Auto-instrumentation for AI model streaming and external provider HTTP requests
try:
    logfire.instrument_openai()
    logfire.instrument_httpx(capture_all=False)
    logger.info("Logfire AI model & provider auto-instrumentations active (openai, httpx).")
except Exception as inst_err:
    logger.warning(f"Could not initialize Logfire sub-instrumentations: {inst_err}")

app = FastAPI(
    title="RAG Chatbot API",
    description="Document Q&A with streaming answers and source citations",
    version="1.0.0",
    lifespan=lifespan,
)


def _logfire_request_attributes_mapper(request_or_ws, attributes):
    """Extract session_id and doc_id so every trace has first-class session and document tags."""
    try:
        url = getattr(request_or_ws, "url", None)
        if url:
            path = url.path
            parts = path.strip("/").split("/")
            if "sessions" in parts:
                idx = parts.index("sessions")
                if idx + 1 < len(parts):
                    attributes["session_id"] = parts[idx + 1]
            if "documents" in parts:
                idx = parts.index("documents")
                if idx + 1 < len(parts):
                    attributes["doc_id"] = parts[idx + 1]
    except Exception:
        pass
    return attributes


# Instrument FastAPI while excluding routine healthchecks, docs, and high-frequency polling paths
logfire.instrument_fastapi(
    app,
    capture_headers=True,
    record_send_receive=True,
    request_attributes_mapper=_logfire_request_attributes_mapper,
    excluded_urls="/api/health,/health,/docs,/openapi.json,/api/chat/sessions$,/api/documents$,.*messages$,.*status$",
)


def _is_routine_request(method: str, path: str) -> bool:
    """Filter out routine background polling, health-checks, and read-only list fetches."""
    if path in ("/api/health", "/health", "/docs", "/openapi.json", "/favicon.ico"):
        return True
    # Filter routine read-only GET polls that flood the dashboard
    if method in ("GET", "HEAD", "OPTIONS"):
        if path in ("/api/chat/sessions", "/api/documents"):
            return True
        if path.endswith("/messages") or path.endswith("/status"):
            return True
    return False


# ── Global HTTP Request / Response Observability Middleware ──────────────────
@app.middleware("http")
async def logfire_request_middleware(request: Request, call_next):
    """Capture significant API actions (queries, uploads, deletions) while suppressing routine background polling."""
    path = request.url.path
    method = request.method
    is_routine = _is_routine_request(method, path)

    if is_routine:
        return await call_next(request)

    start_time = time.perf_counter()
    client_ip = request.client.host if request.client else "unknown"

    # Extract session_id or doc_id from path
    session_id = None
    doc_id = None
    parts = path.strip("/").split("/")
    if "sessions" in parts:
        try:
            idx = parts.index("sessions")
            if idx + 1 < len(parts):
                session_id = parts[idx + 1]
        except Exception:
            pass
    if "documents" in parts:
        try:
            idx = parts.index("documents")
            if idx + 1 < len(parts):
                doc_id = parts[idx + 1]
        except Exception:
            pass

    logfire.info(
        "📥 HTTP {method} {path} [{client_ip}]",
        method=method,
        path=path,
        client_ip=client_ip,
        session_id=session_id,
        doc_id=doc_id,
    )

    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logfire.info(
            "📤 HTTP {method} {path} -> {status_code} ({duration_ms}ms)",
            method=method,
            path=path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            session_id=session_id,
            doc_id=doc_id,
        )
        return response
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logfire.error(
            "💥 HTTP {method} {path} failed: {error} ({duration_ms}ms)",
            method=method,
            path=path,
            error=str(exc),
            duration_ms=duration_ms,
            session_id=session_id,
            doc_id=doc_id,
        )
        raise

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
