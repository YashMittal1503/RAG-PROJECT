"""
Health check endpoint.

Used by the frontend to detect whether the backend is awake, by Docker's HEALTHCHECK, and by the
deploy pipeline, which waits until `commit` reports the commit it just shipped.
"""

import os

from fastapi import APIRouter

from app.schemas import HealthResponse
from app.services.embedding import is_model_loaded

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns the backend status and whether the embedding model is loaded.

    The frontend polls this endpoint to show a "waking up" banner
    when the backend is cold-starting.
    """
    return HealthResponse(
        status="ok",
        model_loaded=is_model_loaded(),
        version="2.1.0",
        max_chunk_tokens=512,
        commit=os.environ.get("GIT_SHA", "dev"),
    )
