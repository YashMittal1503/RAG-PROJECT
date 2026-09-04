"""
Health check endpoint.

Used by the frontend to detect whether the backend is awake
(Render free tier sleeps after 15 min of inactivity).
"""

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
    )
