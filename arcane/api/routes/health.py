"""Health check endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from arcane import __version__
from arcane.api.schemas import HealthResponse
from arcane.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health and Redis connectivity."""
    redis_ok = False

    try:
        import redis as redis_lib
        from arcane.config import get_settings

        settings = get_settings()
        client = redis_lib.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        redis_ok = True
    except Exception as e:
        logger.warning("health_redis_failed", error=str(e))

    status = "healthy" if redis_ok else "degraded"

    return HealthResponse(
        status=status,
        redis_connected=redis_ok,
        version=__version__,
    )
