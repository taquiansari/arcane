"""FastAPI application factory for Arcane."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from arcane import __version__
from arcane.config import get_settings
from arcane.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup and shutdown events."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info(
        "arcane_starting",
        version=__version__,
        environment=settings.environment,
    )

    # Verify Redis connectivity on startup
    try:
        import redis

        client = redis.from_url(settings.redis_url, decode_responses=True)
        client.ping()
        logger.info("redis_connected", url=settings.redis_url)
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))

    yield

    logger.info("arcane_shutting_down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Arcane — Agentic Research Intelligence",
        description=(
            "Multi-agent research system with RAG pipeline, "
            "LangGraph orchestration, and CrewAI task delegation."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    from arcane.api.routes.health import router as health_router
    from arcane.api.routes.research import router as research_router
    from arcane.api.routes.sessions import router as sessions_router
    from arcane.api.websocket import router as ws_router

    app.include_router(health_router, prefix="/api/v1")
    app.include_router(research_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(ws_router)

    # Serve static frontend files
    frontend_dir = Path(__file__).parent.parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/")
        async def serve_frontend():
            return FileResponse(str(frontend_dir / "index.html"))

    logger.info("arcane_app_created", routes=len(app.routes))
    return app
