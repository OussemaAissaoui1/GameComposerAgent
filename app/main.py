"""
Game Maker Agent v1 — FastAPI Application Entry Point
Stateless. Structured logging. Event-bus compatible.
"""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import router


def _configure_logging(level: str) -> None:
    """Set up structured logging to stdout."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def create_app() -> FastAPI:
    """Application factory — creates a fresh FastAPI instance.

    No global mutable state. Each call produces an independent app.
    """
    settings = get_settings()

    _configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Production-ready Game Maker Agent that generates MCQ puzzles "
            "from PDF chapters with blockchain-compatible anchor strings."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — permissive for dev; tighten in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/")
    async def root() -> dict:
        return {
            "service": settings.app_name,
            "version": settings.version,
            "endpoints": {
                "generate_game": "POST /generate-game",
                "health": "GET /health",
                "docs": "GET /docs",
            },
        }

    @app.get("/health")
    async def health_check() -> dict:
        return {
            "status": "healthy",
            "version": settings.version,
            "model": settings.llm.model_name,
        }

    return app


# ── ASGI entry point ─────────────────────────────────────────────────────────
app = create_app()
