"""
FastAPI application entry point.

KEY CONCEPTS:

LIFESPAN CONTEXT MANAGER:
- Replaces the deprecated @app.on_event("startup") / @app.on_event("shutdown")
- Code before `yield` runs at startup, code after runs at shutdown
- Uses Python's contextlib.asynccontextmanager
- Perfect for: DB connection pools, service initialization, file cleanup

WHY CONTEXT MANAGERS FOR STARTUP/SHUTDOWN?
- Resources allocated at startup MUST be cleaned up at shutdown
- Context managers guarantee cleanup even if the server crashes
- Without cleanup: leaked DB connections, orphaned temp files, zombie processes

CORS MIDDLEWARE:
- Cross-Origin Resource Sharing
- Browsers block requests from frontend (localhost:3000) to API (localhost:8000)
- CORS headers tell the browser "it's OK, allow this cross-origin request" 
- Without CORS: your React/Vue frontend will break with "CORS policy" errors

APP.STATE:
- FastAPI's built-in mechanism for sharing objects across requests
- Services are created once at startup, shared by all request handlers
- Thread-safe because asyncio is single-threaded
- Alternative: dependency injection with Depends() (used more in Day 3+)
"""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.services.generation import GenerationService
from app.services.ingestion import IngestionService
from app.services.retrieval import RetrievalService

# ─── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# ─── Lifespan Context Manager ───────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    STARTUP (before yield):
    - Initialize services
    - Create upload directory
    - In Day 3: create DB connection pool
    - In Day 7: verify external dependencies are reachable

    SHUTDOWN (after yield):
    - Clean up resources
    - In Day 3: close DB connection pool
    - In Day 7: flush logs, close monitoring connections

    The `yield` is where the application runs and serves requests.
    Everything before it = startup. Everything after it = shutdown.
    """
    logger.info("🚀 Starting %s v%s", settings.app_name, settings.app_version)

    # Initialize services
    ingestion_service = IngestionService()
    await ingestion_service.initialize()

    retrieval_service = RetrievalService(ingestion_service)
    generation_service = GenerationService()

    # Store services on app.state for access in route handlers
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.state.generation_service = generation_service

    logger.info("✅ All services initialized")

    yield  # ← Application runs here, serving requests

    # Shutdown cleanup
    logger.info("🛑 Shutting down %s...", settings.app_name)
    # Day 3: await db_pool.close()
    # Day 7: await monitoring.flush()
    logger.info("👋 Shutdown complete")


# ─── FastAPI App ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "A production-quality RAG knowledge assistant. "
        "Upload documents, ask questions, get answers backed by your data."
    ),
    lifespan=lifespan,
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc at /redoc
    openapi_url="/openapi.json",
)

# ─── CORS Middleware ─────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include Routers ─────────────────────────────────────────────────────────

app.include_router(router)


# ─── Entry Point ─────────────────────────────────────────────────────────────


def start() -> None:
    """Start the application via uvicorn.

    This is the entry point for the `knowledge-assistant` script
    defined in pyproject.toml [project.scripts].

    UVICORN:
    - ASGI server (Async Server Gateway Interface)
    - Uses uvloop (C implementation of asyncio) for performance
    - --reload watches files and restarts on changes (dev only)
    - Workers: each worker is a separate process (use for CPU-bound work)
    """
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    start()
