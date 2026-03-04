"""
Pytest fixtures and configuration for async testing.

TESTING CONCEPTS:
- pytest-asyncio: enables `async def test_...` functions
- asyncio_mode = "auto": all async tests run automatically (no @pytest.mark.asyncio needed)
- httpx.AsyncClient + ASGITransport: sends requests to the FastAPI app in-process
  (no real HTTP server needed — faster and more reliable than TestClient)
- Fixtures with `yield`: setup runs before yield, teardown after

WHY httpx.AsyncClient INSTEAD OF TestClient?
- TestClient is synchronous — doesn't test your async code paths properly
- AsyncClient speaks ASGI directly — same protocol as uvicorn
- You're testing the REAL async handlers, not sync wrappers
"""

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from app.config import settings
from app.main import app, lifespan


@pytest.fixture(autouse=True)
def setup_upload_dir(tmp_path: Path):
    """Override the upload directory to use a temp dir for each test.

    autouse=True means this runs for EVERY test automatically.
    tmp_path is a built-in pytest fixture that creates a unique temp dir per test.
    """
    original_dir = settings.upload_dir
    settings.upload_dir = tmp_path / "uploads"
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    yield
    settings.upload_dir = original_dir


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """Create an async HTTP client with the FastAPI lifespan properly triggered.

    KEY INSIGHT: ASGITransport doesn't automatically trigger the lifespan.
    We must manually enter the lifespan context manager so that app.state
    gets populated with the initialized services (ingestion, retrieval, etc).
    """
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as ac:
            yield ac


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file for upload testing."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "The quick brown fox jumps over the lazy dog. "
        "This is a sample document for testing the knowledge assistant. "
        "It contains multiple sentences to test chunking behavior. "
        "Each chunk should contain meaningful text segments."
    )
    return file_path


@pytest.fixture
def sample_markdown_file(tmp_path: Path) -> Path:
    """Create a sample markdown file for upload testing."""
    file_path = tmp_path / "sample.md"
    file_path.write_text(
        "# Introduction\n\n"
        "This is a sample markdown document about machine learning.\n\n"
        "## What is Machine Learning?\n\n"
        "Machine learning is a subset of artificial intelligence that "
        "enables systems to learn from data and improve from experience "
        "without being explicitly programmed.\n\n"
        "## Types of Machine Learning\n\n"
        "There are three main types:\n"
        "1. Supervised learning\n"
        "2. Unsupervised learning\n"
        "3. Reinforcement learning\n\n"
        "Each type has its own strengths and use cases in the real world."
    )
    return file_path
