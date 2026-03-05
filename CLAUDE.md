# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

A 4-week structured learning plan: **From Senior Backend Engineer to AI/ML Engineer**. Each day produces a working project pushed to GitHub. The plan follows a build-first approach — all learning happens through building, not reading.

**Weekly Themes:**
- Week 1: RAG System + Python Mastery
- Week 2: Training + Fine-tuning
- Week 3: Production ML Systems
- Week 4: Advanced AI Engineering

## Project Structure

Each day's project lives at `week-{N}/day-{N}-{topic}/{project-name}/`. Currently implemented:

- `week-1/day-1-foundation-fastapi-async-python/knowledge-assistant/` — FastAPI RAG foundation

### knowledge-assistant Architecture

```
app/
├── main.py           # FastAPI app, lifespan context manager, CORS, logging
├── config.py         # Pydantic Settings singleton (env-based)
├── models/schemas.py # Request/response Pydantic models
├── api/routes.py     # HTTP endpoints: /health, /query, /documents/upload
└── services/
    ├── ingestion.py  # Async doc pipeline: save → extract → chunk → store
    ├── retrieval.py  # Search service (keyword now, vector+BM25 in Day 3)
    └── generation.py # LLM response generation (template now, real LLM Day 4)
```

The services are designed with the Strategy pattern — `RetrievalService` and `IngestionService` expose stable interfaces so the underlying implementation can be swapped (e.g., in-memory → PGVector) without changing routes.

## Commands

All commands run from within a project directory (e.g., `week-1/day-1-.../knowledge-assistant/`).

```bash
# Setup (first time)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run dev server
uvicorn app.main:app --reload

# Lint
ruff check app/ tests/

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_api.py -v

# Run a single test
pytest tests/test_api.py::test_health_endpoint -v
```

## Python/Stack Conventions

- **Python 3.11+**, FastAPI, Pydantic v2, pydantic-settings
- **Packaging**: `pyproject.toml` with setuptools; install with `pip install -e ".[dev]"`
- **Linting**: `ruff` (line-length 100, rules: E, F, I, N, W, UP, B, A, SIM)
- **Testing**: `pytest-asyncio` in auto mode; use `httpx.AsyncClient` with `ASGITransport` for API tests
- **Config**: All settings via `app/config.py` `Settings` singleton — never hardcode values
- **Background tasks**: Use `asyncio.create_task()` for fire-and-forget processing; HTTP endpoints return immediately with an ID for polling
- **Chunk IDs**: SHA-256 hash of `{document_id}:{content}` (deterministic, enables deduplication)

## Planned Upgrades (per the 4-week plan)

- Day 2: PDF loading, token-based chunking, real embeddings
- Day 3: PGVector, hybrid search (vector + BM25 + RRF), connection pooling
- Day 4: Real LLM integration, streaming responses, structured extraction
- Day 5: Evaluation framework, observability
