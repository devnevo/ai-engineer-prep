# Knowledge Assistant — Day 1 Foundation

A production-quality RAG knowledge assistant built with **FastAPI** and **async Python**.  
Part of the [AI Engineer Prep](../../README.md) 4-week learning plan.

## Architecture

```
knowledge-assistant/
├── app/
│   ├── main.py              # FastAPI app with lifespan, CORS, logging
│   ├── config.py             # Pydantic Settings (env-based config)
│   ├── models/
│   │   └── schemas.py        # Request/Response Pydantic models
│   ├── api/
│   │   └── routes.py         # HTTP endpoints (health, query, upload)
│   └── services/
│       ├── ingestion.py      # Async document processing pipeline
│       ├── retrieval.py      # Search (keyword for now, vector Day 3)
│       └── generation.py     # LLM response (template, real LLM Day 4)
├── tests/
│   ├── conftest.py           # Fixtures (async client, sample files)
│   ├── test_api.py           # API endpoint tests
│   └── test_ingestion.py     # Ingestion pipeline unit tests
├── Dockerfile                # Multi-stage build, non-root user
├── docker-compose.yml        # Local dev (+ PGVector/Redis placeholders)
└── pyproject.toml            # Python packaging & dependencies
```

## Quick Start

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in dev mode
pip install -e ".[dev]"

# Run the server
uvicorn app.main:app --reload

# Open API docs
open http://127.0.0.1:8000/docs
```

## API Endpoints

| Method | Path | Description | Status Code |
|--------|------|-------------|-------------|
| `GET` | `/health` | Service health + dependency checks | 200 |
| `POST` | `/query` | Search documents & get answers | 200 |
| `POST` | `/documents/upload` | Upload a document for processing | 201 |
| `GET` | `/documents/{id}/status` | Check processing status | 200 / 404 |

## Running Tests

```bash
# Lint
ruff check app/ tests/

# Tests
pytest tests/ -v
```

## Concepts Learned

- **Async/Await**: `asyncio.create_task()`, `asyncio.gather()`, background processing
- **Pydantic v2**: `BaseModel`, `Field()` constraints, auto OpenAPI docs
- **FastAPI**: lifespan context manager, CORS, `UploadFile`, path parameters
- **HTTP**: Status codes (200/201/404/422), `multipart/form-data`, validation errors
- **Project Structure**: `pyproject.toml`, proper Python packaging, service layer pattern
- **Testing**: `pytest-asyncio`, `httpx.AsyncClient` with `ASGITransport`

## What's Next

- **Day 2**: Document loaders (PDF), chunking strategies, embeddings
- **Day 3**: PGVector, hybrid search (vector + BM25), connection pooling
- **Day 4**: LLM integration, streaming, structured extraction
