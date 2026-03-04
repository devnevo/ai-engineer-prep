"""
API route definitions — the HTTP interface to the knowledge assistant.

HTTP CONCEPTS DEMONSTRATED:
- Status codes: 200 (OK), 201 (Created), 404 (Not Found), 422 (Validation Error)
- Content-Type: application/json (default) vs multipart/form-data (file upload)
- FastAPI automatically returns 422 when Pydantic validation fails

FASTAPI ROUTE ANATOMY:
- @router.get("/path") → decorator registers the function as a GET handler
- response_model=HealthResponse → FastAPI serializes the return value through this model
- status_code=201 → override the default 200 for creation endpoints

FILE UPLOADS:
- UploadFile is FastAPI's wrapper around Starlette's uploaded file
- Content-Type must be multipart/form-data (not application/json)
- file.read() gives bytes, file.filename gives the original name
- Large files should use streaming (file.read(chunk_size) in a loop)

DEPENDENCY INJECTION:
- FastAPI's Depends() lets you inject services, DB connections, auth, etc.
- Here we use the lifespan-initialized services from app.state
- Day 3 will use Depends() for database session injection
"""

import logging
import time
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, UploadFile, status

from app.config import settings
from app.models.schemas import (
    DocumentStatusResponse,
    DocumentUploadResponse,
    HealthCheck,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Track when the server started for uptime calculation
_start_time = time.time()


# ─── Health Check ────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns the health status of the service and its dependencies.",
    tags=["System"],
)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint — critical for Kubernetes probes.

    Kubernetes uses this for:
    - Liveness probe: is the process alive? (restart if not)
    - Readiness probe: can it serve traffic? (remove from load balancer if not)

    We check each dependency and report individual status.
    Day 7 will add checks for PGVector and Redis.
    """
    checks: dict[str, HealthCheck] = {}

    # Check ingestion service is initialized
    ingestion = request.app.state.ingestion_service
    checks["ingestion_service"] = HealthCheck(
        status="healthy" if ingestion is not None else "unhealthy",
        latency_ms=0.0,
    )

    # Check upload directory is writable
    try:
        test_file = settings.upload_dir / ".health_check"
        test_file.touch()
        test_file.unlink()
        checks["upload_dir"] = HealthCheck(status="healthy", latency_ms=0.0)
    except OSError:
        checks["upload_dir"] = HealthCheck(status="unhealthy")

    # Overall status: degraded if any check is unhealthy
    overall = "healthy" if all(c.status == "healthy" for c in checks.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.now(tz=UTC),
        checks=checks,
    )


# ─── Query Endpoint ─────────────────────────────────────────────────────────


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the knowledge base",
    description="Search for relevant documents and generate an answer.",
    tags=["Query"],
)
async def query(request: Request, body: QueryRequest) -> QueryResponse:
    """Query endpoint — the main interface for asking questions.

    Flow:
    1. Search for relevant chunks (RetrievalService)
    2. Generate answer from context (GenerationService)
    3. Return results with processing time

    FastAPI automatically:
    - Parses the JSON body into a QueryRequest (validated by Pydantic)
    - Returns 422 if validation fails (e.g., empty query, top_k > 20)
    - Serializes the return dict through QueryResponse model
    """
    retrieval = request.app.state.retrieval_service

    # Search for relevant chunks
    results, search_time = await retrieval.search(
        query=body.query,
        top_k=body.top_k,
        filters=body.filters,
    )

    return QueryResponse(
        query=body.query,
        results=results,
        total_results=len(results),
        processing_time_seconds=round(search_time, 4),
    )


# ─── Document Upload ────────────────────────────────────────────────────────


@router.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document",
    description="Upload a document for processing and indexing.",
    tags=["Documents"],
)
async def upload_document(
    request: Request,
    file: UploadFile,
) -> DocumentUploadResponse:
    """Upload a document for async processing.

    Content-Type must be multipart/form-data (not application/json).
    This is because we're sending binary file data, not JSON.

    The endpoint:
    1. Validates file type and size
    2. Reads file content into memory
    3. Queues the document for background processing
    4. Returns immediately with a document_id

    Client can then poll GET /documents/{document_id}/status for progress.

    WHY 201 CREATED?
    - 200 OK = request succeeded, here's the result
    - 201 Created = request succeeded AND a new resource was created
    - The document is a new resource with its own URI
    """
    # Validate file extension
    filename = file.filename or "unknown"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File type '{suffix}' not allowed. Allowed: {settings.allowed_extensions}",
        )

    # Read file content
    content = await file.read()

    # Validate file size
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB",
        )

    # Queue for processing
    ingestion = request.app.state.ingestion_service
    record = await ingestion.ingest_document(
        file_content=content,
        filename=filename,
    )

    return DocumentUploadResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status,
        message="Document queued for processing. Check status at /documents/{document_id}/status",
    )


# ─── Document Status ────────────────────────────────────────────────────────


@router.get(
    "/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Check document processing status",
    description="Get the current processing status of an uploaded document.",
    tags=["Documents"],
)
async def get_document_status(
    request: Request,
    document_id: UUID,
) -> DocumentStatusResponse:
    """Check the processing status of an uploaded document.

    Path parameters (like document_id) are automatically parsed by FastAPI.
    UUID type ensures the path parameter is a valid UUID — returns 422 otherwise.
    """
    ingestion = request.app.state.ingestion_service
    record = await ingestion.get_document_status(document_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return DocumentStatusResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status,
        chunks_created=record.chunks_created,
        processing_time_seconds=(
            round(record.processing_time, 4) if record.processing_time else None
        ),
        error=record.error,
    )
