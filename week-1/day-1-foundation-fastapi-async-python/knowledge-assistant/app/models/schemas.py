"""
Pydantic models (schemas) for request/response validation.

WHY PYDANTIC FOR API SCHEMAS?
- FastAPI uses Pydantic models to automatically:
  1. Validate incoming request data (returns 422 if invalid)
  2. Serialize response data to JSON
  3. Generate OpenAPI/Swagger documentation
- Field() adds constraints (min_length, ge, le) AND documentation

KEY CONCEPTS:
- BaseModel: immutable data container with validation
- Field(): metadata + constraints for each field
- Generic types: list[str], dict[str, Any], Optional[str] (same as str | None)
- Enum: type-safe constants that show up as dropdowns in Swagger UI

PYDANTIC V2 PERFORMANCE:
- Core validation is written in Rust (via pydantic-core)
- 5-50x faster than Pydantic v1
- model_validate() replaces .parse_obj()
- model_dump() replaces .dict()
"""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

# ─── Enums ────────────────────────────────────────────────────────────────────


class DocumentStatus(StrEnum):
    """Processing status of an uploaded document.

    Inheriting from str makes the enum JSON-serializable automatically.
    FastAPI will show these as a dropdown in Swagger UI.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ─── Request Models ──────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for the /query endpoint.

    Example:
        {
            "query": "What is retrieval augmented generation?",
            "top_k": 5,
            "filters": {"source": "whitepaper.pdf"}
        }
    """

    query: str = Field(
        ...,  # ... means required (no default)
        min_length=1,
        max_length=1000,
        description="The search query to find relevant documents",
        examples=["What is retrieval augmented generation?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,  # greater than or equal
        le=20,  # less than or equal
        description="Number of results to return",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filters to narrow search",
    )


class DocumentMetadata(BaseModel):
    """Optional metadata to attach to an uploaded document."""

    source: str | None = Field(default=None, description="Source of the document")
    author: str | None = Field(default=None, description="Author of the document")
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")


# ─── Response Models ─────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    """A single search result with content, relevance score, and metadata."""

    content: str = Field(description="The text content of the matched chunk")
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevance score between 0 and 1",
    )
    source: str | None = Field(default=None, description="Source document of this chunk")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the chunk",
    )


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    query: str = Field(description="The original query")
    results: list[SearchResult] = Field(description="List of matching search results")
    total_results: int = Field(description="Total number of results found")
    processing_time_seconds: float = Field(description="Time taken to process the query")


class HealthCheck(BaseModel):
    """Health status of a single dependency."""

    status: str = Field(description="healthy or unhealthy")
    latency_ms: float | None = Field(default=None, description="Response time in milliseconds")


class HealthResponse(BaseModel):
    """Response from the /health endpoint.

    Includes overall status plus individual dependency checks.
    This is critical for Kubernetes readiness/liveness probes (Day 7).
    """

    status: str = Field(description="Overall health: healthy or degraded")
    version: str = Field(description="Application version")
    uptime_seconds: float = Field(description="Time since server started")
    timestamp: datetime = Field(description="Current server time")
    checks: dict[str, HealthCheck] = Field(
        default_factory=dict,
        description="Individual dependency health checks",
    )


class DocumentUploadResponse(BaseModel):
    """Response after uploading a document for processing."""

    document_id: UUID = Field(description="Unique identifier for tracking the document")
    filename: str = Field(description="Original filename")
    status: DocumentStatus = Field(description="Current processing status")
    message: str = Field(description="Human-readable status message")


class DocumentStatusResponse(BaseModel):
    """Response for checking document processing status."""

    document_id: UUID = Field(description="Unique identifier of the document")
    filename: str = Field(description="Original filename")
    status: DocumentStatus = Field(description="Current processing status")
    chunks_created: int = Field(default=0, description="Number of chunks created so far")
    processing_time_seconds: float | None = Field(
        default=None,
        description="Total processing time (null if still processing)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if processing failed",
    )
