"""
API endpoint tests — verify HTTP interface behavior.

TESTING PHILOSOPHY:
- Test the contract (HTTP status codes, response shapes), not implementation details
- Each test is independent — fixtures reset state between tests
- Use descriptive test names that read like specifications

WHAT WE TEST:
1. Health check returns 200 with expected fields
2. Query endpoint validates input (422 on invalid)
3. Query endpoint returns results when documents exist
4. Document upload returns 201 with document_id
5. Document status returns 404 for unknown documents
6. Invalid file types are rejected with 422
"""

import asyncio
from pathlib import Path

import httpx


class TestHealthCheck:
    """Tests for GET /health endpoint."""

    async def test_health_returns_200(self, client: httpx.AsyncClient):
        """Health check should return 200 with status 'healthy'."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert "checks" in data

    async def test_health_includes_dependency_checks(self, client: httpx.AsyncClient):
        """Health check should report status of each dependency."""
        response = await client.get("/health")
        data = response.json()

        checks = data["checks"]
        assert "ingestion_service" in checks
        assert checks["ingestion_service"]["status"] == "healthy"


class TestQueryEndpoint:
    """Tests for POST /query endpoint."""

    async def test_query_returns_200(self, client: httpx.AsyncClient):
        """Query with valid input should return 200."""
        response = await client.post(
            "/query",
            json={"query": "test query", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert "results" in data
        assert "total_results" in data
        assert "processing_time_seconds" in data

    async def test_query_empty_returns_no_results(self, client: httpx.AsyncClient):
        """Query with no documents indexed should return empty results."""
        response = await client.post(
            "/query",
            json={"query": "anything"},
        )

        data = response.json()
        assert data["total_results"] == 0
        assert data["results"] == []

    async def test_query_validation_rejects_empty(self, client: httpx.AsyncClient):
        """Empty query string should return 422 validation error."""
        response = await client.post(
            "/query",
            json={"query": ""},
        )

        assert response.status_code == 422

    async def test_query_validation_rejects_invalid_top_k(self, client: httpx.AsyncClient):
        """top_k outside allowed range should return 422."""
        response = await client.post(
            "/query",
            json={"query": "test", "top_k": 100},
        )

        assert response.status_code == 422

    async def test_query_finds_uploaded_document(
        self, client: httpx.AsyncClient, sample_text_file: Path,
    ):
        """Query should find content from a previously uploaded document."""
        # Upload a document first
        with open(sample_text_file, "rb") as f:
            upload_response = await client.post(
                "/documents/upload",
                files={"file": ("sample.txt", f, "text/plain")},
            )
        assert upload_response.status_code == 201

        # Wait for async processing to complete
        await asyncio.sleep(0.5)

        # Query for content from the uploaded document
        query_response = await client.post(
            "/query",
            json={"query": "fox jumps lazy dog"},
        )

        data = query_response.json()
        assert data["total_results"] > 0
        assert any("fox" in r["content"].lower() for r in data["results"])


class TestDocumentUpload:
    """Tests for POST /documents/upload endpoint."""

    async def test_upload_returns_201(self, client: httpx.AsyncClient, sample_text_file: Path):
        """Successful upload should return 201 Created."""
        with open(sample_text_file, "rb") as f:
            response = await client.post(
                "/documents/upload",
                files={"file": ("sample.txt", f, "text/plain")},
            )

        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "sample.txt"
        assert data["status"] == "pending"

    async def test_upload_markdown_file(
        self, client: httpx.AsyncClient, sample_markdown_file: Path,
    ):
        """Markdown files should be accepted."""
        with open(sample_markdown_file, "rb") as f:
            response = await client.post(
                "/documents/upload",
                files={"file": ("sample.md", f, "text/markdown")},
            )

        assert response.status_code == 201

    async def test_upload_rejects_invalid_extension(
        self, client: httpx.AsyncClient, tmp_path: Path,
    ):
        """Files with disallowed extensions should return 422."""
        bad_file = tmp_path / "script.exe"
        bad_file.write_text("not a real exe")

        with open(bad_file, "rb") as f:
            response = await client.post(
                "/documents/upload",
                files={"file": ("script.exe", f, "application/octet-stream")},
            )

        assert response.status_code == 422
        assert "not allowed" in response.json()["detail"].lower()


class TestDocumentStatus:
    """Tests for GET /documents/{document_id}/status endpoint."""

    async def test_status_returns_404_for_unknown(self, client: httpx.AsyncClient):
        """Unknown document_id should return 404."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = await client.get(f"/documents/{fake_id}/status")

        assert response.status_code == 404

    async def test_status_returns_valid_uuid_error(self, client: httpx.AsyncClient):
        """Invalid UUID format should return 422."""
        response = await client.get("/documents/not-a-uuid/status")

        assert response.status_code == 422

    async def test_status_after_upload(self, client: httpx.AsyncClient, sample_text_file: Path):
        """Status should be retrievable after uploading a document."""
        with open(sample_text_file, "rb") as f:
            upload_response = await client.post(
                "/documents/upload",
                files={"file": ("sample.txt", f, "text/plain")},
            )

        document_id = upload_response.json()["document_id"]

        # Wait briefly for processing
        await asyncio.sleep(0.5)

        status_response = await client.get(f"/documents/{document_id}/status")

        assert status_response.status_code == 200
        data = status_response.json()
        assert data["document_id"] == document_id
        assert data["status"] in ["pending", "processing", "completed"]
