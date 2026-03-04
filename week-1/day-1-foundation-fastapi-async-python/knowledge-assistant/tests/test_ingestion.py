"""
Unit tests for the ingestion service — tests the processing pipeline in isolation.

These test the LOGIC, not the HTTP interface:
- Text extraction from different file types
- Chunking behavior with various sizes and overlaps
- Chunk ID determinism
- Full pipeline processing
"""

import asyncio
from pathlib import Path

import pytest

from app.services.ingestion import IngestionService


@pytest.fixture
def ingestion_service() -> IngestionService:
    """Create a fresh IngestionService for each test."""
    return IngestionService()


class TestTextExtraction:
    """Tests for the _extract_text method."""

    async def test_extract_text_file(self, ingestion_service: IngestionService, tmp_path: Path):
        """Should extract text from .txt files."""
        file = tmp_path / "test.txt"
        file.write_text("Hello, World!")

        text = await ingestion_service._extract_text(file)

        assert text == "Hello, World!"

    async def test_extract_markdown_file(self, ingestion_service: IngestionService, tmp_path: Path):
        """Should extract text from .md files."""
        file = tmp_path / "test.md"
        file.write_text("# Title\n\nContent here")

        text = await ingestion_service._extract_text(file)

        assert "# Title" in text
        assert "Content here" in text

    async def test_extract_unsupported_type_raises(
        self, ingestion_service: IngestionService, tmp_path: Path,
    ):
        """Unsupported file types should raise ValueError."""
        file = tmp_path / "test.xyz"
        file.write_text("some content")

        with pytest.raises(ValueError, match="Unsupported file type"):
            await ingestion_service._extract_text(file)


class TestChunking:
    """Tests for the _chunk_text method."""

    async def test_short_text_single_chunk(self, ingestion_service: IngestionService):
        """Text shorter than chunk_size should produce one chunk."""
        chunks = await ingestion_service._chunk_text(
            "Short text.", chunk_size=100, chunk_overlap=10,
        )

        assert len(chunks) == 1
        assert chunks[0] == "Short text."

    async def test_empty_text_no_chunks(self, ingestion_service: IngestionService):
        """Empty text should produce no chunks."""
        chunks = await ingestion_service._chunk_text("", chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 0

    async def test_whitespace_only_no_chunks(self, ingestion_service: IngestionService):
        """Whitespace-only text should produce no chunks."""
        chunks = await ingestion_service._chunk_text("   \n\n  ", chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 0

    async def test_long_text_multiple_chunks(self, ingestion_service: IngestionService):
        """Text longer than chunk_size should produce multiple chunks."""
        # Create text that's ~300 chars
        text = "This is a sentence. " * 15  # ~300 chars

        chunks = await ingestion_service._chunk_text(text, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 1
        # Each chunk should be non-empty
        assert all(len(c) > 0 for c in chunks)

    async def test_chunks_have_overlap(self, ingestion_service: IngestionService):
        """Adjacent chunks should have overlapping content."""
        text = "Word " * 100  # Simple repeating text

        chunks = await ingestion_service._chunk_text(text, chunk_size=50, chunk_overlap=20)

        # With overlap, neighboring chunks should share some content
        assert len(chunks) > 2  # enough chunks to verify overlap


class TestChunkIdGeneration:
    """Tests for deterministic chunk ID generation."""

    def test_same_input_same_id(self):
        """Same content + document should always produce the same ID."""
        id1 = IngestionService._generate_chunk_id("hello world", "doc-1")
        id2 = IngestionService._generate_chunk_id("hello world", "doc-1")

        assert id1 == id2

    def test_different_content_different_id(self):
        """Different content should produce different IDs."""
        id1 = IngestionService._generate_chunk_id("hello world", "doc-1")
        id2 = IngestionService._generate_chunk_id("goodbye world", "doc-1")

        assert id1 != id2

    def test_different_document_different_id(self):
        """Same content but different document should produce different IDs."""
        id1 = IngestionService._generate_chunk_id("hello world", "doc-1")
        id2 = IngestionService._generate_chunk_id("hello world", "doc-2")

        assert id1 != id2

    def test_id_is_hex_string(self):
        """Chunk ID should be a 16-character hex string."""
        chunk_id = IngestionService._generate_chunk_id("test", "doc")

        assert len(chunk_id) == 16
        assert all(c in "0123456789abcdef" for c in chunk_id)


class TestFullPipeline:
    """End-to-end tests of the ingestion pipeline."""

    async def test_ingest_document_completes(
        self, ingestion_service: IngestionService, tmp_path: Path,
    ):
        """Ingesting a document should eventually complete processing."""
        # Setup
        await ingestion_service.initialize()

        from app.config import settings
        original = settings.upload_dir
        settings.upload_dir = tmp_path
        settings.upload_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_content = b"This is a test document with enough content to create chunks."
            record = await ingestion_service.ingest_document(file_content, "test.txt")

            # Wait for async processing
            await asyncio.sleep(1.0)

            assert record.status.value == "completed"
            assert record.chunks_created > 0
            assert record.processing_time is not None
        finally:
            settings.upload_dir = original

    async def test_chunks_are_searchable(self, ingestion_service: IngestionService, tmp_path: Path):
        """After ingestion, chunks should be available via get_all_chunks()."""
        await ingestion_service.initialize()

        from app.config import settings
        original = settings.upload_dir
        settings.upload_dir = tmp_path
        settings.upload_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_content = b"Machine learning is a subset of artificial intelligence."
            await ingestion_service.ingest_document(file_content, "ml.txt")

            # Wait for processing
            await asyncio.sleep(1.0)

            chunks = ingestion_service.get_all_chunks()
            assert len(chunks) > 0
            assert any("machine learning" in c["content"].lower() for c in chunks)
        finally:
            settings.upload_dir = original
