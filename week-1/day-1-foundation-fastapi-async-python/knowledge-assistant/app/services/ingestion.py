"""
Async document ingestion pipeline.

THIS IS THE CORE OF DAY 1 — demonstrates async Python patterns:

ASYNC/AWAIT CONCEPTS:
- asyncio event loop: a single-threaded loop that manages many concurrent tasks
- async def: marks a function as a coroutine (can be paused/resumed)
- await: pauses this coroutine, lets the event loop run others
- asyncio.create_task(): schedules a coroutine to run in the background
- asyncio.gather(): runs multiple coroutines concurrently, waits for all

HOW FASTAPI USES ASYNCIO:
- FastAPI runs on uvicorn, which uses uvloop (C implementation of asyncio)
- Each request is a coroutine — hundreds can run concurrently on one thread
- When one request awaits (DB query, API call), others run
- Your Tornado experience maps directly: Tornado's IOLoop IS an event loop

PIPELINE DESIGN:
1. Save uploaded file to disk
2. Extract text from file (text/markdown for now)
3. Split text into chunks (fixed-size with overlap)
4. Generate chunk IDs (deterministic via hashing)
5. Store chunks in memory (PGVector in Day 3)

The pipeline runs as a background task — the upload endpoint returns
immediately with a document_id, and processing happens async.
"""

import asyncio
import hashlib
import logging
import time
from pathlib import Path
from uuid import UUID, uuid4

from app.config import settings
from app.models.schemas import DocumentStatus

logger = logging.getLogger(__name__)


class DocumentRecord:
    """Tracks the state of a document through the processing pipeline.

    This is a simple mutable object (not Pydantic) because we update it
    in-place during processing. In production, this would be a DB row.
    """

    def __init__(self, document_id: UUID, filename: str, file_path: Path) -> None:
        self.document_id = document_id
        self.filename = filename
        self.file_path = file_path
        self.status = DocumentStatus.PENDING
        self.chunks: list[dict] = []
        self.chunks_created = 0
        self.processing_time: float | None = None
        self.error: str | None = None
        self.created_at = time.time()


class IngestionService:
    """Async document ingestion and chunking service.

    Manages the full lifecycle: upload → extract → chunk → store.
    Uses in-memory storage for now (replaced with PGVector on Day 3).
    """

    def __init__(self) -> None:
        # In-memory stores — will be replaced with database in Day 3
        self.documents: dict[UUID, DocumentRecord] = {}
        self.chunks: dict[str, dict] = {}  # chunk_id → chunk data

    async def initialize(self) -> None:
        """Create the upload directory if it doesn't exist.

        Called during FastAPI startup via the lifespan context manager.
        """
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Ingestion service initialized. Upload dir: %s", settings.upload_dir)

    async def ingest_document(
        self,
        file_content: bytes,
        filename: str,
    ) -> DocumentRecord:
        """Accept a document for processing.

        This returns immediately — actual processing happens in the background
        via asyncio.create_task(). The caller gets a document_id to check status.

        WHY BACKGROUND TASKS?
        - File processing can take seconds to minutes
        - HTTP request should return fast (< 500ms)
        - Client polls /documents/{id}/status for progress
        - Alternative: FastAPI BackgroundTasks, but create_task gives more control
        """
        document_id = uuid4()

        # Save file to disk
        file_path = settings.upload_dir / f"{document_id}_{filename}"
        await self._save_file(file_path, file_content)

        # Create tracking record
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            file_path=file_path,
        )
        self.documents[document_id] = record

        # Schedule processing as a background task
        # asyncio.create_task() adds the coroutine to the event loop's queue
        # It will run when the current handler yields (at next await)
        asyncio.create_task(self._process_document(record))

        logger.info("Document queued for processing: %s (%s)", filename, document_id)
        return record

    async def get_document_status(self, document_id: UUID) -> DocumentRecord | None:
        """Look up a document's processing status."""
        return self.documents.get(document_id)

    def get_all_chunks(self) -> list[dict]:
        """Return all stored chunks. Used by RetrievalService for search."""
        return list(self.chunks.values())

    # ─── Private Pipeline Methods ────────────────────────────────────────────

    async def _save_file(self, file_path: Path, content: bytes) -> None:
        """Save uploaded file content to disk.

        NOTE: This uses synchronous file I/O wrapped in an async function.
        For true async file I/O, use aiofiles library. For Day 1, this is fine
        because file writes are fast for small files. In production with large
        files, you'd use:
            async with aiofiles.open(path, 'wb') as f:
                await f.write(content)
        """
        file_path.write_bytes(content)
        logger.debug("Saved file: %s (%d bytes)", file_path, len(content))

    async def _process_document(self, record: DocumentRecord) -> None:
        """Full processing pipeline for a single document.

        This runs as a background task after the HTTP response is sent.
        Any exception here won't crash the server — it just updates the
        document status to FAILED.
        """
        start_time = time.time()
        record.status = DocumentStatus.PROCESSING

        try:
            # Step 1: Extract text from the file
            text = await self._extract_text(record.file_path)
            logger.info("Extracted %d characters from %s", len(text), record.filename)

            # Step 2: Split into chunks
            chunks = await self._chunk_text(
                text=text,
                chunk_size=settings.default_chunk_size,
                chunk_overlap=settings.default_chunk_overlap,
            )
            logger.info("Created %d chunks from %s", len(chunks), record.filename)

            # Step 3: Generate IDs and enrich chunks with metadata
            enriched_chunks = await self._enrich_chunks(
                chunks=chunks,
                document_id=record.document_id,
                filename=record.filename,
            )

            # Step 4: Store chunks (in-memory for now, PGVector in Day 3)
            for chunk in enriched_chunks:
                self.chunks[chunk["chunk_id"]] = chunk

            # Update record
            record.chunks = enriched_chunks
            record.chunks_created = len(enriched_chunks)
            record.status = DocumentStatus.COMPLETED
            record.processing_time = time.time() - start_time

            logger.info(
                "Successfully processed %s: %d chunks in %.2fs",
                record.filename,
                record.chunks_created,
                record.processing_time,
            )

        except Exception as exc:
            record.status = DocumentStatus.FAILED
            record.error = str(exc)
            record.processing_time = time.time() - start_time
            logger.error("Failed to process %s: %s", record.filename, exc, exc_info=True)

    async def _extract_text(self, file_path: Path) -> str:
        """Extract text content from a file.

        Currently supports plain text and markdown files.
        PDF support will be added in Day 2 with pypdf or pdfplumber.

        WHY ASYNC HERE?
        Even though file I/O is sync, making this async lets us swap in
        an async implementation later (e.g., aiofiles) without changing
        the pipeline interface.
        """
        suffix = file_path.suffix.lower()

        if suffix in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    async def _chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> list[str]:
        """Split text into overlapping chunks of approximately chunk_size characters.

        CHUNKING STRATEGY: Fixed-size with overlap
        - Simple and predictable
        - Overlap ensures context isn't lost at chunk boundaries
        - Day 2 will add: recursive chunking and semantic chunking

        WHY OVERLAP?
        If a sentence spans two chunks, overlap ensures both chunks
        contain the full sentence. Without overlap, retrieval might
        return a chunk that starts mid-sentence.

        IMPORTANT: In production, chunk size should be measured in TOKENS,
        not characters (using tiktoken). 512 chars ≈ 128 tokens for English.
        We'll switch to token-based chunking in Day 2.
        """
        if not text.strip():
            return []

        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size

            # If not at the end, try to break at a sentence boundary
            if end < text_length:
                # Look for the last period, newline, or space within the chunk
                for boundary_char in [".\n", "\n\n", "\n", ". ", " "]:
                    boundary = text.rfind(boundary_char, start, end)
                    if boundary > start:
                        end = boundary + len(boundary_char)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward, accounting for overlap
            start = end - chunk_overlap if end < text_length else text_length

        return chunks

    async def _enrich_chunks(
        self,
        chunks: list[str],
        document_id: UUID,
        filename: str,
    ) -> list[dict]:
        """Add metadata and deterministic IDs to each chunk.

        USES asyncio.gather() TO PROCESS CHUNKS CONCURRENTLY:
        - Each chunk enrichment is independent
        - gather() runs them all concurrently on the event loop
        - With real embedding generation (Day 2), this means parallel API calls

        DETERMINISTIC IDS:
        - Same content → same chunk_id (via SHA-256 hash)
        - Enables deduplication and idempotent re-processing
        - If you re-upload the same doc, chunks won't be duplicated
        """

        async def enrich_single(index: int, content: str) -> dict:
            chunk_id = self._generate_chunk_id(content, str(document_id))

            # Simulate embedding generation delay (replaced with real embeddings Day 2)
            # In production: await embedding_model.encode(content)
            await asyncio.sleep(0.01)  # Simulates API call latency

            return {
                "chunk_id": chunk_id,
                "content": content,
                "document_id": str(document_id),
                "filename": filename,
                "chunk_index": index,
                "metadata": {
                    "char_count": len(content),
                    "source": filename,
                },
            }

        # asyncio.gather() runs all enrichment tasks concurrently
        # The * unpacks the list into positional arguments
        tasks = [enrich_single(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        return list(results)

    @staticmethod
    def _generate_chunk_id(content: str, document_id: str) -> str:
        """Generate a deterministic chunk ID from content + document.

        Uses SHA-256 hash truncated to 16 hex chars.
        Same content from same document → same ID every time.
        """
        hash_input = f"{document_id}:{content}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
