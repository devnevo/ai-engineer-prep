"""
Search/retrieval service — finds relevant chunks for a query.

DAY 1: Simple keyword matching against in-memory chunks.
DAY 3: Will be upgraded to hybrid search (vector + BM25) with PGVector.

DESIGN PATTERN:
- The service exposes a clean interface: search(query, top_k) → results
- Callers don't care HOW search works internally
- When we swap in vector search on Day 3, the API routes don't change
- This is the Strategy pattern — the algorithm is interchangeable
"""

import logging
import time

from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)


class RetrievalService:
    """Retrieval service that searches document chunks.

    Currently uses simple keyword matching. Will be upgraded to:
    - Vector similarity search (Day 3)
    - BM25 keyword search (Day 3)
    - Hybrid search with RRF fusion (Day 3)
    - Reranking (Day 3)
    """

    def __init__(self, ingestion_service) -> None:  # noqa: ANN001
        """Initialize with a reference to the ingestion service for chunk access.

        In Day 3, this will take a database connection pool instead.
        """
        self.ingestion_service = ingestion_service

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> tuple[list[SearchResult], float]:
        """Search for chunks relevant to the query.

        Returns:
            Tuple of (results, processing_time_seconds)

        The simple keyword matching works like this:
        1. Lowercase the query and split into words
        2. For each chunk, count how many query words appear in it
        3. Score = matching words / total query words
        4. Sort by score, return top_k

        This is basically a naive TF scorer. Day 3 replaces this with
        proper BM25 + vector search.
        """
        start_time = time.time()

        chunks = self.ingestion_service.get_all_chunks()
        if not chunks:
            return [], time.time() - start_time

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_chunks: list[tuple[dict, float]] = []

        for chunk in chunks:
            content_lower = chunk["content"].lower()

            # Count matching query words in this chunk
            matching = sum(1 for word in query_words if word in content_lower)
            if matching > 0:
                score = matching / len(query_words)

                # Apply metadata filters if provided
                if filters:
                    chunk_meta = chunk.get("metadata", {})
                    if not all(chunk_meta.get(k) == v for k, v in filters.items()):
                        continue

                scored_chunks.append((chunk, score))

        # Sort by score descending, take top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_results = scored_chunks[:top_k]

        results = [
            SearchResult(
                content=chunk["content"],
                score=round(score, 4),
                source=chunk.get("filename"),
                metadata=chunk.get("metadata", {}),
            )
            for chunk, score in top_results
        ]

        processing_time = time.time() - start_time
        logger.info(
            "Search for '%s': %d results in %.3fs",
            query[:50],
            len(results),
            processing_time,
        )

        return results, processing_time
