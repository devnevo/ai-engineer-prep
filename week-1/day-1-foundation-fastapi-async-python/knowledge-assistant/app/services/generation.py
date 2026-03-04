"""
LLM generation service — produces answers from retrieved context.

DAY 1: Template-based response (no LLM calls yet).
DAY 4: Will be upgraded to real LLM calls with:
  - Streaming responses (Server-Sent Events)
  - Structured output extraction (Pydantic + function calling)
  - Retry logic with exponential backoff
  - httpx.AsyncClient for async HTTP to LLM APIs

DESIGN DECISION:
- Even as a stub, this service demonstrates the interface that Day 4 will implement
- The API route doesn't change when we swap in real LLM generation
- This is Dependency Inversion: routes depend on the interface, not the implementation
"""

import logging

from app.models.schemas import SearchResult

logger = logging.getLogger(__name__)


class GenerationService:
    """Generates answers from retrieved context chunks.

    Currently returns a template response. Will be upgraded in Day 4 to:
    - Call OpenAI/Anthropic APIs via httpx.AsyncClient
    - Stream tokens via FastAPI StreamingResponse
    - Extract structured data via function calling
    """

    async def generate(
        self,
        query: str,
        context_chunks: list[SearchResult],
    ) -> str:
        """Generate an answer based on retrieved context.

        Args:
            query: The user's question
            context_chunks: Retrieved chunks with relevance scores

        Returns:
            Generated answer text

        In Day 4, this becomes:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={...},
                    headers={"Authorization": f"Bearer {api_key}"},
                )
        """
        if not context_chunks:
            return (
                "I don't have enough context to answer that question. "
                "Try uploading some documents first!"
            )

        # Build a simple response from the retrieved chunks
        context_text = "\n\n".join(
            f"[Source: {chunk.source or 'unknown'}] {chunk.content[:200]}..."
            if len(chunk.content) > 200
            else f"[Source: {chunk.source or 'unknown'}] {chunk.content}"
            for chunk in context_chunks
        )

        answer = (
            f"Based on {len(context_chunks)} relevant document(s), "
            f"here's what I found for your query: \"{query}\"\n\n"
            f"---\n\n"
            f"{context_text}\n\n"
            f"---\n\n"
            f"Note: This is a template response. Real LLM generation "
            f"will be added in Day 4 with streaming and structured extraction."
        )

        logger.info(
            "Generated response for query: '%s' (%d chunks)",
            query[:50],
            len(context_chunks),
        )
        return answer
