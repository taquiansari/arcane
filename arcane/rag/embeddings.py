"""Cohere embedding wrapper for Arcane."""

from __future__ import annotations

from typing import Literal

import cohere
from pydantic import BaseModel, Field

from arcane.config import get_settings
from arcane.utils.logging import get_logger
from arcane.utils.retry import retry_on_api_error

logger = get_logger(__name__)


class CohereEmbedder:
    """Wrapper around Cohere Embed API with batching and input type handling.

    Uses Cohere's embed-english-v3.0 model (1024 dimensions).
    Handles the distinction between 'search_document' (for indexing)
    and 'search_query' (for retrieval) input types.
    """

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        batch_size: int = 96,
    ):
        settings = get_settings()
        self.client = cohere.Client(api_key=settings.cohere_api_key)
        self.model = model
        self.batch_size = batch_size
        self.dimensions = 1024

    @retry_on_api_error(max_attempts=3)
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents for indexing.

        Uses input_type='search_document' for optimal retrieval performance.
        Automatically batches large inputs.
        """
        if not texts:
            return []

        logger.info("embedding_documents", count=len(texts))

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            # Clean empty strings
            batch = [t if t.strip() else "empty" for t in batch]

            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type="search_document",
                truncate="END",
            )
            all_embeddings.extend(response.embeddings)

        logger.info("embedding_documents_complete", count=len(all_embeddings))
        return all_embeddings

    @retry_on_api_error(max_attempts=3)
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query for retrieval.

        Uses input_type='search_query' for optimal retrieval performance.
        """
        if not text.strip():
            raise ValueError("Query text cannot be empty")

        logger.debug("embedding_query", query_preview=text[:80])

        response = self.client.embed(
            texts=[text],
            model=self.model,
            input_type="search_query",
            truncate="END",
        )
        return response.embeddings[0]

    @retry_on_api_error(max_attempts=3)
    def embed_texts(
        self,
        texts: list[str],
        input_type: Literal["search_document", "search_query", "classification", "clustering"] = "search_document",
    ) -> list[list[float]]:
        """Embed texts with explicit input type control."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch = [t if t.strip() else "empty" for t in batch]

            response = self.client.embed(
                texts=batch,
                model=self.model,
                input_type=input_type,
                truncate="END",
            )
            all_embeddings.extend(response.embeddings)

        return all_embeddings


# Module-level singleton
_embedder: CohereEmbedder | None = None


def get_embedder() -> CohereEmbedder:
    """Get or create the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = CohereEmbedder()
    return _embedder
