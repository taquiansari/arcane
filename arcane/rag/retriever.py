"""Hybrid retriever combining vector search with BM25 and Cohere reranking."""

from __future__ import annotations

import json
from typing import Any

from arcane.config import get_settings
from arcane.rag.vectorstore import Document, get_vectorstore
from arcane.utils.logging import get_logger
from arcane.utils.retry import retry_on_api_error

logger = get_logger(__name__)


class HybridRetriever:
    """Retriever that combines vector similarity, keyword search, and Cohere reranking.

    Pipeline:
    1. Hybrid search (vector + BM25) from Redis
    2. Cohere Rerank for final relevance scoring
    3. Return top-k most relevant documents
    """

    def __init__(
        self,
        top_k_retrieval: int = 20,
        top_k_rerank: int = 5,
        rerank_model: str = "rerank-v3.5",
        use_reranking: bool = True,
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.rerank_model = rerank_model
        self.use_reranking = use_reranking
        self.vectorstore = get_vectorstore()

    def retrieve(
        self,
        query: str,
        session_id: str | None = None,
    ) -> list[Document]:
        """Retrieve the most relevant documents for a query.

        Args:
            query: Search query.
            session_id: Optional session filter.

        Returns:
            List of Documents sorted by relevance.
        """
        logger.info("retriever_executing", query_preview=query[:80])

        # Step 1: Hybrid search (vector + BM25)
        candidates = self.vectorstore.hybrid_search(
            query=query,
            top_k=self.top_k_retrieval,
            session_id=session_id,
        )

        if not candidates:
            logger.info("retriever_no_candidates", query_preview=query[:80])
            return []

        # Step 2: Rerank with Cohere (if enabled and enough candidates)
        if self.use_reranking and len(candidates) > 1:
            candidates = self._rerank(query, candidates)

        logger.info(
            "retriever_complete",
            query_preview=query[:80],
            result_count=len(candidates),
        )
        return candidates

    @retry_on_api_error(max_attempts=2)
    def _rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Rerank documents using Cohere Rerank API."""
        import cohere

        settings = get_settings()
        client = cohere.Client(api_key=settings.cohere_api_key)

        # Prepare texts for reranking
        doc_texts = [doc.content for doc in documents]

        try:
            response = client.rerank(
                query=query,
                documents=doc_texts,
                top_n=min(self.top_k_rerank, len(doc_texts)),
                model=self.rerank_model,
            )

            # Map results back to Documents
            reranked = []
            for result in response.results:
                idx = result.index
                if idx < len(documents):
                    doc = documents[idx]
                    doc.score = result.relevance_score
                    reranked.append(doc)

            logger.info(
                "reranking_complete",
                input_count=len(documents),
                output_count=len(reranked),
            )
            return reranked

        except Exception as e:
            logger.warning("reranking_failed_using_original_order", error=str(e))
            return documents[: self.top_k_rerank]

    def retrieve_as_context(
        self,
        query: str,
        session_id: str | None = None,
        max_context_length: int = 6000,
    ) -> str:
        """Retrieve documents and format as a context string for LLM consumption.

        Args:
            query: Search query.
            session_id: Optional session filter.
            max_context_length: Maximum characters for the assembled context.

        Returns:
            Formatted context string with source attribution.
        """
        documents = self.retrieve(query, session_id)

        if not documents:
            return "No relevant documents found."

        context_parts = []
        total_length = 0

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "Untitled")
            score_str = f" (relevance: {doc.score:.3f})" if doc.score is not None else ""

            entry = f"[Source {i}: {title} ({source}){score_str}]\n{doc.content}\n"

            if total_length + len(entry) > max_context_length:
                break

            context_parts.append(entry)
            total_length += len(entry)

        return "\n---\n".join(context_parts)
