"""End-to-end RAG pipeline for Arcane.

Orchestrates: Cache Check → Retrieval → Reranking → Generation → Cache Store
"""

from __future__ import annotations

import json
from typing import Any

import cohere

from arcane.config import get_settings
from arcane.rag.cache import get_semantic_cache
from arcane.rag.retriever import HybridRetriever
from arcane.rag.vectorstore import get_vectorstore
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """Complete Retrieval-Augmented Generation pipeline.

    Flow:
    1. Check semantic cache for similar past queries
    2. If cache miss: retrieve relevant documents (hybrid search + rerank)
    3. Build context from retrieved documents
    4. Generate response using Cohere with grounded context
    5. Store result in semantic cache
    """

    def __init__(
        self,
        use_cache: bool = True,
        use_reranking: bool = True,
        top_k: int = 5,
        model: str = "command-r-plus",
    ):
        self.use_cache = use_cache
        self.retriever = HybridRetriever(
            top_k_retrieval=20,
            top_k_rerank=top_k,
            use_reranking=use_reranking,
        )
        self.model = model
        settings = get_settings()
        self.cohere_client = cohere.Client(api_key=settings.cohere_api_key)

    def query(
        self,
        question: str,
        session_id: str | None = None,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Execute the full RAG pipeline for a question.

        Args:
            question: The user's question.
            session_id: Optional session filter for retrieval.
            system_prompt: Optional system prompt for the LLM.

        Returns:
            Dict with 'answer', 'sources', 'cached', and metadata.
        """
        logger.info("rag_pipeline_start", question_preview=question[:80])

        # Step 1: Check semantic cache
        if self.use_cache:
            from arcane.rag.cache import get_semantic_cache

            cache = get_semantic_cache()
            cached_response = cache.get(question)
            if cached_response:
                return {
                    "answer": cached_response,
                    "sources": [],
                    "cached": True,
                    "documents_used": 0,
                }

        # Step 2: Retrieve relevant documents
        documents = self.retriever.retrieve(question, session_id=session_id)
        context = self.retriever.retrieve_as_context(question, session_id=session_id)

        # Step 3: Generate response with Cohere
        default_system = (
            "You are Arcane, an expert research assistant. "
            "Answer the question using ONLY the provided context. "
            "If the context doesn't contain enough information, say so clearly. "
            "Always cite your sources using [Source N] notation."
        )

        preamble = system_prompt or default_system

        try:
            response = self.cohere_client.chat(
                message=f"Question: {question}\n\nContext:\n{context}",
                model=self.model,
                preamble=preamble,
                temperature=0.3,
            )
            answer = response.text
        except Exception as e:
            logger.error("rag_generation_failed", error=str(e))
            answer = f"I encountered an error generating the response: {e}"

        # Step 4: Extract source information
        sources = []
        for doc in documents:
            sources.append({
                "title": doc.metadata.get("title", ""),
                "url": doc.metadata.get("url", ""),
                "source": doc.metadata.get("source", ""),
                "relevance_score": doc.score,
            })

        # Step 5: Cache the result
        if self.use_cache and answer and not answer.startswith("I encountered an error"):
            cache = get_semantic_cache()
            cache.set(question, answer)

        result = {
            "answer": answer,
            "sources": sources,
            "cached": False,
            "documents_used": len(documents),
        }

        logger.info(
            "rag_pipeline_complete",
            question_preview=question[:80],
            documents_used=len(documents),
        )
        return result

    def ingest(
        self,
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
    ) -> list[str]:
        """Ingest documents into the vector store.

        Args:
            contents: List of document texts to store.
            metadatas: Optional metadata for each document.
            session_id: Optional session ID to tag documents with.

        Returns:
            List of document IDs.
        """
        metas = metadatas or [{} for _ in contents]

        # Add session_id to all metadata
        if session_id:
            for m in metas:
                m["session_id"] = session_id

        vectorstore = get_vectorstore()
        return vectorstore.add_documents(contents, metas)
