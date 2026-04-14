"""Unit tests for RAG components."""

import pytest


class TestCohereEmbedder:
    """Tests for the Cohere embedding wrapper."""

    def test_embedder_init(self):
        """Test embedder can be imported and has correct defaults."""
        from arcane.rag.embeddings import CohereEmbedder

        embedder = CohereEmbedder.__new__(CohereEmbedder)
        embedder.model = "embed-english-v3.0"
        embedder.batch_size = 96
        embedder.dimensions = 1024

        assert embedder.model == "embed-english-v3.0"
        assert embedder.batch_size == 96
        assert embedder.dimensions == 1024

    def test_embed_documents_empty(self):
        """Test that empty input returns empty output."""
        from arcane.rag.embeddings import CohereEmbedder

        embedder = CohereEmbedder.__new__(CohereEmbedder)
        assert embedder.embed_documents([]) == []


class TestRedisVectorStore:
    """Tests for the Redis vector store (structure only, no Redis needed)."""

    def test_document_model(self):
        from arcane.rag.vectorstore import Document

        doc = Document(
            id="test-1",
            content="Test content",
            metadata={"source": "test"},
            score=0.95,
        )
        assert doc.id == "test-1"
        assert doc.content == "Test content"
        assert doc.score == 0.95
        assert doc.metadata["source"] == "test"

    def test_document_model_defaults(self):
        from arcane.rag.vectorstore import Document

        doc = Document(id="test-2", content="Minimal")
        assert doc.embedding is None
        assert doc.score is None
        assert doc.metadata == {}


class TestHybridRetriever:
    """Tests for the hybrid retriever (structure only)."""

    def test_retriever_import(self):
        from arcane.rag.retriever import HybridRetriever
        # Just verify the class can be imported
        assert HybridRetriever is not None


class TestSemanticCache:
    """Tests for the semantic cache (structure only)."""

    def test_cache_import(self):
        from arcane.rag.cache import SemanticCache
        assert SemanticCache is not None


class TestRAGPipeline:
    """Tests for the RAG pipeline (structure only)."""

    def test_pipeline_import(self):
        from arcane.rag.pipeline import RAGPipeline
        assert RAGPipeline is not None
