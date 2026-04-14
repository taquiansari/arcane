"""Redis vector store for Arcane using RedisVL."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis
from pydantic import BaseModel

from arcane.config import get_settings
from arcane.rag.embeddings import get_embedder
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


class Document(BaseModel):
    """A document stored in the vector store."""

    id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = {}
    score: float | None = None


class RedisVectorStore:
    """Redis-based vector store with HNSW indexing.

    Supports document storage, vector similarity search,
    and hybrid search combining vector + keyword matching.
    Uses Redis Search (RediSearch) module.
    """

    def __init__(
        self,
        index_name: str = "arcane_docs",
        prefix: str = "doc:",
        vector_dim: int = 1024,
    ):
        settings = get_settings()
        self.redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self.index_name = index_name
        self.prefix = prefix
        self.vector_dim = vector_dim
        self.embedder = get_embedder()
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the RediSearch index if it doesn't exist."""
        try:
            self.redis_client.ft(self.index_name).info()
            logger.debug("redis_index_exists", index=self.index_name)
        except redis.ResponseError:
            logger.info("creating_redis_index", index=self.index_name)
            from redis.commands.search.field import (
                TextField,
                NumericField,
                VectorField,
                TagField,
            )
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType

            schema = [
                TextField("content", weight=1.0),
                TextField("title", weight=2.0),
                TagField("source"),
                TagField("session_id"),
                NumericField("timestamp"),
                VectorField(
                    "embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                        "M": 16,
                        "EF_CONSTRUCTION": 200,
                    },
                ),
            ]

            definition = IndexDefinition(
                prefix=[self.prefix],
                index_type=IndexType.HASH,
            )

            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition,
            )
            logger.info("redis_index_created", index=self.index_name)

    def add_documents(
        self,
        contents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents to the vector store.

        Embeds the documents using Cohere and stores them in Redis.

        Args:
            contents: List of document text contents.
            metadatas: Optional list of metadata dicts for each document.
            ids: Optional list of document IDs. Auto-generated if not provided.

        Returns:
            List of document IDs.
        """
        if not contents:
            return []

        logger.info("adding_documents", count=len(contents))

        # Generate IDs if not provided
        doc_ids = ids or [str(uuid.uuid4()) for _ in contents]
        metas = metadatas or [{} for _ in contents]

        # Embed all documents
        embeddings = self.embedder.embed_documents(contents)

        # Store in Redis
        pipe = self.redis_client.pipeline()
        for doc_id, content, embedding, meta in zip(doc_ids, contents, embeddings, metas):
            key = f"{self.prefix}{doc_id}"

            import struct

            embedding_bytes = struct.pack(f"{len(embedding)}f", *embedding)

            doc_data = {
                "content": content,
                "title": meta.get("title", ""),
                "source": meta.get("source", "unknown"),
                "session_id": meta.get("session_id", ""),
                "timestamp": meta.get("timestamp", int(time.time())),
                "metadata": json.dumps(meta),
                "embedding": embedding_bytes,
            }
            pipe.hset(key, mapping=doc_data)

        pipe.execute()
        logger.info("documents_added", count=len(doc_ids))
        return doc_ids

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[Document]:
        """Search for similar documents using vector similarity.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            session_id: Optional filter by session ID.

        Returns:
            List of Document objects sorted by similarity.
        """
        logger.info("similarity_search", query_preview=query[:80], top_k=top_k)

        # Embed the query
        query_embedding = self.embedder.embed_query(query)

        import struct

        query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)

        # Build the search query
        from redis.commands.search.query import Query

        filter_str = f"@session_id:{{{session_id}}}" if session_id else "*"

        q = (
            Query(f"({filter_str})=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("content", "title", "source", "metadata", "score")
            .paging(0, top_k)
            .dialect(2)
        )

        results = self.redis_client.ft(self.index_name).search(
            q, query_params={"vec": query_bytes}
        )

        documents = []
        for doc in results.docs:
            meta = {}
            try:
                meta = json.loads(doc.metadata) if hasattr(doc, "metadata") and doc.metadata else {}
            except (json.JSONDecodeError, AttributeError):
                pass

            documents.append(Document(
                id=doc.id.replace(self.prefix, ""),
                content=doc.content if hasattr(doc, "content") else "",
                metadata={
                    **meta,
                    "title": doc.title if hasattr(doc, "title") else "",
                    "source": doc.source if hasattr(doc, "source") else "",
                },
                score=float(doc.score) if hasattr(doc, "score") else None,
            ))

        logger.info("similarity_search_complete", result_count=len(documents))
        return documents

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[Document]:
        """Hybrid search combining vector similarity with keyword matching.

        Executes both a vector search and a keyword search, then merges
        and deduplicates results.
        """
        logger.info("hybrid_search", query_preview=query[:80])

        # Vector search
        vector_results = self.similarity_search(query, top_k=top_k, session_id=session_id)

        # Keyword search (BM25 via RediSearch)
        keyword_results = self._keyword_search(query, top_k=top_k, session_id=session_id)

        # Merge and deduplicate
        seen_ids: set[str] = set()
        merged: list[Document] = []

        for doc in vector_results:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                merged.append(doc)

        for doc in keyword_results:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                merged.append(doc)

        logger.info(
            "hybrid_search_complete",
            vector_count=len(vector_results),
            keyword_count=len(keyword_results),
            merged_count=len(merged),
        )
        return merged[:top_k]

    def _keyword_search(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[Document]:
        """BM25 keyword search using RediSearch full-text index."""
        try:
            from redis.commands.search.query import Query

            # Escape special RediSearch characters
            escaped_query = query.replace("-", " ").replace(":", " ")
            search_str = f"@content:({escaped_query})"

            if session_id:
                search_str += f" @session_id:{{{session_id}}}"

            q = (
                Query(search_str)
                .return_fields("content", "title", "source", "metadata")
                .paging(0, top_k)
            )

            results = self.redis_client.ft(self.index_name).search(q)

            documents = []
            for doc in results.docs:
                meta = {}
                try:
                    meta = json.loads(doc.metadata) if hasattr(doc, "metadata") and doc.metadata else {}
                except (json.JSONDecodeError, AttributeError):
                    pass

                documents.append(Document(
                    id=doc.id.replace(self.prefix, ""),
                    content=doc.content if hasattr(doc, "content") else "",
                    metadata=meta,
                ))

            return documents

        except Exception as e:
            logger.warning("keyword_search_failed", error=str(e))
            return []

    def delete_by_session(self, session_id: str) -> int:
        """Delete all documents belonging to a session."""
        from redis.commands.search.query import Query

        q = Query(f"@session_id:{{{session_id}}}").paging(0, 10000)
        results = self.redis_client.ft(self.index_name).search(q)

        pipe = self.redis_client.pipeline()
        for doc in results.docs:
            pipe.delete(doc.id)
        pipe.execute()

        logger.info("documents_deleted", session_id=session_id, count=len(results.docs))
        return len(results.docs)

    def health_check(self) -> bool:
        """Check Redis connectivity and index health."""
        try:
            self.redis_client.ping()
            self.redis_client.ft(self.index_name).info()
            return True
        except Exception:
            return False


# Module-level singleton
_vectorstore: RedisVectorStore | None = None


def get_vectorstore() -> RedisVectorStore:
    """Get or create the global vector store instance."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = RedisVectorStore()
    return _vectorstore
