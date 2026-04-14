"""Semantic cache — Redis-backed caching of query-response pairs."""

from __future__ import annotations

import hashlib
import json
import struct
import time
from typing import Any

import redis

from arcane.config import get_settings
from arcane.rag.embeddings import get_embedder
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


class SemanticCache:
    """Redis-based semantic cache for query-response pairs.

    Before making expensive LLM or search calls, checks if a
    semantically similar query has been answered recently.
    Uses cosine similarity on query embeddings to find cache hits.

    Benefits:
    - ~60x faster responses on cache hits
    - Significant cost reduction on API calls
    - Configurable TTL for freshness control
    """

    def __init__(
        self,
        index_name: str = "arcane_cache",
        prefix: str = "cache:",
        similarity_threshold: float = 0.92,
        ttl_hours: int | None = None,
    ):
        settings = get_settings()
        self.redis_client = redis.from_url(
            settings.redis_url,
            decode_responses=False,  # Need bytes for vectors
        )
        self.text_client = redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self.index_name = index_name
        self.prefix = prefix
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = (ttl_hours or settings.cache_ttl_hours) * 3600
        self.embedder = get_embedder()
        self._ensure_index()

    def _ensure_index(self) -> None:
        """Create the cache index if it doesn't exist."""
        try:
            self.text_client.ft(self.index_name).info()
        except redis.ResponseError:
            logger.info("creating_cache_index", index=self.index_name)
            from redis.commands.search.field import TextField, NumericField, VectorField
            from redis.commands.search.indexDefinition import IndexDefinition, IndexType

            schema = [
                TextField("query"),
                TextField("response"),
                NumericField("timestamp"),
                VectorField(
                    "query_embedding",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": 1024,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            ]

            definition = IndexDefinition(
                prefix=[self.prefix],
                index_type=IndexType.HASH,
            )

            self.text_client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition,
            )
            logger.info("cache_index_created", index=self.index_name)

    def get(self, query: str) -> str | None:
        """Check the cache for a semantically similar query.

        Args:
            query: The search query.

        Returns:
            Cached response if a match is found, None otherwise.
        """
        try:
            query_embedding = self.embedder.embed_query(query)
            query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)

            from redis.commands.search.query import Query

            q = (
                Query("(*)=>[KNN 1 @query_embedding $vec AS distance]")
                .sort_by("distance")
                .return_fields("query", "response", "timestamp", "distance")
                .paging(0, 1)
                .dialect(2)
            )

            results = self.text_client.ft(self.index_name).search(
                q, query_params={"vec": query_bytes}
            )

            if results.docs:
                doc = results.docs[0]
                distance = float(doc.distance) if hasattr(doc, "distance") else 1.0
                similarity = 1.0 - distance  # Cosine distance to similarity

                if similarity >= self.similarity_threshold:
                    # Check TTL
                    timestamp = float(doc.timestamp) if hasattr(doc, "timestamp") else 0
                    age = time.time() - timestamp
                    if age <= self.ttl_seconds:
                        logger.info(
                            "cache_hit",
                            query_preview=query[:60],
                            similarity=f"{similarity:.3f}",
                            cached_query=doc.query[:60] if hasattr(doc, "query") else "",
                        )
                        return doc.response if hasattr(doc, "response") else None

            logger.debug("cache_miss", query_preview=query[:60])
            return None

        except Exception as e:
            logger.warning("cache_get_failed", error=str(e))
            return None

    def set(self, query: str, response: str) -> None:
        """Store a query-response pair in the cache.

        Args:
            query: The original query.
            response: The generated response.
        """
        try:
            query_embedding = self.embedder.embed_query(query)
            query_bytes = struct.pack(f"{len(query_embedding)}f", *query_embedding)

            cache_id = hashlib.sha256(query.encode()).hexdigest()[:16]
            key = f"{self.prefix}{cache_id}"

            self.redis_client.hset(
                key,
                mapping={
                    "query": query.encode("utf-8"),
                    "response": response.encode("utf-8"),
                    "timestamp": str(time.time()).encode("utf-8"),
                    "query_embedding": query_bytes,
                },
            )

            # Set TTL on the key
            self.redis_client.expire(key, self.ttl_seconds)

            logger.info("cache_stored", query_preview=query[:60], key=key)

        except Exception as e:
            logger.warning("cache_set_failed", error=str(e))

    def invalidate(self, query: str | None = None) -> int:
        """Invalidate cache entries.

        Args:
            query: If provided, invalidate only the matching entry.
                   If None, clears the entire cache.

        Returns:
            Number of entries invalidated.
        """
        try:
            if query:
                cache_id = hashlib.sha256(query.encode()).hexdigest()[:16]
                key = f"{self.prefix}{cache_id}"
                deleted = self.redis_client.delete(key)
                logger.info("cache_invalidated", key=key)
                return deleted

            # Clear all cache entries
            from redis.commands.search.query import Query

            q = Query("*").paging(0, 10000)
            results = self.text_client.ft(self.index_name).search(q)

            pipe = self.redis_client.pipeline()
            for doc in results.docs:
                pipe.delete(doc.id)
            pipe.execute()

            logger.info("cache_cleared", count=len(results.docs))
            return len(results.docs)

        except Exception as e:
            logger.warning("cache_invalidate_failed", error=str(e))
            return 0


# Module-level singleton
_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Get or create the global semantic cache instance."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
