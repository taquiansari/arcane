"""Redis-backed conversation memory for Arcane sessions."""

from __future__ import annotations

import json
import time
from typing import Any

import redis

from arcane.config import get_settings
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


class RedisMemory:
    """Redis-backed conversation memory with configurable window size.

    Stores conversation turns (user queries + agent responses) in Redis
    lists, enabling multi-turn research conversations with context.
    """

    def __init__(
        self,
        prefix: str = "memory:",
        max_turns: int = 20,
        ttl_hours: int = 72,
    ):
        settings = get_settings()
        self.client = redis.from_url(settings.redis_url, decode_responses=True)
        self.prefix = prefix
        self.max_turns = max_turns
        self.ttl_seconds = ttl_hours * 3600

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a conversation turn to memory.

        Args:
            session_id: Session identifier.
            role: Either 'user' or 'assistant'.
            content: The message content.
            metadata: Optional metadata (e.g., node name, timestamp).
        """
        key = self._key(session_id)

        turn = {
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {},
        }

        self.client.rpush(key, json.dumps(turn))

        # Trim to max turns
        self.client.ltrim(key, -self.max_turns, -1)

        # Set/refresh TTL
        self.client.expire(key, self.ttl_seconds)

        logger.debug("memory_turn_added", session_id=session_id, role=role)

    def get_history(
        self,
        session_id: str,
        last_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve conversation history for a session.

        Args:
            session_id: Session identifier.
            last_n: Optional limit on number of recent turns.

        Returns:
            List of conversation turn dicts.
        """
        key = self._key(session_id)

        if last_n:
            raw = self.client.lrange(key, -last_n, -1)
        else:
            raw = self.client.lrange(key, 0, -1)

        turns = []
        for item in raw:
            try:
                turns.append(json.loads(item))
            except json.JSONDecodeError:
                continue

        return turns

    def get_context_string(
        self,
        session_id: str,
        last_n: int = 10,
    ) -> str:
        """Get conversation history formatted as a context string.

        Args:
            session_id: Session identifier.
            last_n: Number of recent turns to include.

        Returns:
            Formatted conversation history string.
        """
        turns = self.get_history(session_id, last_n)

        if not turns:
            return ""

        lines = []
        for turn in turns:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")

        return "\n\n".join(lines)

    def clear(self, session_id: str) -> None:
        """Clear all memory for a session."""
        key = self._key(session_id)
        self.client.delete(key)
        logger.info("memory_cleared", session_id=session_id)

    def exists(self, session_id: str) -> bool:
        """Check if memory exists for a session."""
        return self.client.exists(self._key(session_id)) > 0


# Module-level singleton
_memory: RedisMemory | None = None


def get_memory() -> RedisMemory:
    """Get or create the global memory instance."""
    global _memory
    if _memory is None:
        _memory = RedisMemory()
    return _memory
