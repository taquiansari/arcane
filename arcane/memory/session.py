"""Session state management — create, retrieve, update research sessions."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis

from arcane.config import get_settings
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages research sessions in Redis.

    Each session tracks:
    - Session metadata (ID, query, timestamps, status)
    - Research state snapshots
    - Configuration overrides
    """

    def __init__(self, prefix: str = "session:", ttl_hours: int = 168):
        settings = get_settings()
        self.client = redis.from_url(settings.redis_url, decode_responses=True)
        self.prefix = prefix
        self.ttl_seconds = ttl_hours * 3600  # Default: 7 days

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}{session_id}"

    def create_session(
        self,
        query: str,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new research session.

        Args:
            query: The research question.
            config: Optional configuration overrides.

        Returns:
            Session metadata dict.
        """
        session_id = str(uuid.uuid4())
        now = time.time()

        session = {
            "session_id": session_id,
            "query": query,
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "config": config or {},
            "result": None,
            "error": None,
        }

        key = self._key(session_id)
        self.client.set(key, json.dumps(session), ex=self.ttl_seconds)

        logger.info("session_created", session_id=session_id, query=query[:80])
        return session

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve a session by ID."""
        key = self._key(session_id)
        raw = self.client.get(key)

        if raw is None:
            return None

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def update_session(
        self,
        session_id: str,
        updates: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Update a session with new data.

        Args:
            session_id: Session to update.
            updates: Fields to update.

        Returns:
            Updated session dict, or None if not found.
        """
        session = self.get_session(session_id)
        if session is None:
            return None

        session.update(updates)
        session["updated_at"] = time.time()

        key = self._key(session_id)
        self.client.set(key, json.dumps(session), ex=self.ttl_seconds)

        logger.debug("session_updated", session_id=session_id, fields=list(updates.keys()))
        return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        key = self._key(session_id)
        deleted = self.client.delete(key)
        if deleted:
            logger.info("session_deleted", session_id=session_id)
        return deleted > 0

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """List all active sessions, sorted by most recently updated.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session metadata dicts.
        """
        pattern = f"{self.prefix}*"
        keys = list(self.client.scan_iter(match=pattern, count=100))

        sessions = []
        for key in keys:
            raw = self.client.get(key)
            if raw:
                try:
                    session = json.loads(raw)
                    sessions.append(session)
                except json.JSONDecodeError:
                    continue

        # Sort by updated_at descending
        sessions.sort(key=lambda s: s.get("updated_at", 0), reverse=True)
        return sessions[:limit]

    def set_status(self, session_id: str, status: str) -> None:
        """Update just the status field."""
        self.update_session(session_id, {"status": status})

    def set_result(self, session_id: str, result: dict[str, Any]) -> None:
        """Store the final research result."""
        self.update_session(session_id, {"result": result, "status": "complete"})

    def set_error(self, session_id: str, error: str) -> None:
        """Store an error and mark session as failed."""
        self.update_session(session_id, {"error": error, "status": "failed"})


# Module-level singleton
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
