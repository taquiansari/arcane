"""Session management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from arcane.api.schemas import SessionInfo, SessionListResponse
from arcane.memory.session import get_session_manager
from arcane.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=SessionListResponse)
async def list_sessions(limit: int = 50) -> SessionListResponse:
    """List all research sessions, sorted by most recent."""
    manager = get_session_manager()
    sessions = manager.list_sessions(limit=limit)

    items = [
        SessionInfo(
            session_id=s["session_id"],
            query=s["query"],
            status=s["status"],
            created_at=s["created_at"],
            updated_at=s["updated_at"],
        )
        for s in sessions
    ]

    return SessionListResponse(sessions=items, total=len(items))


@router.get("/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get full details for a specific session."""
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return session


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    """Delete a research session."""
    manager = get_session_manager()
    deleted = manager.delete_session(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": "Session deleted", "session_id": session_id}
