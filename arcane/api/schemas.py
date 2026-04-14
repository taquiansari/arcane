"""Pydantic request/response schemas for the Arcane API."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────


class ResearchRequest(BaseModel):
    """Request to start a new research session."""

    query: str = Field(..., min_length=5, max_length=2000, description="Research question")
    max_revisions: int | None = Field(None, ge=1, le=10, description="Max critique revisions")
    human_review: bool = Field(False, description="Enable human-in-the-loop review")


class FeedbackRequest(BaseModel):
    """Human feedback on a research draft."""

    feedback: str = Field(..., min_length=1, max_length=5000, description="Feedback text")
    approve: bool = Field(False, description="Approve the current draft as final")


# ── Response Models ─────────────────────────────────────────


class SessionInfo(BaseModel):
    """Summary info for a research session."""

    session_id: str
    query: str
    status: str
    created_at: float
    updated_at: float


class ResearchResponse(BaseModel):
    """Response after starting a research session."""

    session_id: str
    query: str
    status: str
    message: str


class ResearchResultResponse(BaseModel):
    """Full research results for a completed session."""

    session_id: str
    query: str
    status: str
    report: str | None = None
    citations: list[dict[str, Any]] = []
    critique_score: float | None = None
    revision_count: int | None = None
    errors: list[str] = []


class SessionListResponse(BaseModel):
    """List of research sessions."""

    sessions: list[SessionInfo]
    total: int


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    redis_connected: bool
    version: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None


# ── WebSocket Event Models ──────────────────────────────────


class WSEvent(BaseModel):
    """WebSocket event sent to clients during research."""

    type: str = Field(
        ...,
        description="Event type: status, progress, draft, critique, final, error",
    )
    data: dict[str, Any] = Field(default_factory=dict)
