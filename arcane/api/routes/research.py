"""Research endpoints — start, monitor, and manage research sessions."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, BackgroundTasks

from arcane.api.schemas import (
    ResearchRequest,
    ResearchResponse,
    ResearchResultResponse,
    FeedbackRequest,
)
from arcane.config import get_settings
from arcane.memory.session import get_session_manager
from arcane.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/research", tags=["research"])


async def _run_research_pipeline(session_id: str, query: str, config: dict[str, Any]) -> None:
    """Background task that runs the full LangGraph research pipeline."""
    manager = get_session_manager()

    try:
        manager.set_status(session_id, "running")

        from arcane.graph.builder import compile_research_graph
        from arcane.graph.checkpointer import create_memory_checkpointer

        checkpointer = create_memory_checkpointer()
        graph = compile_research_graph(checkpointer=checkpointer)

        settings = get_settings()

        initial_state = {
            "query": query,
            "session_id": session_id,
            "max_revisions": config.get("max_revisions", settings.max_revisions),
            "human_review_requested": config.get("human_review", False),
            "revision_count": 0,
            "intermediate_findings": [],
            "search_results": [],
            "source_urls": [],
            "errors": [],
            "status": "starting",
        }

        # Run the graph
        result = await asyncio.to_thread(
            graph.invoke,
            initial_state,
            config={"configurable": {"thread_id": session_id}},
        )

        # Store the result
        manager.set_result(session_id, {
            "final_report": result.get("final_report", ""),
            "citations": result.get("citations", []),
            "critique_score": result.get("critique_score", 0.0),
            "revision_count": result.get("revision_count", 0),
            "errors": result.get("errors", []),
        })

        logger.info("research_pipeline_complete", session_id=session_id)

    except Exception as e:
        logger.error("research_pipeline_failed", session_id=session_id, error=str(e))
        manager.set_error(session_id, str(e))


@router.post("", response_model=ResearchResponse)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    """Start a new research session.

    Kicks off the full LangGraph research pipeline in the background.
    Use GET /research/{session_id} to monitor progress and retrieve results.
    """
    manager = get_session_manager()

    config = {}
    if request.max_revisions is not None:
        config["max_revisions"] = request.max_revisions
    config["human_review"] = request.human_review

    session = manager.create_session(request.query, config)
    session_id = session["session_id"]

    # Launch the pipeline in the background
    background_tasks.add_task(
        _run_research_pipeline,
        session_id,
        request.query,
        config,
    )

    logger.info("research_started", session_id=session_id, query=request.query[:80])

    return ResearchResponse(
        session_id=session_id,
        query=request.query,
        status="started",
        message="Research pipeline started. Poll GET /api/v1/research/{session_id} for results.",
    )


@router.get("/{session_id}", response_model=ResearchResultResponse)
async def get_research_results(session_id: str) -> ResearchResultResponse:
    """Get the current status and results of a research session."""
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    result = session.get("result") or {}

    return ResearchResultResponse(
        session_id=session_id,
        query=session["query"],
        status=session["status"],
        report=result.get("final_report"),
        citations=result.get("citations", []),
        critique_score=result.get("critique_score"),
        revision_count=result.get("revision_count"),
        errors=result.get("errors", []),
    )


@router.post("/{session_id}/feedback")
async def submit_feedback(
    session_id: str,
    request: FeedbackRequest,
) -> dict:
    """Submit human feedback on a research draft.

    Used when human_review mode is enabled. The feedback will be
    incorporated into the next revision cycle.
    """
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if request.approve:
        manager.set_status(session_id, "approved")
        return {
            "message": "Draft approved. Report will be finalized.",
            "session_id": session_id,
        }

    # Store feedback for the next revision cycle
    manager.update_session(session_id, {
        "human_feedback": request.feedback,
        "status": "feedback_received",
    })

    return {
        "message": "Feedback received. Revision cycle will incorporate your input.",
        "session_id": session_id,
    }


@router.delete("/{session_id}")
async def cancel_research(session_id: str) -> dict:
    """Cancel a running research session."""
    manager = get_session_manager()
    session = manager.get_session(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    manager.set_status(session_id, "cancelled")

    return {"message": "Research session cancelled", "session_id": session_id}
