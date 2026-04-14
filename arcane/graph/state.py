"""State schema for the LangGraph research workflow."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class ResearchState(TypedDict, total=False):
    """The central state object passed between all LangGraph nodes.

    Every agent reads from and writes to this state. It serves as the
    single source of truth for the entire research pipeline.
    """

    # --- User Input ---
    query: str                              # Original user research question
    session_id: str                         # Unique session identifier

    # --- Planning ---
    research_plan: dict[str, Any]           # Structured plan from Planner agent
    sub_queries: list[str]                  # Generated sub-queries for retrieval
    search_queries: list[dict[str, str]]    # Optimized queries from Query Generator
    current_query_index: int                # Track progress through sub-queries

    # --- Retrieval ---
    search_results: list[dict[str, Any]]    # Raw search results from tools
    retrieved_documents: list[dict[str, Any]]  # Processed & reranked documents
    source_urls: list[str]                  # All source URLs for citation

    # --- Analysis ---
    intermediate_findings: list[dict[str, Any]]  # Per-query analysis results

    # --- Critique Loop ---
    draft_report: str                       # Current draft of the report
    critique: dict[str, Any]                # Critic's structured feedback
    critique_score: float                   # 0.0-1.0 quality score
    revision_count: int                     # Number of revisions performed
    max_revisions: int                      # Maximum allowed revisions (default: 3)

    # --- Final Output ---
    final_report: str                       # Approved final report
    citations: list[dict[str, str]]         # Structured citation list

    # --- Control ---
    messages: Annotated[list, add_messages] # Conversation history
    errors: list[str]                       # Error log
    status: str                             # Current pipeline status
    human_review_requested: bool            # Whether HITL approval is needed
    human_feedback: str | None              # Feedback from human reviewer
