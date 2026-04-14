"""Conditional edge functions for the LangGraph research workflow."""

from __future__ import annotations

from typing import Literal

from arcane.config import get_settings
from arcane.graph.state import ResearchState
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


def should_continue_retrieval(state: ResearchState) -> Literal["retrieve", "analyze"]:
    """Decide whether to continue retrieving or move to analysis.

    Continues if there are more search queries to process.
    """
    search_queries = state.get("search_queries", [])
    current_index = state.get("current_query_index", 0)

    remaining = len(search_queries) - current_index

    if remaining > 0:
        logger.info("edge_continue_retrieval", remaining=remaining)
        return "retrieve"
    else:
        logger.info("edge_move_to_analysis")
        return "analyze"


def should_revise_or_finalize(
    state: ResearchState,
) -> Literal["revise", "human_review", "finalize"]:
    """Decide whether the report needs revision, human review, or is ready.

    Logic:
    1. If score >= threshold → finalize
    2. If max revisions reached → finalize (accept best effort)
    3. If human review is requested and this is the first pass → human_review
    4. Otherwise → revise (another synthesis + critique loop)
    """
    settings = get_settings()

    score = state.get("critique_score", 0.0)
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", settings.max_revisions)
    threshold = settings.critique_threshold
    human_review = state.get("human_review_requested", False)

    critique = state.get("critique", {})
    passed = critique.get("passed", False)

    logger.info(
        "edge_revise_decision",
        score=score,
        revision=revision_count,
        max_revisions=max_revisions,
        threshold=threshold,
        passed=passed,
    )

    # Quality passed or we've hit the revision limit
    if passed or score >= threshold:
        logger.info("edge_quality_passed", score=score)
        return "finalize"

    if revision_count >= max_revisions:
        logger.warning("edge_max_revisions_reached", revision_count=revision_count)
        return "finalize"

    # Optional human review gate
    if human_review and revision_count == 1:
        logger.info("edge_human_review_gate")
        return "human_review"

    # Needs revision
    logger.info("edge_needs_revision", score=score, revision=revision_count)
    return "revise"
