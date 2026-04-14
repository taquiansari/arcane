"""Graph builder — constructs and compiles the LangGraph research workflow."""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END, START

from arcane.graph.state import ResearchState
from arcane.graph.nodes import (
    plan_research,
    generate_queries,
    retrieve_and_search,
    analyze_findings,
    synthesize_report,
    critique_report,
    finalize_report,
)
from arcane.graph.edges import (
    should_continue_retrieval,
    should_revise_or_finalize,
)
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


def build_research_graph() -> StateGraph:
    """Build the research workflow state graph.

    Graph topology:
        START → plan → generate_queries → retrieve → [loop until done] → analyze
        → synthesize → critique → [revise loop / human review / finalize] → END

    Returns:
        Compiled StateGraph ready for execution.
    """
    logger.info("building_research_graph")

    graph = StateGraph(ResearchState)

    # --- Add nodes ---
    graph.add_node("plan", plan_research)
    graph.add_node("generate_queries", generate_queries)
    graph.add_node("retrieve", retrieve_and_search)
    graph.add_node("analyze", analyze_findings)
    graph.add_node("synthesize", synthesize_report)
    graph.add_node("critique", critique_report)
    graph.add_node("finalize", finalize_report)

    # --- Add edges ---

    # Linear flow: START → plan → generate_queries → retrieve
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "generate_queries")
    graph.add_edge("generate_queries", "retrieve")

    # Retrieval loop: retrieve → [more queries?] → retrieve OR analyze
    graph.add_conditional_edges(
        "retrieve",
        should_continue_retrieval,
        {
            "retrieve": "retrieve",
            "analyze": "analyze",
        },
    )

    # Linear: analyze → synthesize → critique
    graph.add_edge("analyze", "synthesize")
    graph.add_edge("synthesize", "critique")

    # Critique decision: critique → revise OR human_review OR finalize
    graph.add_conditional_edges(
        "critique",
        should_revise_or_finalize,
        {
            "revise": "synthesize",       # Loop back for revision
            "human_review": "synthesize",  # Will wait for human input, then re-synthesize
            "finalize": "finalize",        # Quality passed!
        },
    )

    # Finalize → END
    graph.add_edge("finalize", END)

    logger.info("research_graph_built")
    return graph


def compile_research_graph(
    checkpointer: Any | None = None,
) -> Any:
    """Compile the research graph with optional checkpointing.

    Args:
        checkpointer: Optional LangGraph checkpointer for state persistence.

    Returns:
        Compiled graph ready for invocation.
    """
    graph = build_research_graph()

    compile_kwargs: dict[str, Any] = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    compiled = graph.compile(**compile_kwargs)
    logger.info("research_graph_compiled", has_checkpointer=checkpointer is not None)
    return compiled
