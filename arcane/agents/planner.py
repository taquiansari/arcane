"""Planner Agent — decomposes research questions into structured plans."""

from __future__ import annotations

from crewai import Agent
from arcane.config import get_settings


def create_planner_agent(llm: object | None = None) -> Agent:
    """Create the Research Planner agent.

    The Planner decomposes a complex research question into a structured
    plan with prioritized sub-questions and a search strategy.

    Args:
        llm: Optional LLM instance. Uses Cohere Command R+ by default.

    Returns:
        Configured CrewAI Agent.
    """
    return Agent(
        role="Senior Research Strategist",
        goal=(
            "Decompose complex research questions into a structured research plan "
            "with prioritized sub-questions, identify the most productive search "
            "strategies, and anticipate what types of sources will be most valuable."
        ),
        backstory=(
            "You are a veteran academic researcher with 20 years of experience "
            "conducting systematic literature reviews across multiple disciplines. "
            "You excel at breaking down ambiguous, broad research topics into "
            "precise, searchable sub-questions. You understand the difference "
            "between exploratory questions (what is), comparative questions "
            "(how does X compare to Y), and evaluative questions (what are the "
            "limitations of). You always consider both academic and industry "
            "perspectives and prioritize questions that will yield the most "
            "informative answers."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm,
    )
