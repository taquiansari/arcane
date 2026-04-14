"""Crew assembly — creates task definitions and assembles agent teams."""

from __future__ import annotations

import os
from typing import Any

from crewai import Crew, Task, Process

from arcane.agents.planner import create_planner_agent
from arcane.agents.researcher import create_researcher_agent
from arcane.agents.critic import create_critic_agent
from arcane.agents.synthesizer import create_synthesizer_agent
from arcane.agents.query_generator import create_query_generator_agent
from arcane.config import get_settings
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


def get_default_llm() -> str:
    """Get the default LLM string for CrewAI (litellm format).

    CrewAI uses litellm under the hood. For Cohere, the format
    is 'cohere/command-r-plus'. We also ensure the COHERE_API_KEY
    env var is set so litellm can authenticate.
    """
    settings = get_settings()
    # litellm reads COHERE_API_KEY from env
    if settings.cohere_api_key:
        os.environ["COHERE_API_KEY"] = settings.cohere_api_key
    return "cohere/command-a-03-2025"


def create_planning_task(query: str) -> Task:
    """Create a research planning task."""
    return Task(
        description=(
            f"Analyze the following research question and create a comprehensive "
            f"research plan:\n\n"
            f'"{query}"\n\n'
            f"Your plan must include:\n"
            f"1. A clear main thesis or research objective\n"
            f"2. 3-6 prioritized sub-questions that break down the main question\n"
            f"3. A search strategy (academic first, web first, or parallel)\n"
            f"4. Expected source types (journals, conferences, blogs, docs)\n\n"
            f"Output your plan as structured JSON with these exact fields:\n"
            f'{{"main_thesis": "...", "sub_questions": [...], '
            f'"search_strategy": "...", "expected_sources": [...]}}'
        ),
        expected_output=(
            "A structured JSON object containing the research plan with "
            "main_thesis, sub_questions, search_strategy, and expected_sources."
        ),
    )


def create_query_generation_task(sub_questions: list[str]) -> Task:
    """Create a query generation task from sub-questions."""
    questions_str = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_questions))
    return Task(
        description=(
            f"Generate optimized search queries for each of these research "
            f"sub-questions:\n\n{questions_str}\n\n"
            f"For each sub-question, generate 1-2 queries optimized for web search "
            f"and 1-2 queries optimized for academic database search.\n\n"
            f"Output as structured JSON:\n"
            f'{{"queries": [{{"query": "...", "target": "web"|"academic", '
            f'"intent": "...", "sub_question_index": 0}}]}}'
        ),
        expected_output=(
            "A structured JSON object with a 'queries' list, each containing "
            "'query', 'target', 'intent', and 'sub_question_index' fields."
        ),
    )


def create_research_task(query: str, context: str = "") -> Task:
    """Create a research execution task for a single query."""
    context_section = f"\n\nAdditional context:\n{context}" if context else ""
    return Task(
        description=(
            f"Execute the following search query and extract relevant findings:\n\n"
            f'Query: "{query}"{context_section}\n\n'
            f"Steps:\n"
            f"1. Search using web_search and/or academic_search tools\n"
            f"2. Rerank results using the reranker tool for relevance\n"
            f"3. Scrape the top 2-3 most relevant URLs for full content\n"
            f"4. Extract key findings with specific facts, data, and quotes\n"
            f"5. Record all source URLs for citation\n\n"
            f"Output as structured JSON:\n"
            f'{{"findings": [{{"fact": "...", "source_title": "...", '
            f'"source_url": "...", "confidence": 0.0-1.0}}], '
            f'"sources": [{{"title": "...", "url": "...", "type": "..."}}]}}'
        ),
        expected_output=(
            "A structured JSON object with 'findings' (list of facts with sources) "
            "and 'sources' (list of all sources consulted)."
        ),
    )


def create_synthesis_task(
    query: str,
    findings: str,
    critique_feedback: str | None = None,
) -> Task:
    """Create a report synthesis task."""
    feedback_section = ""
    if critique_feedback:
        feedback_section = (
            f"\n\n--- REVIEWER FEEDBACK (must be addressed) ---\n"
            f"{critique_feedback}\n"
            f"--- END FEEDBACK ---\n\n"
            f"You MUST specifically address each issue raised by the reviewer."
        )

    return Task(
        description=(
            f"Synthesize the following research findings into a comprehensive, "
            f"well-structured report.\n\n"
            f"Original Research Question: {query}\n\n"
            f"Research Findings:\n{findings}\n"
            f"{feedback_section}\n"
            f"Write a detailed report (2000-4000 words) with:\n"
            f"1. Executive Summary\n"
            f"2. Background & Context\n"
            f"3. Detailed Findings (organized by theme)\n"
            f"4. Analysis & Discussion\n"
            f"5. Limitations & Gaps\n"
            f"6. Conclusions & Recommendations\n"
            f"7. References\n\n"
            f"Use inline citations [1], [2] throughout. Every factual claim "
            f"must be attributed to a source."
        ),
        expected_output="A comprehensive research report in Markdown format with citations.",
    )


def create_critique_task(report: str, query: str) -> Task:
    """Create a report critique task."""
    return Task(
        description=(
            f"Critically evaluate the following research report against the "
            f"quality rubric.\n\n"
            f"Original Research Question: {query}\n\n"
            f"Report to Evaluate:\n{report}\n\n"
            f"Evaluate across these 5 dimensions (0-10 each):\n"
            f"1. Source Credibility\n"
            f"2. Logical Consistency\n"
            f"3. Completeness\n"
            f"4. Citation Quality\n"
            f"5. Relevance to Query\n\n"
            f"Output as structured JSON:\n"
            f'{{"score": 0.0-1.0, "passed": true/false, '
            f'"dimension_scores": {{"source_credibility": 0-10, ...}}, '
            f'"issues": [{{"category": "...", "severity": "high|medium|low", '
            f'"detail": "..."}}], '
            f'"suggestions": ["..."]}}'\
            f"\n\nThe overall score is the average of all dimensions divided by 10. "
            f"Set passed=true only if score >= 0.8."
        ),
        expected_output=(
            "A structured JSON evaluation with score, passed, dimension_scores, "
            "issues, and suggestions."
        ),
    )


def assemble_planning_crew(query: str, llm: object | None = None) -> Crew:
    """Assemble a crew for the planning phase."""
    llm = llm or get_default_llm()
    planner = create_planner_agent(llm)
    task = create_planning_task(query)
    task.agent = planner

    return Crew(
        agents=[planner],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def assemble_research_crew(
    queries: list[str],
    llm: object | None = None,
) -> Crew:
    """Assemble a crew for the research phase."""
    llm = llm or get_default_llm()
    researcher = create_researcher_agent(llm)
    query_gen = create_query_generator_agent(llm)

    tasks = []
    for q in queries:
        task = create_research_task(q)
        task.agent = researcher
        tasks.append(task)

    return Crew(
        agents=[researcher, query_gen],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def assemble_synthesis_crew(
    query: str,
    findings: str,
    critique_feedback: str | None = None,
    llm: object | None = None,
) -> Crew:
    """Assemble a crew for the synthesis phase."""
    llm = llm or get_default_llm()
    synthesizer = create_synthesizer_agent(llm)
    task = create_synthesis_task(query, findings, critique_feedback)
    task.agent = synthesizer

    return Crew(
        agents=[synthesizer],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def assemble_critique_crew(
    report: str,
    query: str,
    llm: object | None = None,
) -> Crew:
    """Assemble a crew for the critique phase."""
    llm = llm or get_default_llm()
    critic = create_critic_agent(llm)
    task = create_critique_task(report, query)
    task.agent = critic

    return Crew(
        agents=[critic],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
