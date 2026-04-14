"""Graph node functions — each node is a step in the research pipeline."""

from __future__ import annotations

import json
from typing import Any

from arcane.graph.state import ResearchState
from arcane.agents.crew import (
    assemble_planning_crew,
    assemble_research_crew,
    assemble_synthesis_crew,
    assemble_critique_crew,
    create_query_generation_task,
)
from arcane.config import get_settings
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


def plan_research(state: ResearchState) -> dict[str, Any]:
    """Node: Decompose the research query into a structured plan.

    Invokes the Planner agent to break down the user's question
    into sub-questions and a search strategy.
    """
    query = state["query"]
    logger.info("node_plan_research", query=query)

    try:
        crew = assemble_planning_crew(query)
        result = crew.kickoff()

        # Parse the structured output
        result_text = str(result)
        plan = _extract_json(result_text)

        sub_questions = plan.get("sub_questions", [])
        if not sub_questions:
            # Fallback: use the query itself as a sub-question
            sub_questions = [query]

        return {
            "research_plan": plan,
            "sub_queries": sub_questions,
            "current_query_index": 0,
            "intermediate_findings": [],
            "search_results": [],
            "source_urls": [],
            "errors": state.get("errors", []),
            "status": "planning_complete",
        }

    except Exception as e:
        logger.error("plan_research_failed", error=str(e))
        return {
            "research_plan": {"main_thesis": query, "sub_questions": [query]},
            "sub_queries": [query],
            "current_query_index": 0,
            "intermediate_findings": [],
            "search_results": [],
            "source_urls": [],
            "errors": state.get("errors", []) + [f"Planning failed: {e}"],
            "status": "planning_fallback",
        }


def generate_queries(state: ResearchState) -> dict[str, Any]:
    """Node: Generate optimized search queries from sub-questions.

    Invokes the Query Generator agent to create tool-optimized
    queries for web and academic search.
    """
    sub_queries = state.get("sub_queries", [])
    logger.info("node_generate_queries", sub_query_count=len(sub_queries))

    try:
        from arcane.agents.query_generator import create_query_generator_agent
        from crewai import Crew, Process
        from arcane.agents.crew import create_query_generation_task, get_default_llm

        llm = get_default_llm()
        agent = create_query_generator_agent(llm)
        task = create_query_generation_task(sub_queries)
        task.agent = agent

        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
        result = crew.kickoff()

        result_text = str(result)
        parsed = _extract_json(result_text)
        search_queries = parsed.get("queries", [])

        if not search_queries:
            # Fallback: create basic queries from sub-questions
            search_queries = [
                {"query": q, "target": "web", "intent": "general"}
                for q in sub_queries
            ]

        return {
            "search_queries": search_queries,
            "status": "queries_generated",
        }

    except Exception as e:
        logger.error("generate_queries_failed", error=str(e))
        search_queries = [
            {"query": q, "target": "web", "intent": "general"}
            for q in sub_queries
        ]
        return {
            "search_queries": search_queries,
            "errors": state.get("errors", []) + [f"Query generation failed: {e}"],
            "status": "queries_fallback",
        }


def retrieve_and_search(state: ResearchState) -> dict[str, Any]:
    """Node: Execute search queries and retrieve relevant documents.

    Processes the current batch of search queries using the
    Researcher agent and tools.
    """
    search_queries = state.get("search_queries", [])
    current_index = state.get("current_query_index", 0)
    existing_findings = state.get("intermediate_findings", [])
    existing_sources = state.get("source_urls", [])

    # Process queries in batches of 3
    batch_size = 3
    batch = search_queries[current_index : current_index + batch_size]

    if not batch:
        return {"status": "retrieval_complete"}

    logger.info(
        "node_retrieve_and_search",
        batch_start=current_index,
        batch_size=len(batch),
        total_queries=len(search_queries),
    )

    try:
        query_texts = [q["query"] if isinstance(q, dict) else q for q in batch]
        crew = assemble_research_crew(query_texts)
        result = crew.kickoff()

        result_text = str(result)
        parsed = _extract_json(result_text)

        new_findings = parsed.get("findings", [])
        new_sources = parsed.get("sources", [])

        # Extract source URLs
        new_urls = [s.get("url", "") for s in new_sources if s.get("url")]

        return {
            "intermediate_findings": existing_findings + new_findings,
            "source_urls": existing_sources + new_urls,
            "current_query_index": current_index + batch_size,
            "status": "retrieval_in_progress",
        }

    except Exception as e:
        logger.error("retrieve_and_search_failed", error=str(e), batch_start=current_index)
        return {
            "current_query_index": current_index + batch_size,
            "errors": state.get("errors", []) + [f"Retrieval failed for batch {current_index}: {e}"],
            "status": "retrieval_error",
        }


def analyze_findings(state: ResearchState) -> dict[str, Any]:
    """Node: Analyze and consolidate intermediate findings.

    Deduplicates findings, scores relevance, and prepares
    the data for synthesis.
    """
    findings = state.get("intermediate_findings", [])
    logger.info("node_analyze_findings", finding_count=len(findings))

    # Deduplicate by content similarity (simple hash-based)
    seen_hashes: set[str] = set()
    unique_findings = []

    for finding in findings:
        content = str(finding.get("fact", finding.get("content", "")))
        content_hash = hash(content[:200])
        str_hash = str(content_hash)
        if str_hash not in seen_hashes:
            seen_hashes.add(str_hash)
            unique_findings.append(finding)

    return {
        "intermediate_findings": unique_findings,
        "status": "analysis_complete",
    }


def synthesize_report(state: ResearchState) -> dict[str, Any]:
    """Node: Synthesize findings into a comprehensive report.

    Uses the Synthesizer agent to create a draft report from
    all intermediate findings. Incorporates critique feedback
    for revisions.
    """
    query = state["query"]
    findings = state.get("intermediate_findings", [])
    revision_count = state.get("revision_count", 0)
    critique = state.get("critique", {})
    human_feedback = state.get("human_feedback")

    logger.info(
        "node_synthesize_report",
        query=query,
        finding_count=len(findings),
        revision=revision_count,
    )

    # Format findings as text
    findings_text = json.dumps(findings, indent=2, ensure_ascii=False)

    # Build critique feedback for revision
    critique_feedback = None
    if revision_count > 0 and critique:
        issues = critique.get("issues", [])
        suggestions = critique.get("suggestions", [])
        critique_parts = []
        for issue in issues:
            critique_parts.append(
                f"- [{issue.get('severity', 'medium').upper()}] "
                f"{issue.get('category', 'general')}: {issue.get('detail', '')}"
            )
        for suggestion in suggestions:
            critique_parts.append(f"- SUGGESTION: {suggestion}")
        if human_feedback:
            critique_parts.append(f"- HUMAN FEEDBACK: {human_feedback}")
        critique_feedback = "\n".join(critique_parts)

    try:
        crew = assemble_synthesis_crew(query, findings_text, critique_feedback)
        result = crew.kickoff()
        draft = str(result)

        return {
            "draft_report": draft,
            "revision_count": revision_count + 1,
            "human_feedback": None,  # Clear after use
            "status": "synthesis_complete",
        }

    except Exception as e:
        logger.error("synthesize_report_failed", error=str(e))
        return {
            "draft_report": f"Error generating report: {e}",
            "revision_count": revision_count + 1,
            "errors": state.get("errors", []) + [f"Synthesis failed: {e}"],
            "status": "synthesis_error",
        }


def critique_report(state: ResearchState) -> dict[str, Any]:
    """Node: Evaluate the draft report against quality rubrics.

    Uses the Critic agent to score the report and provide
    actionable feedback for improvement.
    """
    query = state["query"]
    draft = state.get("draft_report", "")
    logger.info("node_critique_report", query=query, draft_length=len(draft))

    try:
        crew = assemble_critique_crew(draft, query)
        result = crew.kickoff()

        result_text = str(result)
        critique = _extract_json(result_text)

        score = critique.get("score", 0.0)
        passed = critique.get("passed", False)

        # Ensure score is a float between 0 and 1
        if isinstance(score, (int, float)):
            if score > 1:
                score = score / 10.0  # Handle 0-10 scale
            score = max(0.0, min(1.0, float(score)))
        else:
            score = 0.0

        return {
            "critique": critique,
            "critique_score": score,
            "status": "critique_complete",
        }

    except Exception as e:
        logger.error("critique_report_failed", error=str(e))
        return {
            "critique": {"score": 0.5, "passed": False, "issues": [], "suggestions": []},
            "critique_score": 0.5,
            "errors": state.get("errors", []) + [f"Critique failed: {e}"],
            "status": "critique_error",
        }


def finalize_report(state: ResearchState) -> dict[str, Any]:
    """Node: Finalize the approved report with metadata and citations.

    Formats the final report with metadata header, proper citations,
    and stores it in the session.
    """
    from arcane.utils.formatting import format_report_metadata, format_citations_section

    query = state["query"]
    session_id = state.get("session_id", "unknown")
    draft = state.get("draft_report", "")
    score = state.get("critique_score", 0.0)
    revision_count = state.get("revision_count", 0)
    source_urls = state.get("source_urls", [])

    logger.info(
        "node_finalize_report",
        query=query,
        score=score,
        revisions=revision_count,
    )

    # Build metadata header
    metadata = format_report_metadata(query, session_id, revision_count, score)

    # Build citations from sources
    citations = []
    for i, url in enumerate(set(source_urls), 1):
        citations.append({"index": i, "url": url})

    final_report = f"{metadata}\n\n{draft}"

    return {
        "final_report": final_report,
        "citations": citations,
        "status": "complete",
    }


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from agent output text.

    Handles cases where the JSON is embedded in markdown code blocks
    or surrounded by other text.
    """
    import re

    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code blocks
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON between braces
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # Give up — return as raw text
    logger.warning("json_extraction_failed", text_preview=text[:200])
    return {"raw_output": text}
