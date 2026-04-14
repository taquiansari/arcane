"""Researcher Agent — executes searches and extracts relevant information."""

from __future__ import annotations

from crewai import Agent
from crewai.tools import BaseTool as CrewBaseTool
from pydantic import Field
from typing import Type

from arcane.config import get_settings


# ── CrewAI-compatible tool wrappers ──────────────────────────
# CrewAI expects its own BaseTool subclass, not LangChain's.
# These thin wrappers delegate to the underlying LangChain tools.


class CrewWebSearch(CrewBaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web for current information on any topic. "
        "Returns a list of results with titles, snippets, and URLs."
    )

    def _run(self, query: str) -> str:
        from arcane.tools.web_search import WebSearchTool
        return WebSearchTool()._run(query)


class CrewNewsSearch(CrewBaseTool):
    name: str = "news_search"
    description: str = "Search for recent news articles on any topic."

    def _run(self, query: str) -> str:
        from arcane.tools.web_search import WebNewsSearchTool
        return WebNewsSearchTool()._run(query)


class CrewAcademicSearch(CrewBaseTool):
    name: str = "academic_search"
    description: str = (
        "Search for academic research papers. Returns titles, abstracts, "
        "authors, year, citation counts, and URLs."
    )

    def _run(self, query: str) -> str:
        from arcane.tools.academic_search import SemanticScholarSearchTool
        settings = get_settings()
        return SemanticScholarSearchTool(api_key=settings.semantic_scholar_api_key)._run(query)


class CrewArxivSearch(CrewBaseTool):
    name: str = "arxiv_search"
    description: str = "Search arXiv for preprints and research papers."

    def _run(self, query: str) -> str:
        from arcane.tools.academic_search import ArxivSearchTool
        return ArxivSearchTool()._run(query)


class CrewWebScraper(CrewBaseTool):
    name: str = "web_scraper"
    description: str = (
        "Extract the main text content from a web page URL. "
        "Input should be a valid URL."
    )

    def _run(self, url: str) -> str:
        from arcane.tools.web_scraper import WebScraperTool
        return WebScraperTool()._run(url)


class CrewReranker(CrewBaseTool):
    name: str = "reranker"
    description: str = (
        "Rerank a list of search results by relevance. "
        "Input should be a JSON string with 'query' and 'documents' fields."
    )

    def _run(self, input_str: str) -> str:
        from arcane.tools.reranker import CohereRerankerTool
        return CohereRerankerTool()._run(input_str)


def create_researcher_agent(llm: object | None = None) -> Agent:
    """Create the Deep Researcher agent.

    The Researcher executes search queries across multiple sources,
    reads full articles, reranks results, and extracts key findings
    with source attribution.

    Args:
        llm: Optional LLM instance.

    Returns:
        Configured CrewAI Agent with search and scraping tools.
    """
    tools = [
        CrewWebSearch(),
        CrewNewsSearch(),
        CrewAcademicSearch(),
        CrewArxivSearch(),
        CrewWebScraper(),
        CrewReranker(),
    ]

    return Agent(
        role="Deep Research Specialist",
        goal=(
            "Execute search queries across web and academic sources, read and "
            "extract the most relevant information from each source, and compile "
            "findings with precise source attribution. Always verify claims across "
            "multiple sources when possible."
        ),
        backstory=(
            "You are an information retrieval expert with a background in library "
            "science and data journalism. You are skilled at mining both academic "
            "databases (Semantic Scholar, arXiv) and the open web (DuckDuckGo) to "
            "find the most authoritative and relevant sources. You are known for "
            "finding obscure but highly relevant papers and articles that others "
            "miss. You always attribute information to its source and flag when "
            "a claim comes from a single unverified source. You use the reranker "
            "tool to filter noise and prioritize the most relevant results."
        ),
        verbose=True,
        allow_delegation=False,
        tools=tools,
        llm=llm,
    )
