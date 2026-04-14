"""DuckDuckGo web search tool for Arcane."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from arcane.utils.logging import get_logger
from arcane.utils.retry import retry_on_api_error

logger = get_logger(__name__)


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo for current information.

    Returns structured results with titles, snippets, and URLs.
    Free to use — no API key required.
    """

    name: str = "web_search"
    description: str = (
        "Search the web for current information on any topic. "
        "Returns a list of results with titles, snippets, and URLs. "
        "Use this for general web research, news, blog posts, and documentation."
    )
    max_results: int = Field(default=10, description="Maximum number of results to return")

    @retry_on_api_error(max_attempts=3)
    def _run(self, query: str) -> str:
        """Execute a DuckDuckGo web search."""
        from duckduckgo_search import DDGS

        logger.info("web_search_executing", query=query, max_results=self.max_results)

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.text(
                    keywords=query,
                    max_results=self.max_results,
                    safesearch="moderate",
                ))

            results = []
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                    "source": "web",
                })

            logger.info("web_search_complete", query=query, result_count=len(results))
            return json.dumps(results, ensure_ascii=False)

        except Exception as e:
            logger.error("web_search_failed", query=query, error=str(e))
            return json.dumps({"error": str(e), "results": []})

    async def _arun(self, query: str) -> str:
        """Async version — delegates to sync for DuckDuckGo."""
        return self._run(query)


class WebNewsSearchTool(BaseTool):
    """Search DuckDuckGo News for recent news articles."""

    name: str = "news_search"
    description: str = (
        "Search for recent news articles on any topic. "
        "Returns news results with titles, snippets, dates, and URLs. "
        "Use this when the user needs current events or recent developments."
    )
    max_results: int = Field(default=10, description="Maximum number of results to return")

    @retry_on_api_error(max_attempts=3)
    def _run(self, query: str) -> str:
        """Execute a DuckDuckGo news search."""
        from duckduckgo_search import DDGS

        logger.info("news_search_executing", query=query)

        try:
            with DDGS() as ddgs:
                raw_results = list(ddgs.news(
                    keywords=query,
                    max_results=self.max_results,
                    safesearch="moderate",
                ))

            results = []
            for r in raw_results:
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("url", ""),
                    "date": r.get("date", ""),
                    "source_name": r.get("source", ""),
                    "source": "news",
                })

            logger.info("news_search_complete", query=query, result_count=len(results))
            return json.dumps(results, ensure_ascii=False)

        except Exception as e:
            logger.error("news_search_failed", query=query, error=str(e))
            return json.dumps({"error": str(e), "results": []})

    async def _arun(self, query: str) -> str:
        return self._run(query)
