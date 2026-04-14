"""Academic paper search tools — Semantic Scholar and arXiv APIs."""

from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import Field

from arcane.utils.logging import get_logger
from arcane.utils.retry import retry_on_api_error

logger = get_logger(__name__)

SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_API_BASE = "http://export.arxiv.org/api/query"


class SemanticScholarSearchTool(BaseTool):
    """Search academic papers via Semantic Scholar API.

    Returns structured paper metadata including title, abstract,
    authors, year, citation count, and URLs.
    """

    name: str = "academic_search"
    description: str = (
        "Search for academic research papers on any scientific topic. "
        "Returns paper titles, abstracts, authors, publication year, "
        "citation counts, and URLs. Use this for rigorous academic research, "
        "finding peer-reviewed sources, and understanding the state of the art."
    )
    max_results: int = Field(default=10, description="Maximum papers to return")
    api_key: str | None = Field(default=None, description="Optional Semantic Scholar API key")

    @retry_on_api_error(max_attempts=3)
    def _run(self, query: str) -> str:
        """Search Semantic Scholar for academic papers."""
        logger.info("academic_search_executing", query=query, max_results=self.max_results)

        try:
            headers = {}
            if self.api_key:
                headers["x-api-key"] = self.api_key

            params = {
                "query": query,
                "limit": self.max_results,
                "fields": "title,abstract,authors,year,citationCount,url,openAccessPdf,venue,externalIds",
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                data = response.json()

            papers = []
            for paper in data.get("data", []):
                authors_list = paper.get("authors", [])
                author_names = ", ".join(a.get("name", "") for a in authors_list[:5])
                if len(authors_list) > 5:
                    author_names += " et al."

                # Try to get the best URL
                url = paper.get("url", "")
                pdf_url = ""
                open_access = paper.get("openAccessPdf")
                if open_access and isinstance(open_access, dict):
                    pdf_url = open_access.get("url", "")

                # Get DOI if available
                external_ids = paper.get("externalIds", {}) or {}
                doi = external_ids.get("DOI", "")

                papers.append({
                    "title": paper.get("title", "Untitled"),
                    "abstract": (paper.get("abstract") or "")[:500],
                    "authors": author_names,
                    "year": paper.get("year"),
                    "citation_count": paper.get("citationCount", 0),
                    "venue": paper.get("venue", ""),
                    "url": url,
                    "pdf_url": pdf_url,
                    "doi": doi,
                    "source": "semantic_scholar",
                })

            logger.info("academic_search_complete", query=query, result_count=len(papers))
            return json.dumps(papers, ensure_ascii=False)

        except httpx.HTTPStatusError as e:
            logger.error("academic_search_http_error", status=e.response.status_code, query=query)
            return json.dumps({"error": f"HTTP {e.response.status_code}", "results": []})
        except Exception as e:
            logger.error("academic_search_failed", query=query, error=str(e))
            return json.dumps({"error": str(e), "results": []})

    async def _arun(self, query: str) -> str:
        return self._run(query)


class ArxivSearchTool(BaseTool):
    """Search arXiv for preprints and research papers."""

    name: str = "arxiv_search"
    description: str = (
        "Search arXiv for preprints and research papers in physics, "
        "mathematics, computer science, biology, and more. "
        "Use this for cutting-edge research that may not yet be peer-reviewed."
    )
    max_results: int = Field(default=10, description="Maximum papers to return")

    @retry_on_api_error(max_attempts=3)
    def _run(self, query: str) -> str:
        """Search arXiv API for papers."""
        import xml.etree.ElementTree as ET

        logger.info("arxiv_search_executing", query=query)

        try:
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            with httpx.Client(timeout=30.0) as client:
                response = client.get(ARXIV_API_BASE, params=params)
                response.raise_for_status()

            # Parse Atom XML
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            root = ET.fromstring(response.text)
            entries = root.findall("atom:entry", ns)

            papers = []
            for entry in entries:
                title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
                abstract = (entry.findtext("atom:summary", "", ns) or "").strip()[:500]
                published = entry.findtext("atom:published", "", ns)
                year = published[:4] if published else ""

                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.findtext("atom:name", "", ns)
                    if name:
                        authors.append(name)

                author_str = ", ".join(authors[:5])
                if len(authors) > 5:
                    author_str += " et al."

                # Get PDF link
                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break

                # Get arXiv ID
                entry_id = entry.findtext("atom:id", "", ns)
                arxiv_id = entry_id.split("/abs/")[-1] if entry_id else ""

                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "authors": author_str,
                    "year": year,
                    "url": entry_id,
                    "pdf_url": pdf_url,
                    "arxiv_id": arxiv_id,
                    "source": "arxiv",
                })

            logger.info("arxiv_search_complete", query=query, result_count=len(papers))
            return json.dumps(papers, ensure_ascii=False)

        except Exception as e:
            logger.error("arxiv_search_failed", query=query, error=str(e))
            return json.dumps({"error": str(e), "results": []})

    async def _arun(self, query: str) -> str:
        return self._run(query)
