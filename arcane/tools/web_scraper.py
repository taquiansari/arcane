"""Web scraper tool — clean text extraction from URLs."""

from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.tools import BaseTool
from pydantic import Field

from arcane.utils.logging import get_logger
from arcane.utils.retry import retry_on_api_error

logger = get_logger(__name__)


class WebScraperTool(BaseTool):
    """Extract clean text content from a web page URL.

    Uses trafilatura for high-quality content extraction,
    falling back to BeautifulSoup if needed.
    """

    name: str = "web_scraper"
    description: str = (
        "Extract the main text content from a web page URL. "
        "Use this to read the full content of articles, blog posts, "
        "documentation pages, or any web page found during search. "
        "Input should be a valid URL."
    )
    max_content_length: int = Field(
        default=8000,
        description="Maximum characters of content to return",
    )

    @retry_on_api_error(max_attempts=2)
    def _run(self, url: str) -> str:
        """Extract content from a URL."""
        logger.info("web_scraper_executing", url=url)

        try:
            # Fetch the page
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }

            with httpx.Client(timeout=20.0, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                html = response.text

            # Try trafilatura first (best quality extraction)
            content = None
            title = ""

            try:
                import trafilatura

                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    content = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        favor_precision=True,
                    )
                    metadata = trafilatura.extract_metadata(downloaded)
                    if metadata:
                        title = metadata.title or ""
            except Exception:
                pass

            # Fallback to BeautifulSoup
            if not content:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(html, "html.parser")

                # Get title
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""

                # Remove script, style, nav, footer elements
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()

                # Get main content
                main = soup.find("main") or soup.find("article") or soup.find("body")
                if main:
                    content = main.get_text(separator="\n", strip=True)
                else:
                    content = soup.get_text(separator="\n", strip=True)

            if not content:
                return json.dumps({
                    "url": url,
                    "error": "Could not extract content from this URL",
                    "content": "",
                })

            # Truncate if needed
            if len(content) > self.max_content_length:
                content = content[: self.max_content_length] + "\n\n[Content truncated...]"

            result = {
                "url": url,
                "title": title,
                "content": content,
                "content_length": len(content),
                "source": "web_scraper",
            }

            logger.info(
                "web_scraper_complete",
                url=url,
                title=title[:50],
                content_length=len(content),
            )
            return json.dumps(result, ensure_ascii=False)

        except httpx.HTTPStatusError as e:
            logger.error("web_scraper_http_error", url=url, status=e.response.status_code)
            return json.dumps({"url": url, "error": f"HTTP {e.response.status_code}", "content": ""})
        except Exception as e:
            logger.error("web_scraper_failed", url=url, error=str(e))
            return json.dumps({"url": url, "error": str(e), "content": ""})

    async def _arun(self, url: str) -> str:
        return self._run(url)
