"""Output formatting helpers for Arcane research reports."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone


def format_citation(index: int, source: dict) -> str:
    """Format a single citation in a consistent style.

    Args:
        index: Citation number (1-based).
        source: Dict with keys like 'title', 'url', 'authors', 'year'.

    Returns:
        Formatted citation string.
    """
    title = source.get("title", "Untitled")
    url = source.get("url", "")
    authors = source.get("authors", "")
    year = source.get("year", "")

    parts = [f"[{index}]"]
    if authors:
        parts.append(f"{authors}.")
    parts.append(f'"{title}."')
    if year:
        parts.append(f"({year}).")
    if url:
        parts.append(f"Available at: {url}")

    return " ".join(parts)


def format_citations_section(sources: list[dict]) -> str:
    """Format a full citations/references section."""
    if not sources:
        return ""

    lines = ["## References\n"]
    for i, source in enumerate(sources, 1):
        lines.append(format_citation(i, source))
    return "\n".join(lines)


def format_report_metadata(
    query: str,
    session_id: str,
    revision_count: int,
    critique_score: float,
) -> str:
    """Format metadata header for a research report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"---\n"
        f"**Research Query:** {query}\n"
        f"**Session:** `{session_id}`\n"
        f"**Generated:** {now}\n"
        f"**Revisions:** {revision_count}\n"
        f"**Quality Score:** {critique_score:.0%}\n"
        f"---\n"
    )


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to a maximum length, preserving word boundaries."""
    if len(text) <= max_length:
        return text
    truncated = text[: max_length - len(suffix)]
    # Cut at last space to avoid breaking words
    last_space = truncated.rfind(" ")
    if last_space > max_length // 2:
        truncated = truncated[:last_space]
    return truncated + suffix


def clean_text(text: str) -> str:
    """Clean extracted text by removing excessive whitespace and artifacts."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove common web artifacts
    text = re.sub(r"(cookie|privacy|subscribe|newsletter).*?\.", "", text, flags=re.IGNORECASE)
    return text.strip()


def to_json_safe(obj: object) -> str:
    """Serialize object to JSON, handling non-serializable types gracefully."""
    def default_handler(o: object) -> str:
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)

    return json.dumps(obj, indent=2, default=default_handler, ensure_ascii=False)
