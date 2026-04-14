"""Arcane application entry point — CLI and programmatic access."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from arcane.config import get_settings
from arcane.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def run_research(query: str, session_id: str | None = None) -> dict:
    """Run a research session programmatically.

    Args:
        query: The research question to investigate.
        session_id: Optional session ID. Generated if not provided.

    Returns:
        Dict containing the research results.
    """
    from arcane.graph.builder import compile_research_graph
    from arcane.graph.checkpointer import create_memory_checkpointer

    settings = get_settings()
    sid = session_id or str(uuid.uuid4())

    logger.info("starting_research", query=query, session_id=sid)

    checkpointer = create_memory_checkpointer()
    graph = compile_research_graph(checkpointer=checkpointer)

    initial_state = {
        "query": query,
        "session_id": sid,
        "max_revisions": settings.max_revisions,
        "human_review_requested": False,
        "revision_count": 0,
        "intermediate_findings": [],
        "search_results": [],
        "source_urls": [],
        "errors": [],
        "status": "starting",
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": sid}},
    )

    logger.info("research_complete", session_id=sid, status=result.get("status"))
    return result


def cli() -> None:
    """Command-line interface for Arcane."""
    parser = argparse.ArgumentParser(
        prog="arcane",
        description="Arcane — Agentic Research Intelligence Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- research command ---
    research_parser = subparsers.add_parser("research", help="Run a research query")
    research_parser.add_argument("query", type=str, help="Research question to investigate")
    research_parser.add_argument(
        "--session-id", type=str, default=None, help="Session ID (auto-generated if omitted)"
    )
    research_parser.add_argument(
        "--output", "-o", type=str, default=None, help="Save report to file"
    )

    # --- serve command ---
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", type=str, default=None, help="Override API host")
    serve_parser.add_argument("--port", type=int, default=None, help="Override API port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # --- health command ---
    subparsers.add_parser("health", help="Check service health")

    args = parser.parse_args()
    settings = get_settings()
    setup_logging(settings.log_level)

    if args.command == "research":
        result = asyncio.run(run_research(args.query, args.session_id))

        report = result.get("final_report", "No report generated.")
        score = result.get("critique_score", 0)
        errors = result.get("errors", [])

        # Print report
        print("\n" + "═" * 72)
        print(report)
        print("═" * 72)
        print(f"\nQuality Score: {score:.0%}")
        print(f"Revisions: {result.get('revision_count', 0)}")
        if errors:
            print(f"Errors: {len(errors)}")
            for e in errors:
                print(f"  ⚠ {e}")

        # Save to file if requested
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"\nReport saved to: {args.output}")

    elif args.command == "serve":
        import uvicorn

        host = args.host or settings.api_host
        port = args.port or settings.api_port
        reload = args.reload or settings.is_development

        logger.info("starting_api_server", host=host, port=port, reload=reload)
        uvicorn.run(
            "arcane.api.app:create_app",
            host=host,
            port=port,
            reload=reload,
            factory=True,
        )

    elif args.command == "health":
        import redis

        print("Arcane Health Check")
        print("─" * 40)

        # Check Redis
        try:
            client = redis.from_url(settings.redis_url, decode_responses=True)
            client.ping()
            print(f"✓ Redis: connected ({settings.redis_url})")
        except Exception as e:
            print(f"✗ Redis: failed ({e})")

        # Check Cohere API key
        if settings.cohere_api_key and settings.cohere_api_key != "your-cohere-api-key-here":
            print(f"✓ Cohere API key: configured")
        else:
            print(f"✗ Cohere API key: not configured")

        print(f"  Environment: {settings.environment}")
        print(f"  Log level: {settings.log_level}")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli()
