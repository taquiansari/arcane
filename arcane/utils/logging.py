"""Structured logging setup for Arcane using structlog."""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging for the application.

    Uses structlog for rich, structured log output in development
    and JSON-formatted output in production.
    """
    # Determine if we're in a TTY (development) or piped (production)
    is_tty = sys.stderr.isatty()

    # Shared processors for both stdlib and structlog
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if is_tty:
        # Pretty console output for development
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
            pad_event=40,
        )
    else:
        # JSON output for production / log aggregation
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the stdlib logging to use structlog formatting
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Suppress noisy third-party loggers
    for noisy_logger in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a named structlog logger instance."""
    return structlog.get_logger(name)
