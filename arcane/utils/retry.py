"""Retry and backoff decorators for external API calls."""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from arcane.utils.logging import get_logger

logger = get_logger(__name__)


def retry_on_api_error(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 30.0,
):
    """Decorator for retrying external API calls with exponential backoff.

    Retries on common transient errors: connection errors, timeouts, rate limits.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type((
            ConnectionError,
            TimeoutError,
            OSError,
        )),
        before_sleep=before_sleep_log(logger, "WARNING"),  # type: ignore[arg-type]
        reraise=True,
    )
