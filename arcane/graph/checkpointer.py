"""Redis-based checkpointer for LangGraph state persistence."""

from __future__ import annotations

from typing import Any

from arcane.config import get_settings
from arcane.utils.logging import get_logger

logger = get_logger(__name__)


def create_redis_checkpointer() -> Any | None:
    """Create a Redis-backed checkpointer for LangGraph.

    Enables crash recovery — if the pipeline fails mid-execution,
    it can resume from the last checkpointed state.

    Returns:
        LangGraph checkpointer instance, or None if unavailable.
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver

        settings = get_settings()
        checkpointer = RedisSaver.from_conn_string(settings.redis_url)
        logger.info("redis_checkpointer_created")
        return checkpointer
    except ImportError:
        logger.warning(
            "redis_checkpointer_unavailable",
            detail="langgraph-checkpoint-redis not installed. Using in-memory state.",
        )
        return None
    except Exception as e:
        logger.warning("redis_checkpointer_failed", error=str(e))
        return None


def create_memory_checkpointer() -> Any:
    """Create an in-memory checkpointer as fallback.

    Returns:
        MemorySaver checkpointer instance.
    """
    from langgraph.checkpoint.memory import MemorySaver

    logger.info("memory_checkpointer_created")
    return MemorySaver()
