"""Configuration management for Arcane using Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Cohere API ---
    cohere_api_key: str = ""

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Application ---
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    # --- Research Defaults ---
    max_search_results: int = 10
    max_revisions: int = 3
    critique_threshold: float = 0.8
    cache_ttl_hours: int = 24

    # --- API Server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Optional API Keys ---
    semantic_scholar_api_key: str | None = None

    # --- Derived Paths ---
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
