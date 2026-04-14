"""Shared test fixtures for Arcane."""

import pytest


@pytest.fixture
def sample_query() -> str:
    return "What are the latest advances in protein folding prediction using AI?"


@pytest.fixture
def sample_search_results() -> list[dict]:
    return [
        {
            "title": "AlphaFold3: Advances in Protein Structure Prediction",
            "url": "https://example.com/alphafold3",
            "snippet": "AlphaFold3 introduces significant improvements...",
            "source": "web",
        },
        {
            "title": "Comparing RoseTTAFold and AlphaFold2",
            "url": "https://example.com/comparison",
            "snippet": "A comprehensive comparison of modern protein folding methods...",
            "source": "academic",
        },
    ]
