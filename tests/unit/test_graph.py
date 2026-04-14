"""Unit tests for LangGraph orchestration."""

import pytest


class TestResearchState:
    """Tests for the state schema."""

    def test_state_schema_exists(self):
        from arcane.graph.state import ResearchState
        assert ResearchState is not None

    def test_state_annotations(self):
        from arcane.graph.state import ResearchState
        annotations = ResearchState.__annotations__
        assert "query" in annotations
        assert "session_id" in annotations
        assert "research_plan" in annotations
        assert "sub_queries" in annotations
        assert "draft_report" in annotations
        assert "critique_score" in annotations
        assert "final_report" in annotations
        assert "status" in annotations
        assert "human_review_requested" in annotations


class TestGraphEdges:
    """Tests for conditional edge functions."""

    def test_should_continue_retrieval_yes(self):
        from arcane.graph.edges import should_continue_retrieval

        state = {
            "search_queries": [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}],
            "current_query_index": 0,
        }
        assert should_continue_retrieval(state) == "retrieve"

    def test_should_continue_retrieval_no(self):
        from arcane.graph.edges import should_continue_retrieval

        state = {
            "search_queries": [{"query": "q1"}],
            "current_query_index": 1,
        }
        assert should_continue_retrieval(state) == "analyze"

    def test_should_revise_finalize_on_pass(self):
        from arcane.graph.edges import should_revise_or_finalize

        state = {
            "critique_score": 0.9,
            "revision_count": 1,
            "max_revisions": 3,
            "critique": {"passed": True},
            "human_review_requested": False,
        }
        assert should_revise_or_finalize(state) == "finalize"

    def test_should_revise_on_low_score(self):
        from arcane.graph.edges import should_revise_or_finalize

        state = {
            "critique_score": 0.5,
            "revision_count": 1,
            "max_revisions": 3,
            "critique": {"passed": False},
            "human_review_requested": False,
        }
        assert should_revise_or_finalize(state) == "revise"

    def test_should_finalize_on_max_revisions(self):
        from arcane.graph.edges import should_revise_or_finalize

        state = {
            "critique_score": 0.5,
            "revision_count": 3,
            "max_revisions": 3,
            "critique": {"passed": False},
            "human_review_requested": False,
        }
        assert should_revise_or_finalize(state) == "finalize"


class TestGraphBuilder:
    """Tests for the graph builder."""

    def test_build_research_graph(self):
        from arcane.graph.builder import build_research_graph

        graph = build_research_graph()
        assert graph is not None

    def test_graph_has_all_nodes(self):
        from arcane.graph.builder import build_research_graph

        graph = build_research_graph()
        node_names = set(graph.nodes.keys())
        expected = {"plan", "generate_queries", "retrieve", "analyze", "synthesize", "critique", "finalize"}
        assert expected.issubset(node_names)


class TestNodeHelpers:
    """Tests for node helper functions."""

    def test_extract_json_direct(self):
        from arcane.graph.nodes import _extract_json
        import json

        data = {"key": "value", "num": 42}
        result = _extract_json(json.dumps(data))
        assert result == data

    def test_extract_json_from_code_block(self):
        from arcane.graph.nodes import _extract_json

        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_from_braces(self):
        from arcane.graph.nodes import _extract_json

        text = 'Some text before {"key": "value"} and after'
        result = _extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_fallback(self):
        from arcane.graph.nodes import _extract_json

        text = "No JSON here at all"
        result = _extract_json(text)
        assert "raw_output" in result
