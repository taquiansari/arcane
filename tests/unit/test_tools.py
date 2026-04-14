"""Unit tests for Arcane tools."""

import json
import pytest


class TestWebSearchTool:
    """Tests for the DuckDuckGo web search tool."""

    def test_tool_metadata(self):
        from arcane.tools.web_search import WebSearchTool

        tool = WebSearchTool()
        assert tool.name == "web_search"
        assert "search" in tool.description.lower()
        assert tool.max_results == 10

    def test_tool_custom_max_results(self):
        from arcane.tools.web_search import WebSearchTool

        tool = WebSearchTool(max_results=5)
        assert tool.max_results == 5

    def test_news_search_tool_metadata(self):
        from arcane.tools.web_search import WebNewsSearchTool

        tool = WebNewsSearchTool()
        assert tool.name == "news_search"
        assert "news" in tool.description.lower()


class TestAcademicSearchTool:
    """Tests for academic search tools."""

    def test_semantic_scholar_metadata(self):
        from arcane.tools.academic_search import SemanticScholarSearchTool

        tool = SemanticScholarSearchTool()
        assert tool.name == "academic_search"
        assert "academic" in tool.description.lower()

    def test_arxiv_metadata(self):
        from arcane.tools.academic_search import ArxivSearchTool

        tool = ArxivSearchTool()
        assert tool.name == "arxiv_search"
        assert "arxiv" in tool.description.lower()


class TestWebScraperTool:
    """Tests for the web scraper tool."""

    def test_scraper_metadata(self):
        from arcane.tools.web_scraper import WebScraperTool

        tool = WebScraperTool()
        assert tool.name == "web_scraper"
        assert tool.max_content_length == 8000


class TestDocumentLoaderTool:
    """Tests for the document loader tool."""

    def test_loader_metadata(self):
        from arcane.tools.document_loader import DocumentLoaderTool

        tool = DocumentLoaderTool()
        assert tool.name == "document_loader"
        assert tool.chunk_size == 1000
        assert tool.chunk_overlap == 200

    def test_chunk_text(self):
        from arcane.tools.document_loader import DocumentLoaderTool

        tool = DocumentLoaderTool(chunk_size=100, chunk_overlap=20)
        text = "This is sentence one. " * 20
        chunks = tool._chunk_text(text)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    def test_file_not_found(self):
        from arcane.tools.document_loader import DocumentLoaderTool

        tool = DocumentLoaderTool()
        result = json.loads(tool._run("/nonexistent/path/file.pdf"))
        assert "error" in result

    def test_unsupported_format(self, tmp_path):
        from arcane.tools.document_loader import DocumentLoaderTool

        # Create a dummy file with unsupported extension
        dummy = tmp_path / "test.xyz"
        dummy.write_text("test content")

        tool = DocumentLoaderTool()
        result = json.loads(tool._run(str(dummy)))
        assert "error" in result


class TestRerankerTool:
    """Tests for the Cohere reranker tool."""

    def test_reranker_metadata(self):
        from arcane.tools.reranker import CohereRerankerTool

        tool = CohereRerankerTool()
        assert tool.name == "reranker"
        assert tool.top_k == 5
        assert tool.model == "rerank-v3.5"

    def test_reranker_invalid_json(self):
        from arcane.tools.reranker import CohereRerankerTool

        tool = CohereRerankerTool()
        result = json.loads(tool._run("not valid json"))
        assert "error" in result

    def test_reranker_missing_fields(self):
        from arcane.tools.reranker import CohereRerankerTool

        tool = CohereRerankerTool()
        result = json.loads(tool._run(json.dumps({"query": "test"})))
        assert "error" in result
