"""Cohere Reranker tool — rerank search results by relevance."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from arcane.config import get_settings
from arcane.utils.logging import get_logger
from arcane.utils.retry import retry_on_api_error

logger = get_logger(__name__)


class CohereRerankerTool(BaseTool):
    """Rerank a list of documents by relevance to a query using Cohere Rerank.

    Takes a query and candidate documents, returns them sorted by relevance
    with confidence scores. Critical for filtering noise from search results.
    """

    name: str = "reranker"
    description: str = (
        "Rerank a list of search results by relevance to the original query. "
        "Input should be a JSON string with 'query' and 'documents' fields. "
        "Returns the documents sorted by relevance with confidence scores."
    )
    top_k: int = Field(default=5, description="Number of top results to return")
    model: str = Field(default="rerank-v3.5", description="Cohere rerank model")

    @retry_on_api_error(max_attempts=3)
    def _run(self, input_str: str) -> str:
        """Rerank documents using Cohere."""
        import cohere

        logger.info("reranker_executing")

        try:
            # Parse input
            if isinstance(input_str, str):
                input_data = json.loads(input_str)
            else:
                input_data = input_str

            query = input_data.get("query", "")
            documents = input_data.get("documents", [])

            if not query or not documents:
                return json.dumps({"error": "Both 'query' and 'documents' are required", "results": []})

            # Extract text from document objects
            doc_texts = []
            for doc in documents:
                if isinstance(doc, str):
                    doc_texts.append(doc)
                elif isinstance(doc, dict):
                    # Use snippet, content, or abstract — whichever is available
                    text = doc.get("snippet") or doc.get("content") or doc.get("abstract") or doc.get("title", "")
                    doc_texts.append(text)

            if not doc_texts:
                return json.dumps({"error": "No valid document text found", "results": []})

            # Call Cohere Rerank
            settings = get_settings()
            client = cohere.Client(api_key=settings.cohere_api_key)

            response = client.rerank(
                query=query,
                documents=doc_texts,
                top_n=min(self.top_k, len(doc_texts)),
                model=self.model,
            )

            # Map results back to original documents
            reranked = []
            for result in response.results:
                idx = result.index
                original_doc = documents[idx] if idx < len(documents) else {}

                if isinstance(original_doc, str):
                    original_doc = {"content": original_doc}

                reranked.append({
                    **original_doc,
                    "relevance_score": result.relevance_score,
                    "original_index": idx,
                })

            logger.info("reranker_complete", input_count=len(documents), output_count=len(reranked))
            return json.dumps(reranked, ensure_ascii=False)

        except json.JSONDecodeError as e:
            logger.error("reranker_parse_error", error=str(e))
            return json.dumps({"error": f"Invalid JSON input: {e}", "results": []})
        except Exception as e:
            logger.error("reranker_failed", error=str(e))
            return json.dumps({"error": str(e), "results": []})

    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)
