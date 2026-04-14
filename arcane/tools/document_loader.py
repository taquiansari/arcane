"""Document loader tool — PDF and text file ingestion with chunking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import Field

from arcane.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentLoaderTool(BaseTool):
    """Load and extract text from PDF files and other documents.

    Supports PDF files with automatic text chunking for
    efficient processing and embedding.
    """

    name: str = "document_loader"
    description: str = (
        "Load and extract text from a PDF or text document file. "
        "Returns the document content split into manageable chunks. "
        "Input should be a valid file path to a PDF or text file."
    )
    chunk_size: int = Field(default=1000, description="Target size for text chunks (chars)")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks (chars)")

    def _run(self, file_path: str) -> str:
        """Load and chunk a document."""
        logger.info("document_loader_executing", file_path=file_path)

        path = Path(file_path)
        if not path.exists():
            return json.dumps({"error": f"File not found: {file_path}", "chunks": []})

        try:
            if path.suffix.lower() == ".pdf":
                content = self._load_pdf(path)
            elif path.suffix.lower() in (".txt", ".md", ".rst"):
                content = path.read_text(encoding="utf-8")
            else:
                return json.dumps({
                    "error": f"Unsupported file type: {path.suffix}",
                    "chunks": [],
                })

            if not content.strip():
                return json.dumps({"error": "No text content extracted", "chunks": []})

            chunks = self._chunk_text(content)

            result = {
                "file_path": str(path),
                "file_name": path.name,
                "total_length": len(content),
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "index": i,
                        "content": chunk,
                        "length": len(chunk),
                    }
                    for i, chunk in enumerate(chunks)
                ],
            }

            logger.info(
                "document_loader_complete",
                file_path=file_path,
                chunk_count=len(chunks),
                total_length=len(content),
            )
            return json.dumps(result, ensure_ascii=False)

        except Exception as e:
            logger.error("document_loader_failed", file_path=file_path, error=str(e))
            return json.dumps({"error": str(e), "chunks": []})

    def _load_pdf(self, path: Path) -> str:
        """Extract text from a PDF file using pypdf."""
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks at sentence boundaries."""
        # Simple sentence-aware chunking
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ".!?" and len(current) > 50:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap
                overlap_text = current_chunk[-self.chunk_overlap :] if self.chunk_overlap > 0 else ""
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    async def _arun(self, file_path: str) -> str:
        return self._run(file_path)
