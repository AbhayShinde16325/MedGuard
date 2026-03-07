"""RAG Ingestion Pipeline for MarketMinds Chatbot.

This module handles the ingestion of documents into the RAG system:
- Loading docs (PDF, raw text)
- Cleaning text (whitespace, noise)
- Splitting into semantically meaningful chunks with overlap

It does NOT:
- Create embeddings
- Store vectors
- Call LLMs
"""

import re
import logging
from pathlib import Path
from typing import List

from backend.app.utils.pdf_utils import extract_text_from_pdf

logger = logging.getLogger(__name__)


class DocumentIngestor:
    """Responsible for ingesting and preprocessing documents for RAG."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        """
        Args:
            chunk_size:    Target character count per chunk.
            chunk_overlap: Number of characters to overlap between consecutive
                           chunks so that context at boundaries isn't lost.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_pdf(self, pdf_path: Path) -> List[str]:
        """Ingest a PDF document and split into chunks suitable for RAG.

        Args:
            pdf_path: Path to the PDF document.

        Returns:
            List of text chunks.
        """
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = self.ingest_text(raw_text)
        logger.info(
            "Ingested %s → %d chunks (avg %d chars)",
            pdf_path.name,
            len(chunks),
            int(sum(len(c) for c in chunks) / max(len(chunks), 1)),
        )
        return chunks

    def ingest_text(self, raw_text: str) -> List[str]:
        """Ingest raw text and split into chunks suitable for RAG.

        Args:
            raw_text: The raw text document to ingest.

        Returns:
            List of text chunks.
        """
        cleaned_text = self._clean_text(raw_text)
        sentences = self._split_into_sentences(cleaned_text)
        chunks = self._merge_sentences_into_chunks(sentences)
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean raw text by collapsing whitespace and removing noise."""
        # Collapse multiple spaces / newlines into single spaces
        text = re.sub(r"\s+", " ", text)
        # Remove control characters except newlines
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        return text.strip()

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text on sentence-ending punctuation while keeping the delimiter.

        Handles abbreviations like 'Q3.' or 'Dr.' reasonably well by requiring
        the period to be followed by a space and an uppercase letter.
        """
        # Split on '. ', '? ', '! ' followed by uppercase letter or end
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        # Filter out empty strings
        return [s.strip() for s in parts if s.strip()]

    def _merge_sentences_into_chunks(self, sentences: List[str]) -> List[str]:
        """Group sentences into chunks of roughly `chunk_size` characters
        with `chunk_overlap` character overlap between consecutive chunks.

        This preserves sentence boundaries so no sentence is cut in half.
        """
        if not sentences:
            return []

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence would exceed the target size and we
            # already have content, finalise the current chunk.
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Build overlap: keep trailing sentences whose combined length
                # is under chunk_overlap.
                overlap_chunk: List[str] = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) > self.chunk_overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_len += len(s)

                current_chunk = overlap_chunk
                current_length = overlap_len

            current_chunk.append(sentence)
            current_length += sentence_len

        # Don't forget the final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks