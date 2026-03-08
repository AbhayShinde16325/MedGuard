"""Embeddings module for MarketMinds Chatbot.

This module defines how text chunks are converted into
vector embeddings for semantic search.

Provides:
- EmbeddingClient          : Abstract base class
- GeminiEmbeddingClient    : Real embeddings via Google Gemini (default)
- DummyEmbeddingClient     : Fallback for testing without API calls
"""

from typing import List
import logging
from pathlib import Path
import sys

from backend.app.config import BASE_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding dimension used across the RAG pipeline.
# Must match the output dimension of the chosen embedding model.
# 'text-embedding-004' → 768 dimensions
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 768


class EmbeddingClient:
    """Base embedding client interface."""

    # Subclasses must set this so the retriever can initialize FAISS correctly.
    dim: int = EMBEDDING_DIM

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of texts into their corresponding embeddings.

        Args:
            texts: List of text chunks to embed.

        Returns:
            List of embeddings, each represented as a list of floats.
        """
        raise NotImplementedError(
            "EmbeddingClient.embed() must be implemented."
        )


class GeminiEmbeddingClient(EmbeddingClient):
    """Production embedding client using Google Gemini.

    Uses the 'models/text-embedding-004' model for embeddings.
    """

    def __init__(self) -> None:
        from backend.app.config import GEMINI_API_KEY
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Set it in your .env file or as an environment variable."
            )

        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Gemini embeddings require the 'google-generativeai' package. "
                "Install it in your development environment to use Gemini."
            ) from exc

        genai.configure(api_key=GEMINI_API_KEY)
        self.model = "models/embedding-001"
        self.dim = EMBEDDING_DIM
        logger.info("Gemini embedding model configured (dim=%d).", self.dim)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate real semantic embeddings for the given texts using Gemini."""
        if not texts:
            return []
        
        import google.generativeai as genai
        embeddings = []
        try:
            for text in texts:
                result = genai.embed_content(model=self.model, content=text)
                embeddings.append(result['embedding'])
        except Exception as e:
            logger.warning("Gemini embedding failed: %s. Falling back to dummy embeddings.", e)
            # Fallback to dummy
            for text in texts:
                base_val = float(len(text) % 100) / 100.0
                vector = [base_val] * EMBEDDING_DIM
                embeddings.append(vector)
        return embeddings


class DummyEmbeddingClient(EmbeddingClient):
    """Dummy embedding client for unit-testing the pipeline.

    Returns deterministic, 384-dimensional vectors based on text length.
    NOT suitable for any real retrieval.
    """

    dim: int = EMBEDDING_DIM

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate fixed-dimension dummy embeddings."""
        embeddings = []
        for text in texts:
            # Deterministic but meaningless vector
            base_val = float(len(text) % 100) / 100.0
            vector = [base_val] * EMBEDDING_DIM
            embeddings.append(vector)
        return embeddings