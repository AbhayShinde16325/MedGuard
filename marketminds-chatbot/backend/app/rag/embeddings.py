"""Embeddings module for MarketMinds Chatbot.

This module defines how text chunks are converted into
vector embeddings for semantic search.

Provides:
- EmbeddingClient          : Abstract base class
- SentenceTransformerClient: Real embeddings via sentence-transformers (default)
- DummyEmbeddingClient     : Fallback for testing without model downloads
"""

from typing import List
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding dimension used across the RAG pipeline.
# Must match the output dimension of the chosen SentenceTransformer model.
# 'all-MiniLM-L6-v2' → 384 dimensions
# ---------------------------------------------------------------------------
EMBEDDING_DIM: int = 384


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


class SentenceTransformerClient(EmbeddingClient):
    """Production embedding client using sentence-transformers.

    Downloads the model on first use (~80 MB for all-MiniLM-L6-v2).
    All subsequent calls use the cached model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s …", model_name)
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding model loaded (dim=%d).", self.dim)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate real semantic embeddings for the given texts."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


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