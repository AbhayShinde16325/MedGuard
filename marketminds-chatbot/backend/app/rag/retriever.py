"""
Retriever module with FAISS persistence.

Uses the embedding dimension from the EmbeddingClient to initialise the
FAISS index correctly.  Old 1-dim vector stores are incompatible and will
be rebuilt automatically.
"""

from typing import List
from pathlib import Path
import pickle
import logging

import faiss
import numpy as np

from backend.app.rag.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


class Retriever:
    """
    FAISS-backed retriever with persistence.
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store_path: Path,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store_path = vector_store_path
        self.dim = embedding_client.dim

        self.text_chunks: List[str] = []
        self.index = None

        if self.vector_store_path.exists():
            try:
                self._load()
                # If the persisted index has a different dimension
                # (e.g. old 1-dim dummy), rebuild from scratch.
                if self.index.d != self.dim:
                    logger.warning(
                        "Vector store dimension mismatch (stored=%d, expected=%d). "
                        "Rebuilding index.",
                        self.index.d,
                        self.dim,
                    )
                    self._initialize_index()
                    self.text_chunks = []
            except Exception as exc:
                logger.warning("Failed to load vector store: %s. Rebuilding.", exc)
                self._initialize_index()
        else:
            self._initialize_index()

    # ------------------------------------------------------------------

    def _initialize_index(self):
        """Create a fresh FAISS L2 index matching the embedding dimension."""
        self.index = faiss.IndexFlatL2(self.dim)
        logger.info("Initialised new FAISS index (dim=%d).", self.dim)

    # ------------------------------------------------------------------

    def add_documents(self, chunks: List[str]) -> None:
        """Embed *chunks* and add the resulting vectors to the index."""
        if not chunks:
            return

        vectors = self.embedding_client.embed(chunks)
        if not vectors:
            return

        faiss_vectors = np.array(vectors, dtype="float32")
        self.index.add(faiss_vectors)

        self.text_chunks.extend(chunks)
        self._save()

    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Return the *top_k* most similar chunks for the given query."""
        if self.index.ntotal == 0:
            return []

        query_vector = self.embedding_client.embed([query])[0]
        query_np = np.array([query_vector], dtype="float32")

        # Clamp top_k to the number of stored vectors
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_np, k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.text_chunks):
                results.append(self.text_chunks[idx])

        return results

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save(self):
        """Persist the FAISS index + text chunks to disk."""
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vector_store_path, "wb") as f:
            pickle.dump(
                {
                    "index": faiss.serialize_index(self.index),
                    "chunks": self.text_chunks,
                    "dim": self.dim,
                },
                f,
            )

    def _load(self):
        """Load a previously saved FAISS index + text chunks from disk."""
        with open(self.vector_store_path, "rb") as f:
            data = pickle.load(f)

            raw_index = data["index"]

            # faiss.serialize_index returns a numpy array, not bytes
            if isinstance(raw_index, np.ndarray):
                self.index = faiss.deserialize_index(raw_index)
            elif isinstance(raw_index, bytes):
                self.index = faiss.deserialize_index(np.frombuffer(raw_index, dtype="uint8"))
            else:
                # Assume it's a raw FAISS index object (legacy format)
                self.index = raw_index

            self.text_chunks = data.get("chunks", [])
