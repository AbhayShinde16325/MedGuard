"""
Response Builder for MarketMinds

This module orchestrates the full response flow:
- route the query
- gather context (API / documents / fundamentals)
- construct the prompt (with conversation history)
- call the LLM
- return the final answer

This is the ONLY place where LLMs are invoked.
"""

import logging
from collections import defaultdict

from backend.app.config import config
from backend.app.chatbot.router import QueryRouter, QueryType
from backend.app.llm.llm_client import LLMClient
from backend.app.llm.prompt_templates import full_prompt
from backend.app.data_sources.financial_api import (
    MarketDataClient,
    TickerResolver,
    format_fundamentals,
)
from backend.app.rag.ingest import DocumentIngestor
from backend.app.rag.retriever import Retriever
from backend.app.rag.embeddings import GeminiEmbeddingClient

logger = logging.getLogger(__name__)

# Maximum number of messages to keep in conversation history per session
MAX_HISTORY_LENGTH = 20


class ResponseBuilder:
    """
    Builds grounded responses to user questions.
    Maintains per-session conversation memory.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        # Core components
        self.router = QueryRouter(
            llm_client=llm_client,
            use_llm_routing=config.USE_LLM_ROUTING,
        )
        self.llm_client = llm_client

        # Live market data
        self.market_client = MarketDataClient()
        self.ticker_resolver = TickerResolver()

        # RAG components — real semantic embeddings
        self.embedding_client = GeminiEmbeddingClient()
        vector_store_path = config.VECTOR_STORE_DIR / "faiss_store.pkl"

        self.retriever = Retriever(
            embedding_client=self.embedding_client,
            vector_store_path=vector_store_path,
        )

        self.ingestor = DocumentIngestor()

        # Conversation memory: session_id → list of messages
        self._conversations: dict[str, list[dict]] = defaultdict(list)

        # Ingest PDFs on first run (when no vector store exists yet)
        raw_data_dir = config.RAW_DATA_DIR
        if self.retriever.index.ntotal == 0:
            pdf_files = list(raw_data_dir.glob("*.pdf"))
            if pdf_files:
                logger.info(
                    "No existing vector store found. Ingesting %d PDF(s)…",
                    len(pdf_files),
                )
                for pdf_file in pdf_files:
                    chunks = self.ingestor.ingest_pdf(pdf_file)
                    self.retriever.add_documents(chunks)
                logger.info(
                    "PDF ingestion complete. %d vectors stored.",
                    self.retriever.index.ntotal,
                )
            else:
                logger.warning(
                    "No PDFs found in %s. The RAG pipeline has no documents to search.",
                    raw_data_dir,
                )

    # --------------------------------------------------
    # Conversation memory
    # --------------------------------------------------

    def get_history(self, session_id: str) -> list[dict]:
        """Return the conversation history for a session."""
        return list(self._conversations.get(session_id, []))

    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._conversations.pop(session_id, None)

    def _append_to_history(
        self, session_id: str, role: str, content: str
    ) -> None:
        """Add a message to the conversation, trimming if necessary."""
        history = self._conversations[session_id]
        history.append({"role": role, "content": content})

        # Trim to keep only the last MAX_HISTORY_LENGTH messages
        if len(history) > MAX_HISTORY_LENGTH:
            self._conversations[session_id] = history[-MAX_HISTORY_LENGTH:]

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def build_response(
        self, question: str, session_id: str = "default"
    ) -> str:
        """Build a response to the user's question.

        Args:
            question:   The user's question.
            session_id: Session identifier for conversation memory.

        Returns:
            The response string.
        """
        query_type = self.router.route(question)
        logger.info("Query routed as: %s (session=%s)", query_type.value, session_id)

        # Get conversation history for this session
        history = self.get_history(session_id)

        # Record the user's question
        self._append_to_history(session_id, "user", question)

        # Build the response
        answer = self._dispatch(question, query_type, history)

        # Record the assistant's answer
        self._append_to_history(session_id, "assistant", answer)

        return answer

    # --------------------------------------------------
    # Dynamic document ingestion (used by /upload endpoint)
    # --------------------------------------------------

    def ingest_document(self, pdf_path) -> int:
        """Ingest a new PDF into the RAG pipeline at runtime.

        Returns:
            Number of chunks added.
        """
        chunks = self.ingestor.ingest_pdf(pdf_path)
        self.retriever.add_documents(chunks)
        logger.info(
            "Dynamically ingested %s → %d chunks. Total vectors: %d",
            pdf_path.name,
            len(chunks),
            self.retriever.index.ntotal,
        )
        return len(chunks)

    # --------------------------------------------------
    # Internal dispatch
    # --------------------------------------------------

    def _dispatch(
        self,
        question: str,
        query_type: QueryType,
        history: list[dict],
    ) -> str:
        """Route to the correct handler based on query type."""

        # 1. Live market queries
        if query_type == QueryType.LIVE_MARKET:
            market_answer = self._handle_live_market_query(question)
            if market_answer:
                return market_answer
            # fallback to LLM if ticker not resolved

        # 2. Document / RAG queries
        if query_type == QueryType.DOCUMENT:
            context = self._get_rag_context(question)
            return self.llm_client.generate(
                prompt=full_prompt(
                    question=question, context=context, chat_history=history
                )
            )

        # 3. Financial analysis — combine fundamentals + RAG context
        if query_type == QueryType.FINANCIAL_ANALYSIS:
            context = self._handle_financial_analysis(question)
            return self.llm_client.generate(
                prompt=full_prompt(
                    question=question, context=context, chat_history=history
                )
            )

        # 4. General knowledge (LLM fallback)
        return self.llm_client.generate(
            prompt=full_prompt(question=question, chat_history=history)
        )

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------

    def _handle_live_market_query(self, question: str) -> str | None:
        ticker = self.ticker_resolver.resolve(question)

        if not ticker:
            return None

        try:
            data = self.market_client.get_stock_price(ticker)
        except Exception as exc:
            logger.error("Failed to fetch market data for %s: %s", ticker, exc)
            return "I couldn't fetch live market data at the moment."

        change_str = ""
        if data.get("change_pct", 0) != 0:
            direction = "📈" if data["change_pct"] > 0 else "📉"
            change_str = f" {direction} {data['change_pct']:+.2f}% today"

        return (
            f"**{data['ticker']}** is trading at **{data['price']} {data['currency']}**"
            f"{change_str} (as of {data['timestamp']})."
        )

    def _handle_financial_analysis(self, question: str) -> str:
        """Build rich context for financial analysis queries.

        Combines:
        - Live fundamentals data (P/E, market cap, etc.) if a ticker is found
        - RAG document context for additional report-based info
        """
        context_parts: list[str] = []

        # Try to get fundamentals for any mentioned company
        ticker = self.ticker_resolver.resolve(question)
        if ticker:
            try:
                fundamentals = self.market_client.get_fundamentals(ticker)
                context_parts.append(
                    "=== Live Financial Data ===\n" + format_fundamentals(fundamentals)
                )
            except Exception as exc:
                logger.warning("Could not fetch fundamentals for %s: %s", ticker, exc)

        # Also pull relevant document context
        rag_context = self._get_rag_context(question)
        if rag_context != "No relevant documents found.":
            context_parts.append("=== Document Context ===\n" + rag_context)

        if not context_parts:
            return "No relevant financial data or documents found."

        return "\n\n".join(context_parts)

    def _get_rag_context(self, question: str) -> str:
        chunks = self.retriever.retrieve(question, top_k=3)
        if not chunks:
            return "No relevant documents found."
        return "\n\n".join(f"- {chunk}" for chunk in chunks)
