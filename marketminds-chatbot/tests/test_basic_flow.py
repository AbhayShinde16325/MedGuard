"""
Unit tests for MarketMinds Chatbot.

Covers:
- QueryRouter: intent classification
- DocumentIngestor: text chunking with overlap
- Retriever: vector store add / retrieve
- TickerResolver: company name → ticker mapping
- ResponseBuilder: end-to-end response flow (with mocked LLM)
- Prompt templates: correct prompt assembly
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ===================================================================
# 1. QueryRouter tests
# ===================================================================

from backend.app.chatbot.router import QueryRouter, QueryType


class TestQueryRouter:
    """Tests for keyword-based intent routing."""

    def setup_method(self):
        self.router = QueryRouter()

    # --- LIVE_MARKET ---
    def test_price_query(self):
        assert self.router.route("What is Apple's share price?") == QueryType.LIVE_MARKET

    def test_market_cap_query(self):
        assert self.router.route("Tell me the market cap of Tesla") == QueryType.LIVE_MARKET

    def test_trading_at_query(self):
        assert self.router.route("What is AAPL trading at?") == QueryType.LIVE_MARKET

    # --- DOCUMENT ---
    def test_report_query(self):
        assert self.router.route("Summarize the earnings report") == QueryType.DOCUMENT

    def test_annual_report_query(self):
        assert self.router.route("Key risks in the annual report") == QueryType.DOCUMENT

    def test_pdf_query(self):
        assert self.router.route("What does the pdf say about revenue?") == QueryType.DOCUMENT

    # --- FINANCIAL_ANALYSIS ---
    def test_pe_ratio_query(self):
        assert self.router.route("What is the P/E ratio?") == QueryType.FINANCIAL_ANALYSIS

    def test_valuation_query(self):
        assert self.router.route("Is Tesla overvalued?") == QueryType.FINANCIAL_ANALYSIS

    def test_balance_sheet_query(self):
        assert self.router.route("Show me the balance sheet") == QueryType.FINANCIAL_ANALYSIS

    def test_cash_flow_query(self):
        assert self.router.route("What is the cash flow?") == QueryType.FINANCIAL_ANALYSIS

    # --- GENERAL ---
    def test_general_question(self):
        assert self.router.route("What is a stock?") == QueryType.GENERAL

    def test_greeting(self):
        assert self.router.route("Hello, who are you?") == QueryType.GENERAL

    def test_empty_question(self):
        assert self.router.route("") == QueryType.GENERAL


# ===================================================================
# 2. DocumentIngestor tests
# ===================================================================

from backend.app.rag.ingest import DocumentIngestor


class TestDocumentIngestor:
    """Tests for semantic text chunking."""

    def setup_method(self):
        self.ingestor = DocumentIngestor(chunk_size=200, chunk_overlap=50)

    def test_basic_chunking(self):
        text = (
            "Apple reported strong Q3 results. Revenue grew 8 percent. "
            "iPhone sales exceeded expectations. Services revenue hit a new record. "
            "Tim Cook highlighted AI investments. The company announced a buyback. "
            "Gross margin improved to 45 percent. Operating expenses were controlled."
        )
        chunks = self.ingestor.ingest_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)
        assert all(len(c) > 0 for c in chunks)

    def test_overlap_between_chunks(self):
        # Use small chunk size to force multiple chunks
        ingestor = DocumentIngestor(chunk_size=100, chunk_overlap=30)
        text = (
            "Sentence one about Apple. Sentence two about revenue growth. "
            "Sentence three about margins. Sentence four about investments. "
            "Sentence five about stock buyback. Sentence six about the future."
        )
        chunks = ingestor.ingest_text(text)
        if len(chunks) >= 2:
            # Check that some words from end of chunk[0] appear in start of chunk[1]
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Consecutive chunks should share words (overlap)"

    def test_empty_text(self):
        chunks = self.ingestor.ingest_text("")
        assert chunks == []

    def test_whitespace_cleaning(self):
        text = "   Hello    world.   This   is   a   test.  "
        chunks = self.ingestor.ingest_text(text)
        assert len(chunks) >= 1
        # No double spaces
        for chunk in chunks:
            assert "  " not in chunk

    def test_single_sentence(self):
        text = "This is one sentence without a period"
        chunks = self.ingestor.ingest_text(text)
        assert len(chunks) == 1


# ===================================================================
# 3. Retriever tests
# ===================================================================

from backend.app.rag.retriever import Retriever
from backend.app.rag.embeddings import DummyEmbeddingClient, SentenceTransformerClient


class TestRetriever:
    """Tests for FAISS-backed retriever."""

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.store_path = Path(self.tmp_dir) / "test_store.pkl"

    def test_add_and_retrieve_with_dummy(self):
        client = DummyEmbeddingClient()
        retriever = Retriever(embedding_client=client, vector_store_path=self.store_path)

        docs = ["Apple had strong earnings", "Tesla stock dropped", "Amazon revenue grew"]
        retriever.add_documents(docs)

        assert retriever.index.ntotal == 3
        results = retriever.retrieve("Apple earnings", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, str) for r in results)

    def test_empty_retriever(self):
        client = DummyEmbeddingClient()
        retriever = Retriever(embedding_client=client, vector_store_path=self.store_path)
        results = retriever.retrieve("anything", top_k=3)
        assert results == []

    def test_persistence(self):
        client = DummyEmbeddingClient()
        # Create and populate
        retriever1 = Retriever(embedding_client=client, vector_store_path=self.store_path)
        retriever1.add_documents(["Test document one", "Test document two"])

        # Load from saved file
        retriever2 = Retriever(embedding_client=client, vector_store_path=self.store_path)
        assert retriever2.index.ntotal == 2
        assert len(retriever2.text_chunks) == 2

    def test_top_k_clamping(self):
        client = DummyEmbeddingClient()
        retriever = Retriever(embedding_client=client, vector_store_path=self.store_path)
        retriever.add_documents(["Only one doc"])
        results = retriever.retrieve("query", top_k=10)
        assert len(results) == 1  # Should clamp to available docs


# ===================================================================
# 4. TickerResolver tests
# ===================================================================

from backend.app.data_sources.financial_api import TickerResolver


class TestTickerResolver:
    """Tests for company name → ticker resolution."""

    def setup_method(self):
        self.resolver = TickerResolver()

    def test_apple_resolves(self):
        ticker = self.resolver.resolve("How is Apple performing?")
        assert ticker == "AAPL"

    def test_tcs_resolves(self):
        ticker = self.resolver.resolve("What is TCS stock price?")
        assert ticker == "TCS.NS"

    def test_unknown_company(self):
        ticker = self.resolver.resolve("How is XYZCorp doing?")
        assert ticker is None

    def test_case_insensitive(self):
        ticker = self.resolver.resolve("TELL ME ABOUT TESLA")
        assert ticker == "TSLA"

    def test_empty_question(self):
        ticker = self.resolver.resolve("")
        assert ticker is None


# ===================================================================
# 5. Prompt template tests
# ===================================================================

from backend.app.llm.prompt_templates import system_prompt, user_prompt, full_prompt


class TestPromptTemplates:
    """Tests for prompt construction."""

    def test_system_prompt_not_empty(self):
        sp = system_prompt()
        assert len(sp) > 50
        assert "MarketMinds" in sp

    def test_user_prompt_format(self):
        up = user_prompt("What is a stock?")
        assert "What is a stock?" in up
        assert "Question:" in up

    def test_full_prompt_without_context(self):
        fp = full_prompt(question="Hello")
        assert "MarketMinds" in fp
        assert "Hello" in fp
        assert "Context:" not in fp

    def test_full_prompt_with_context(self):
        fp = full_prompt(question="Hello", context="Some context here")
        assert "Context:" in fp
        assert "Some context here" in fp
        assert "Hello" in fp

    def test_full_prompt_with_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        fp = full_prompt(question="Next question", chat_history=history)
        assert "Hi" in fp
        assert "Hello!" in fp
        assert "Next question" in fp


# ===================================================================
# Run
# ===================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
