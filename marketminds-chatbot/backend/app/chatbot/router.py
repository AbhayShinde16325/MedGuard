"""
Intent Router for MarketMinds Chatbot.

Classifies a user question into one of the known QueryType categories
so the ResponseBuilder can dispatch to the correct data source.

Two routing strategies:
  1. KeywordRouter  — fast, deterministic, no LLM call (always used as fallback)
  2. LLMRouter      — uses the LLM for smarter classification (optional)

The default QueryRouter tries the LLM first and falls back to keywords.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class QueryType(Enum):
    LIVE_MARKET = "live_market"
    DOCUMENT = "document"
    FINANCIAL_ANALYSIS = "financial_analysis"
    GENERAL = "general"


# ───────────────────────────────────────────────────────────────────
# Keyword-based router (fast, deterministic)
# ───────────────────────────────────────────────────────────────────

class KeywordRouter:
    """Classify queries using simple keyword matching — always works, no LLM needed."""

    # fmt: off
    _LIVE_KEYWORDS = [
        "price", "share price", "stock price", "market cap",
        "trading at", "current value", "how much is",
        "what is the price", "live", "quote",
    ]
    _DOC_KEYWORDS = [
        "report", "document", "pdf", "earnings call",
        "annual report", "quarterly report", "filing",
        "said", "mentioned", "according to",
    ]
    _FIN_KEYWORDS = [
        "overvalued", "undervalued", "valuation", "fundamentals",
        "ratio", "pe", "p/e", "eps", "roe", "roa",
        "balance sheet", "income statement", "cash flow",
        "financials", "revenue growth", "net income",
        "dividend", "market analysis", "sector analysis",
        "compare", "comparison",
    ]
    # fmt: on

    def route(self, question: str) -> QueryType:
        q = question.lower()

        if any(kw in q for kw in self._LIVE_KEYWORDS):
            return QueryType.LIVE_MARKET

        if any(kw in q for kw in self._DOC_KEYWORDS):
            return QueryType.DOCUMENT

        if any(kw in q for kw in self._FIN_KEYWORDS):
            return QueryType.FINANCIAL_ANALYSIS

        return QueryType.GENERAL


# ───────────────────────────────────────────────────────────────────
# LLM-based router (smarter, requires LLM availability)
# ───────────────────────────────────────────────────────────────────

# Compact prompt for the LLM router
_ROUTER_PROMPT = """Classify the following user question into EXACTLY ONE category.
Reply with ONLY the category name, nothing else.

Categories:
- LIVE_MARKET: Questions about current stock prices, market capitalisation, or real-time trading data.
- DOCUMENT: Questions that reference uploaded documents, PDF reports, earnings calls, or ask to summarize/analyse specific reports.
- FINANCIAL_ANALYSIS: Questions about financial ratios (P/E, EPS, ROE), valuations, balance sheets, income statements, comparisons between companies, or investment analysis.
- GENERAL: Everything else — general finance knowledge, definitions, or greetings.

Question: {question}
Category:"""


class LLMRouter:
    """Classify queries using a lightweight LLM call for more accurate routing."""

    _VALID_CATEGORIES = {qt.value.upper() for qt in QueryType}

    def __init__(self, llm_client) -> None:
        self.llm_client = llm_client

    def route(self, question: str) -> Optional[QueryType]:
        """Attempt LLM-based routing. Returns None if it fails."""
        try:
            prompt = _ROUTER_PROMPT.format(question=question)
            raw = self.llm_client.generate(prompt=prompt)

            # Parse the response — extract the category name
            cleaned = raw.strip().upper().replace(" ", "_")

            # Try to find a valid QueryType in the response
            for qt in QueryType:
                if qt.value.upper() in cleaned:
                    logger.debug("LLM router classified as: %s", qt.value)
                    return qt

            logger.warning("LLM router returned unparseable: '%s'", raw.strip())
            return None

        except Exception as exc:
            logger.warning("LLM router failed: %s — falling back to keywords", exc)
            return None


# ───────────────────────────────────────────────────────────────────
# Combined router (default)
# ───────────────────────────────────────────────────────────────────

class QueryRouter:
    """Smart router: tries LLM classification first, falls back to keywords.

    If no LLM client is provided, keyword routing is used exclusively.
    """

    def __init__(self, llm_client=None, use_llm_routing: bool = False) -> None:
        self._keyword_router = KeywordRouter()
        self._llm_router = None

        if llm_client and use_llm_routing:
            self._llm_router = LLMRouter(llm_client)
            logger.info("QueryRouter: LLM-based routing enabled.")
        else:
            logger.info("QueryRouter: Using keyword-based routing.")

    def route(self, question: str) -> QueryType:
        """Classify the query — LLM first (if enabled), keyword fallback."""
        # Try LLM routing
        if self._llm_router:
            result = self._llm_router.route(question)
            if result is not None:
                return result

        # Keyword fallback (always works)
        return self._keyword_router.route(question)
