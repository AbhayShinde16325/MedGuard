"""
Live market data client using Yahoo Finance.

Provides:
- TickerResolver   : Map company names → stock ticker symbols
- MarketDataClient : Fetch live prices, key fundamentals, and historical data
"""

from datetime import datetime
import logging
from typing import Optional

import yfinance as yf
from pathlib import Path
import csv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------

class TickerResolver:
    """
    Resolves company names to stock tickers using predefined datasets
    (NASDAQ-100, NSE-100, Europe-100).
    """

    def __init__(self):
        self.company_to_ticker: dict[str, str] = {}
        self._load_all()

    def _load_all(self):
        base_dir = Path(__file__).resolve().parents[3]
        data_dir = base_dir / "data" / "processed"

        for file in ["nasdaq_100.csv", "nse_100.csv", "europe_100.csv"]:
            path = data_dir / file
            if path.exists():
                self._load_csv(path)

        logger.info("TickerResolver loaded %d company mappings.", len(self.company_to_ticker))

    def _load_csv(self, path: Path):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["company"].strip().lower()
                ticker = row["ticker"].strip()
                self.company_to_ticker[name] = ticker

    def resolve(self, question: str) -> str | None:
        """Find the first company name mentioned in *question* and return its ticker."""
        q = question.lower()
        # Try longest names first to avoid partial matches
        for company in sorted(self.company_to_ticker, key=len, reverse=True):
            if company in q:
                return self.company_to_ticker[company]
        return None


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

class MarketDataClient:
    """
    Fetches live market data for equities via Yahoo Finance.
    """

    def get_stock_price(self, ticker: str) -> dict:
        """Get the latest stock price for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., AAPL, RELIANCE.NS)

        Returns:
            dict with keys: ticker, price, currency, change_pct, timestamp
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")

        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        latest = data.iloc[-1]

        # Dynamic currency from Yahoo Finance info (fallback to USD)
        info = self._get_info_safe(stock)
        currency = info.get("currency", "USD")

        # Calculate daily change %
        open_price = float(latest.get("Open", latest["Close"]))
        close_price = float(latest["Close"])
        change_pct = round(((close_price - open_price) / open_price) * 100, 2) if open_price else 0.0

        return {
            "ticker": ticker,
            "price": round(close_price, 2),
            "currency": currency,
            "change_pct": change_pct,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def get_fundamentals(self, ticker: str) -> dict:
        """Get key financial fundamentals for a given ticker.

        Returns a dict with analyst-friendly metrics. Missing values are None.
        """
        stock = yf.Ticker(ticker)
        info = self._get_info_safe(stock)

        return {
            "ticker": ticker,
            "name": info.get("shortName") or info.get("longName", ticker),
            "currency": info.get("currency", "USD"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "revenue": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "earnings_growth": info.get("earningsGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }

    def get_historical(self, ticker: str, period: str = "1mo") -> list[dict]:
        """Get historical closing prices.

        Args:
            ticker: Stock ticker symbol.
            period: yfinance period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 5y, max).

        Returns:
            List of dicts with date and close price.
        """
        data = yf.Ticker(ticker).history(period=period)
        if data.empty:
            return []

        return [
            {"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
            for d, c in zip(data.index, data["Close"])
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_info_safe(stock: yf.Ticker) -> dict:
        """Get ticker .info dict, returning {} on failure."""
        try:
            return stock.info or {}
        except Exception:
            return {}


# ---------------------------------------------------------------------------
# Formatting helpers (used by ResponseBuilder)
# ---------------------------------------------------------------------------

def format_fundamentals(data: dict) -> str:
    """Turn a fundamentals dict into a human-readable context string."""
    lines = [f"**{data.get('name', data['ticker'])}** ({data['ticker']})"]

    def _fmt(label: str, key: str, fmt: str = "{}"):
        val = data.get(key)
        if val is not None:
            lines.append(f"- {label}: {fmt.format(val)}")

    _fmt("Sector", "sector")
    _fmt("Industry", "industry")
    _fmt("Currency", "currency")
    _fmt("P/E (TTM)", "pe_ratio", "{:.2f}")
    _fmt("Forward P/E", "forward_pe", "{:.2f}")
    _fmt("Market Cap", "market_cap", "{:,.0f}")
    _fmt("Revenue", "revenue", "{:,.0f}")
    _fmt("Profit Margin", "profit_margin", "{:.2%}")
    _fmt("Earnings Growth", "earnings_growth", "{:.2%}")
    _fmt("Dividend Yield", "dividend_yield", "{:.2%}")
    _fmt("52-Week High", "52w_high", "{:.2f}")
    _fmt("52-Week Low", "52w_low", "{:.2f}")

    return "\n".join(lines)
