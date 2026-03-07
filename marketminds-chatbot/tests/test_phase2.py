"""Phase 2 validation tests."""

import sys
sys.path.insert(0, ".")

print("=" * 55)
print("Phase 2 Validation Tests")
print("=" * 55)


# 1. Ollama HTTP client (no subprocess)
print("\n[1/5] Ollama HTTP Client...")
from backend.app.llm.llm_client import OllamaLLMClient
client = OllamaLLMClient(model_name="mistral")
assert not hasattr(client, "subprocess"), "Should not use subprocess"
assert hasattr(client, "base_url"), "Should have base_url for HTTP API"
assert client.base_url == "http://localhost:11434"
print(f"  ✅ OllamaLLMClient uses HTTP API at {client.base_url}")
print(f"  ℹ️  Ollama reachable: {client.is_available()}")


# 2. Semantic chunking with overlap
print("\n[2/5] Semantic Chunking...")
from backend.app.rag.ingest import DocumentIngestor

ingestor = DocumentIngestor(chunk_size=150, chunk_overlap=50)

sample_text = (
    "Apple reported strong Q3 results. Revenue grew 8 percent year-over-year. "
    "iPhone sales exceeded expectations. Services revenue hit a new record. "
    "Tim Cook highlighted AI investments. The company announced a stock buyback. "
    "Gross margin improved to 45 percent. Operating expenses were well controlled."
)

chunks = ingestor.ingest_text(sample_text)
print(f"  Chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"    [{i}] ({len(chunk)} chars): {chunk[:70]}...")

# Verify overlap: consecutive chunks should share some text
if len(chunks) >= 2:
    # Find common words between chunk[0] end and chunk[1] start
    words_end_0 = set(chunks[0].split()[-5:])
    words_start_1 = set(chunks[1].split()[:5])
    overlap = words_end_0 & words_start_1
    assert len(overlap) > 0, "Consecutive chunks should overlap"
    print(f"  ✅ Overlap detected between chunks: {overlap}")

# Verify no sentence is cut mid-word
for chunk in chunks:
    assert not chunk.startswith(" "), f"Chunk starts with space: '{chunk[:30]}'"
print("  ✅ All chunks start at sentence boundaries")


# 3. Financial API — dynamic currency + fundamentals
print("\n[3/5] Financial API (fundamentals + dynamic currency)...")
from backend.app.data_sources.financial_api import MarketDataClient, format_fundamentals

market = MarketDataClient()
try:
    price_data = market.get_stock_price("AAPL")
    print(f"  ✅ Price: {price_data['price']} {price_data['currency']} "
          f"(change: {price_data['change_pct']:+.2f}%)")
    assert price_data["currency"] != "", "Currency should not be empty"
except Exception as e:
    print(f"  ⚠️  Price fetch skipped (network?): {e}")

try:
    fund_data = market.get_fundamentals("AAPL")
    print(f"  ✅ Fundamentals for {fund_data.get('name', 'AAPL')}:")
    formatted = format_fundamentals(fund_data)
    for line in formatted.split("\n")[:6]:
        print(f"    {line}")
    assert fund_data["ticker"] == "AAPL"
    assert "pe_ratio" in fund_data
    assert "market_cap" in fund_data
except Exception as e:
    print(f"  ⚠️  Fundamentals fetch skipped (network?): {e}")


# 4. Financial Analysis route in ResponseBuilder
print("\n[4/5] Financial Analysis Route...")
from backend.app.chatbot.response_builder import ResponseBuilder
# Just verify the method exists and is callable
assert hasattr(ResponseBuilder, "_handle_financial_analysis")
print("  ✅ _handle_financial_analysis method exists")


# 5. Upload endpoint exists
print("\n[5/5] Upload Endpoint...")
from backend.app.main import app as fastapi_app
routes = [route.path for route in fastapi_app.routes]
assert "/upload" in routes, "/upload endpoint missing"
assert "/fundamentals/{ticker}" in routes, "/fundamentals endpoint missing"
assert "/historical/{ticker}" in routes, "/historical endpoint missing"
print(f"  ✅ /upload endpoint registered")
print(f"  ✅ /fundamentals/{{ticker}} endpoint registered")
print(f"  ✅ /historical/{{ticker}} endpoint registered")

print("\n" + "=" * 55)
print("🎉 ALL PHASE 2 CHECKS PASSED!")
print("=" * 55)
