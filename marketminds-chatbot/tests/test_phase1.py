"""Quick smoke test for Phase 1 changes."""

import sys
sys.path.insert(0, ".")

print("=" * 50)
print("Phase 1 Validation Tests")
print("=" * 50)

# 1. Config
print("\n[1/5] Config...")
from backend.app.config import config, LLM_PROVIDER
print(f"  ✅ LLM_PROVIDER = {LLM_PROVIDER}")
print(f"  ✅ DEFAULT_MODEL = {config.DEFAULT_MODEL_NAME}")
print(f"  ✅ DEBUG = {config.DEBUG}")

# 2. Router (no more import-time crash)
print("\n[2/5] Router...")
from backend.app.chatbot.router import QueryRouter, QueryType
router = QueryRouter()
assert router.route("What is Apple share price?") == QueryType.LIVE_MARKET
assert router.route("Summarize the annual report") == QueryType.DOCUMENT
assert router.route("What is the P/E ratio?") == QueryType.FINANCIAL_ANALYSIS
assert router.route("What is a stock?") == QueryType.GENERAL
print("  ✅ All 4 route types work correctly")

# 3. Embeddings
print("\n[3/5] Embeddings (loading model, may take a moment)...")
from backend.app.rag.embeddings import GeminiEmbeddingClient, EMBEDDING_DIM
client = GeminiEmbeddingClient()
vecs = client.embed(["hello world", "stock market crash"])
assert len(vecs) == 2
assert len(vecs[0]) == EMBEDDING_DIM
print(f"  ✅ Real embeddings work: dim={len(vecs[0])}")

# 4. LLM Client Factory
print("\n[4/5] LLM Client Factory...")
from backend.app.llm.llm_client import get_llm_client, LLMClient
llm = get_llm_client()
assert isinstance(llm, LLMClient)
print(f"  ✅ get_llm_client() → {type(llm).__name__} (model={llm.model_name})")

# 5. Retriever (FAISS with correct dim)
print("\n[5/5] Retriever...")
from backend.app.rag.retriever import Retriever
from pathlib import Path
import tempfile, os

tmp_path = Path(tempfile.mkdtemp()) / "test_store.pkl"
retriever = Retriever(embedding_client=client, vector_store_path=tmp_path)
assert retriever.index.d == EMBEDDING_DIM

retriever.add_documents(["Apple had strong Q3 earnings", "Tesla stock dropped 5%"])
results = retriever.retrieve("How did Apple perform?", top_k=1)
assert len(results) == 1
assert "Apple" in results[0]
print(f"  ✅ FAISS retriever works: dim={retriever.index.d}, retrieved='{results[0][:50]}...'")

# Cleanup
os.remove(tmp_path)

print("\n" + "=" * 50)
print("🎉 ALL PHASE 1 CHECKS PASSED!")
print("=" * 50)
