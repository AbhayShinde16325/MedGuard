# MarketMinds — Development Session Context

This document captures the entire context, progress, and architectural decisions made during the development session for **MarketMinds**, an AI-powered financial intelligence assistant.

## 🚀 Project Overview
MarketMinds is a local-first, RAG-powered chatbot designed to analyze market data, fetch live stock prices, read financial news, and ingest PDF reports (like 10-Ks) to answer advanced financial queries. 

The application uses a **FastAPI backend** and a **vanilla HTML/JS/CSS frontend**, bundled into a lightweight Single Page Application (SPA).

---

## 🛠️ Phases Completed

### Phase 1: Frontend Revamp
*   **Design Upgrade:** Implemented a modern, dark-themed UI (`var(--bg-primary)` #030712) using Vanilla CSS.
*   **Components:** Created a clean header with a tagline, a welcome screen with suggested prompts, and a file drag-and-drop zone.
*   **Chat Interface:** Implemented a markdown-aware chat window (using `marked.js`) with distinct user/bot styling and typing indicators.
*   **Responsiveness:** Ensured the UI works seamlessly on mobile devices.

### Phase 2: RAG Pipeline & PDF Ingestion
*   **Document Ingestion (`ingest.py`):** Built a pipeline using `PyPDF2` to read PDFs, split text into smaller chunks (`langchain.text_splitter.RecursiveCharacterTextSplitter`), and clean whitespace.
*   **Vector Database (`retriever.py`):** Integrated **FAISS** for fast, local similarity search. Vectors are persisted locally using `pickle`.
*   **Embeddings (`embeddings.py`):** Added support for both local (HuggingFace `all-MiniLM-L6-v2`) and cloud (Gemini) embedding models.
*   **Endpoints:** Added the `POST /upload` endpoint to ingest user documents on the fly with a 50MB file size limit.

### Phase 3: Live Market Data & Routing Refinement
*   **Yahoo Finance Integration (`financial_api.py`):** Added the `yfinance` library to fetch real-time stock prices, market cap, P/E ratios, and historical sparkline data.
*   **Ticker Resolution (`ticker_utils.py`):** Built a local mapping system to translate common company names (e.g., "Apple", "TCS") into stock symbols (`AAPL`, `TCS.NS`) without relying on an LLM.
*   **Smart Routing (`router.py`):** Separated the query router into a fast `KeywordRouter` and a fallback `LLMRouter` to correctly route user questions to the right internal tool (Live Market vs. RAG vs. General Chat).

### Phase 4: Conversation Memory & Unit Testing
*   **Session Management:** Implemented per-session memory on the backend (`ResponseBuilder._append_to_history`).
*   **Frontend Tracking:** The frontend now stores unique `session_id`s in `localStorage` and includes them in every API call.
*   **Context Injection:** Updated prompt templates to inject `=== Previous Conversation ===` so the LLM understands follow-up questions like *"What about Tesla?"*
*   **End-to-End Testing:** Wrote 32 comprehensive tests (`pytest`) covering the Router, Ingestor, Retriever, Ticker Resolver, and Prompt Templates. 

### Phase 5: Rich UI, Streaming & News
*   **Server-Sent Events (SSE):** Built `POST /ask/stream` to stream LLM responses token-by-token directly to the UI, complete with a blinking cursor `▊`.
*   **Interactive Ticker Cards:** When the bot mentions a stock ticker (e.g., `**AAPL**`), the frontend intercepts it and injects a rich UI card showing the live price, percentage change, and a 5-day sparkline chart (using `Chart.js`).
*   **Live News Subsystem:** Added a "📰 News" button to ticker cards that fetches the latest 8 Yahoo Finance headlines for that company.
*   **System Status Bar:** Added a living status bar below the header to show the user the active LLM provider, vector database count, and connection health.

### Phase 6: Packaging & Safeguarding
*   **PyInstaller Integration:** Wrote a `build_exe.py` script to bundle the Python backend, UI assets, and dependencies into a standalone `.exe` Windows application.
*   **Credential Safeguarding:** Reconfigured the app to store sensitive data (`.env` with Gemini API keys) and persistent vectors in the user's `%LOCALAPPDATA%\MarketMinds` folder, rather than hardcoding or compiling them into the executable.
*   **Launcher Wrapper:** Created `launcher.py` to auto-open the browser to `localhost:8000` when the user double-clicks the application.

---

## 🏗️ Architecture & Technologies

*   **Backend Framework:** FastAPI, Uvicorn (async, high performance).
*   **LLM Providers:** Dual-support for `Ollama` (local, secure inference) and `Gemini 2.5 Flash` (cloud, fast).
*   **AI/RAG:** FAISS (Vector Store), SentenceTransformers (Embeddings), LangChain (Text Splitting).
*   **Finance Data:** `yfinance` package.
*   **Frontend:** Vanilla JS (`app.js`), Vanilla CSS (`style.css`), Chart.js (sparklines), Marked.js (Markdown parsing).
*   **Testing:** `pytest` (fully passing 32/32 tests).
*   **Deployment:** PyInstaller (Standalone Windows `.exe`).

---

## 🔐 Environment Variables (.env)

The app reads configuration from `%LOCALAPPDATA%\MarketMinds\.env` when packaged, or the project root during development.

```env
# Provider choice (ollama or gemini)
LLM_PROVIDER=gemini

# API key for Gemini cloud model
GEMINI_API_KEY=your_key_here

# Default models
GEMINI_MODEL=gemini-2.5-flash
DEFAULT_MODEL=mistral

# Opt-in for smarter, AI-driven query routing
USE_LLM_ROUTING=false
```

---

## 🏃 How to Run

### Development Mode
```powershell
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Build Executable
```powershell
python build_exe.py
```
*The packaged application will be generated in `dist/launcher`.*
