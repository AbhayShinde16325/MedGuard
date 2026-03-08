"""
MarketMinds FastAPI Backend  (v0.4.0)

Endpoints:
    GET  /                          – Serve frontend
    GET  /health                    – Health check (enhanced)
    POST /ask                       – Chat endpoint (with session memory)
    POST /ask/stream                – Streaming chat via SSE
    POST /upload                    – Upload PDF for RAG ingestion
    GET  /fundamentals/{ticker}     – Financial fundamentals
    GET  /historical/{ticker}       – Historical prices
    GET  /history/{session_id}      – Get conversation history
    DELETE /history/{session_id}    – Clear conversation history
    GET  /news/{ticker}             – Financial news headlines
    GET  /ticker/{symbol}           – Quick ticker card data
"""

import json
import logging
import shutil
import uuid
from pathlib import Path

import yfinance as yf
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.config import config, LLM_PROVIDER
from backend.app.chatbot.response_builder import ResponseBuilder
from backend.app.llm.llm_client import get_llm_client
from backend.app.llm.prompt_templates import full_prompt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MarketMinds API",
    description="Local RAG-powered financial chatbot backend",
    version="0.4.0",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal server error occurred. Please try again.",
            "error_type": type(exc).__name__,
        },
    )


# ---------- Frontend setup ----------
FRONTEND_DIR = config.BASE_DIR / "frontend"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


# ---------- Core system ----------
llm_client = get_llm_client()
response_builder = ResponseBuilder(llm_client)


# ===================================================================
# API models
# ===================================================================

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    session_id: str


class UploadResponse(BaseModel):
    filename: str
    chunks: int
    message: str


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


# ===================================================================
# Health (enhanced)
# ===================================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "0.4.0",
        "llm_provider": LLM_PROVIDER,
        "model": config.DEFAULT_MODEL_NAME,
        "vectors_stored": response_builder.retriever.index.ntotal,
        "documents_ingested": len(set(response_builder.retriever.text_chunks)),
        "tickers_loaded": len(response_builder.ticker_resolver.company_to_ticker),
    }


# ===================================================================
# Chat endpoint (with session memory)
# ===================================================================

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """Answer a financial question with multi-turn session support."""
    session_id = request.session_id or str(uuid.uuid4())

    try:
        answer = response_builder.build_response(
            question=request.question,
            session_id=session_id,
        )
    except Exception as exc:
        logger.error("Error building response: %s", exc)
        answer = (
            "⚠️ Something went wrong while processing your question. "
            "Please try rephrasing or ask a different question."
        )

    return QueryResponse(answer=answer, session_id=session_id)


# ===================================================================
# Streaming chat via SSE
# ===================================================================

@app.post("/ask/stream")
def ask_stream(request: QueryRequest):
    """Stream an LLM response token-by-token using Server-Sent Events.

    The frontend should listen for 'data:' lines and render tokens
    progressively. A final 'data: [DONE]' event signals completion.
    """
    session_id = request.session_id or str(uuid.uuid4())

    # Determine context + prompt (same routing as build_response)
    from backend.app.chatbot.router import QueryType
    query_type = response_builder.router.route(request.question)
    history = response_builder.get_history(session_id)

    # Record user message
    response_builder._append_to_history(session_id, "user", request.question)

    # For live market queries, send the whole response immediately (no streaming)
    if query_type == QueryType.LIVE_MARKET:
        market_answer = response_builder._handle_live_market_query(request.question)
        if market_answer:
            response_builder._append_to_history(session_id, "assistant", market_answer)

            def market_sse():
                yield f"data: {json.dumps({'token': market_answer, 'session_id': session_id})}\n\n"
                yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

            return StreamingResponse(market_sse(), media_type="text/event-stream")

    # Build the prompt
    context = None
    if query_type == QueryType.DOCUMENT:
        context = response_builder._get_rag_context(request.question)
    elif query_type == QueryType.FINANCIAL_ANALYSIS:
        context = response_builder._handle_financial_analysis(request.question)

    prompt = full_prompt(
        question=request.question,
        context=context,
        chat_history=history,
    )

    # Stream the LLM response
    def sse_generator():
        full_response = []
        try:
            for token in llm_client.stream(prompt):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token, 'session_id': session_id})}\n\n"
        except Exception as exc:
            error_msg = f"⚠️ Streaming error: {exc}"
            full_response.append(error_msg)
            yield f"data: {json.dumps({'token': error_msg, 'session_id': session_id})}\n\n"

        # Record the full response in session history
        complete = "".join(full_response)
        response_builder._append_to_history(session_id, "assistant", complete)

        yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ===================================================================
# Conversation history
# ===================================================================

@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    messages = response_builder.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(**m) for m in messages],
    )


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    response_builder.clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# ===================================================================
# Document upload
# ===================================================================

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document to be ingested into the RAG pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please upload a .pdf file.",
        )

    MAX_SIZE = 50 * 1024 * 1024
    content = await file.read()
    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) // (1024*1024)} MB). Maximum is 50 MB.",
        )

    save_path = config.RAW_DATA_DIR / file.filename
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as exc:
        logger.error("Failed to save uploaded file: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.")

    try:
        num_chunks = response_builder.ingest_document(save_path)
    except Exception as exc:
        logger.error("Failed to ingest %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=500,
            detail=f"File saved but ingestion failed: {exc}",
        )

    return UploadResponse(
        filename=file.filename,
        chunks=num_chunks,
        message=f"Successfully ingested '{file.filename}' into {num_chunks} searchable chunks.",
    )


# ===================================================================
# Ticker quick-info card
# ===================================================================

@app.get("/ticker/{symbol}")
def get_ticker_card(symbol: str):
    """Get quick info for a stock ticker — used by interactive ticker cards."""
    try:
        price_data = response_builder.market_client.get_stock_price(symbol.upper())
        hist = response_builder.market_client.get_historical(symbol.upper(), period="5d")
        sparkline = [d["close"] for d in hist] if hist else []

        return {
            "ticker": price_data["ticker"],
            "price": price_data["price"],
            "currency": price_data["currency"],
            "change_pct": price_data.get("change_pct", 0),
            "sparkline": sparkline,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ===================================================================
# Financial fundamentals
# ===================================================================

@app.get("/fundamentals/{ticker}")
def get_fundamentals(ticker: str):
    try:
        data = response_builder.market_client.get_fundamentals(ticker.upper())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ticker": data["ticker"],
        "name": data.get("name"),
        "currency": data.get("currency"),
        "pe_ratio": data.get("pe_ratio"),
        "forward_pe": data.get("forward_pe"),
        "market_cap": data.get("market_cap"),
        "revenue": data.get("revenue"),
        "profit_margin": data.get("profit_margin"),
        "earnings_growth": data.get("earnings_growth"),
        "dividend_yield": data.get("dividend_yield"),
        "52w_high": data.get("52w_high"),
        "52w_low": data.get("52w_low"),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
    }


# ===================================================================
# Historical prices
# ===================================================================

@app.get("/historical/{ticker}")
def get_historical(ticker: str, period: str = "1mo"):
    valid_periods = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"}
    if period not in valid_periods:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid period '{period}'. Use one of: {', '.join(sorted(valid_periods))}",
        )
    try:
        data = response_builder.market_client.get_historical(ticker.upper(), period=period)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"ticker": ticker.upper(), "period": period, "data": data}


# ===================================================================
# Financial news
# ===================================================================

@app.get("/news/{ticker}")
def get_news(ticker: str):
    """Get latest news headlines for a ticker from Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker.upper())
        news_items = stock.news or []

        results = []
        for item in news_items[:8]:
            results.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", ""),
                "link": item.get("link", ""),
                "published": item.get("providerPublishTime", ""),
                "type": item.get("type", ""),
            })

        return {"ticker": ticker.upper(), "news": results}

    except Exception as exc:
        logger.error("News fetch failed for %s: %s", ticker, exc)
        raise HTTPException(status_code=500, detail=str(exc))
