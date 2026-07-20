"""
backend/app.py
==============
FastAPI backend for the Golf Analytics RAG Chatbot.

Endpoints:
  GET  /health          — health check
  POST /chat            — RAG-powered Q&A
  POST /ingest/reload   — rebuild FAISS index from CSVs
  GET  /widget          — serve the floating chatbot HTML

Run:
  python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import pickle
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR    = Path(__file__).resolve().parent.parent
WIDGET_HTML = BASE_DIR / "chatbot" / "widget.html"


# ── startup ────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading retriever …")
    from rag.retriever import retrieve
    try:
        retrieve("warmup", top_k=1)
        logger.info("Retriever loaded successfully.")
    except Exception as e:
        logger.warning(f"Retriever warmup failed (will retry on first request): {e}")
    logger.info("Backend ready.")
    yield
    logger.info("Shutting down.")


# ── app ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Golf Analytics RAG Backend",
    description="RAG-powered chatbot API for the Golf Analytics Dashboard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k:    int = Field(default=8, ge=1, le=20)


class SourceChunk(BaseModel):
    text:   str
    source: str
    score:  float
    course: Optional[str] = None
    date:   Optional[str] = None


class ChatResponse(BaseModel):
    answer:   str
    sources:  list[SourceChunk]
    question: str


class IngestResponse(BaseModel):
    status:       str
    chunks_added: int


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "Golf Analytics RAG Backend"}


@app.get("/widget", include_in_schema=False)
def serve_widget():
    """Serve the floating chatbot widget HTML page."""
    if not WIDGET_HTML.exists():
        raise HTTPException(status_code=404, detail="Widget HTML not found.")
    return FileResponse(str(WIDGET_HTML), media_type="text/html")


_GREETING_RE = re.compile(
    r"^\s*(hi+|hello+|hey+|howdy|greetings|good\s*(morning|afternoon|evening)|what'?s\s*up|sup)\W*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "👋 Hello! I'm your <strong>Golf Analytics Assistant</strong>.<br><br>"
    "Here's what you can ask me:<br>"
    "• 📈 <em>Which course has the highest occupancy?</em><br>"
    "• 💰 <em>What is the average price for Stonegate Golf Club?</em><br>"
    "• 📊 <em>Compare prices across all courses</em><br>"
    "• 🟢 <em>Available tee times on GolfNow?</em><br><br>"
    "How can I help you today?"
)


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(req: ChatRequest):
    """
    RAG pipeline:
      1. Detect greetings and return a friendly guide response
      2. Retrieve top-k relevant chunks from FAISS index
      3. Generate answer with Gemini / OpenAI
      4. Return answer + source chunks
    """
    from rag.retriever import retrieve
    from rag.llm import generate_answer
    from rag.data_assistant import answer_data_query

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Greeting shortcut (handle greetings first to avoid mixing data replies) ─
    if _GREETING_RE.match(question):
        q = question.lower()
        if "good morning" in q or "good evening" in q:
            try:
                from rag.data_assistant import _time_based_greeting
                greeting = _time_based_greeting()
            except Exception:
                greeting = "Hello"
            html = f"<strong>{greeting}!</strong> 😊 How can I help you?"
        else:
            html = "Hello! 😊 How can I help you?"
        return ChatResponse(answer=html, sources=[], question=question)

    # Prefer deterministic data-first answers for analytics and small talk.
    deterministic_answer = answer_data_query(question)
    if deterministic_answer is not None:
        return ChatResponse(
            answer=deterministic_answer,
            sources=[],
            question=question,
        )

    logger.info(f"Query: {question!r}")
    chunks = retrieve(question, top_k=req.top_k)

    if not chunks:
        return ChatResponse(
            answer="I couldn't find relevant data for your question. Try asking about course prices, availability, or occupancy.",
            sources=[],
            question=question,
        )

    answer  = generate_answer(chunks, question)
    sources = [
        SourceChunk(
            text=c["text"],
            source=c.get("source", "unknown"),
            score=round(c.get("score", 0.0), 4),
            course=c.get("course"),
            date=c.get("date"),
        )
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources, question=question)


@app.post("/ingest/reload", response_model=IngestResponse, tags=["Ingest"])
def ingest_reload():
    """Rebuild the FAISS index from CSV files on disk (uses cached Drive data if available)."""
    from rag.ingest import build_index, INDEX_FILE, CHUNKS_FILE
    import rag.retriever as ret

    build_index(force=True)
    ret._index = faiss.read_index(str(INDEX_FILE))
    with open(CHUNKS_FILE, "rb") as f:
        ret._chunks = pickle.load(f)

    # Also clear the data_assistant LRU cache so it picks up the new index data
    try:
        from rag.data_assistant import _load_data
        _load_data.cache_clear()
        logger.info("data_assistant cache cleared after index rebuild")
    except Exception as e:
        logger.warning(f"Could not clear data_assistant cache: {e}")

    return IngestResponse(status="rebuilt", chunks_added=ret._index.ntotal)


@app.post("/drive/refresh", response_model=IngestResponse, tags=["Ingest"])
def drive_refresh():
    """
    Delete the cached Drive dataset and re-download it, then rebuild the FAISS index.
    Use this when the Google Drive dataset has been updated.
    """
    from rag.ingest import build_index, INDEX_FILE, CHUNKS_FILE, _DRIVE_CSV_CACHE
    import rag.retriever as ret

    # Delete cached Drive CSV so _load_drive_dataset() re-downloads it
    if _DRIVE_CSV_CACHE.exists():
        _DRIVE_CSV_CACHE.unlink()
        logger.info(f"Deleted Drive cache: {_DRIVE_CSV_CACHE}")

    build_index(force=True)
    ret._index = faiss.read_index(str(INDEX_FILE))
    with open(CHUNKS_FILE, "rb") as f:
        ret._chunks = pickle.load(f)

    # Clear data_assistant LRU cache so it reloads with fresh Drive data
    try:
        from rag.data_assistant import _load_data
        _load_data.cache_clear()
        logger.info("data_assistant cache cleared after Drive refresh")
    except Exception as e:
        logger.warning(f"Could not clear data_assistant cache: {e}")

    return IngestResponse(status="drive_refreshed", chunks_added=ret._index.ntotal)
