"""
HTTP API for the RAG service.

Endpoints:
  GET  /health           - liveness check
  POST /query            - ask a question, get a grounded answer
  POST /ingest            - trigger (re-)ingestion of the corpus directory

Run with:
  uvicorn app.main:app --reload --port 8000
"""
import logging
import time
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional

from app import config
from app.vectorstore import VectorStore
from app.retriever import retrieve
from app.answer import generate_answer
from app.ingest import ingest as run_ingest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-app")

app = FastAPI(title="Cost-Efficient RAG Service")

# Loaded once at startup, shared across requests
_store = VectorStore()


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(default=None, description="overrides DEFAULT_TOP_K")
    source_filter: Optional[str] = Field(default=None, description="substring match on source file path")


class QueryResponse(BaseModel):
    answer: str
    cited_sources: list
    chunk_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int


@app.get("/health")
def health():
    return {"status": "ok", "vectors_in_store": _store.count}


@app.post("/ingest")
def ingest_endpoint(corpus_dir: Optional[str] = None):
    result = run_ingest(corpus_dir=corpus_dir)
    global _store
    _store = VectorStore()  # reload so new vectors are visible to /query
    return result


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.time()

    results, has_context = retrieve(_store, req.question, k=req.top_k, source_filter=req.source_filter)
    retrieval_latency_ms = round((time.time() - t0) * 1000, 1)

    chunks = [meta for _, meta in results]
    t1 = time.time()
    result = generate_answer(req.question, chunks, has_relevant_context=has_context)
    generation_latency_ms = round((time.time() - t1) * 1000, 1)

    total_latency_ms = round((time.time() - t0) * 1000, 1)

    logger.info(
        f"query='{req.question}' chunk_count={result['chunk_count']} "
        f"retrieval_ms={retrieval_latency_ms} generation_ms={generation_latency_ms} "
        f"total_ms={total_latency_ms} prompt_tokens={result['prompt_tokens']} "
        f"completion_tokens={result['completion_tokens']}"
    )

    return QueryResponse(
        answer=result["answer"],
        cited_sources=result["cited_sources"],
        chunk_count=result["chunk_count"],
        retrieval_latency_ms=retrieval_latency_ms,
        generation_latency_ms=generation_latency_ms,
        total_latency_ms=total_latency_ms,
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
    )
