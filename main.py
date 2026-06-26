"""
FastAPI — REST API for the multi-agent RAG pipeline.

Endpoints:
  POST /query        — run a query through the full pipeline
  POST /ingest       — trigger ingestion from an uploaded file
  GET  /health       — health check
  GET  /docs         — auto-generated Swagger UI

Run:
    uvicorn api.main:app --reload --port 8000
"""

import sys
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile

app = FastAPI(
    title="Multi-Agent RAG API",
    description="LangGraph-powered multi-agent RAG with Planner → Router → Retriever → Critic → Writer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    stream: Optional[bool] = False


class SubQuestionInfo(BaseModel):
    question: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    sub_questions: list[SubQuestionInfo]


class IngestResponse(BaseModel):
    message: str
    chunks_indexed: Optional[int] = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "multi-agent-rag"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Run the full multi-agent RAG pipeline on a user query.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        from core.graph import run_query
        result = run_query(req.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            sub_questions=[
                SubQuestionInfo(question=sq["question"], source=sq["source"])
                for sq in result["sub_questions"]
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    index_name: str = "rag-index",
):
    """
    Upload a PDF or text file and index it into Pinecone.
    Ingestion runs in the background; returns immediately.
    """
    allowed = {".pdf", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    # Save to temp file
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    def _bg_ingest(path: str, idx: str):
        from ingestion.ingest import ingest
        ingest(path, idx)
        os.unlink(path)

    background_tasks.add_task(_bg_ingest, tmp_path, index_name)

    return IngestResponse(
        message=f"Ingestion started for '{file.filename}'. Chunks will appear in Pinecone shortly.",
    )


# ─── Dev server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)