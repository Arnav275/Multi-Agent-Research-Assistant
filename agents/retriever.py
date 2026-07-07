"""
Retriever Agent
───────────────
Executes retrieval for each sub-question based on its routed source:
  - vector → Pinecone semantic search (bge-m3 embeddings)
  - web    → Tavily search API

Only retrieves sub-questions that are not yet approved (supports retries).
"""

import os
from typing import List
from core.state import RAGState
from core.embeddings import embed_query
from dotenv import load_dotenv

load_dotenv()

TOP_K = 5


def _pinecone_search(query: str) -> List[dict]:
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ.get("PINECONE_INDEX_NAME", "rag-index"))

    vector = embed_query(query)
    results = index.query(vector=vector, top_k=TOP_K, include_metadata=True)

    chunks = []
    for match in results.matches:
        chunks.append({
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("source", "pinecone"),
            "score": float(match.score),
            "type": "vector",
        })
    return chunks


def _tavily_search(query: str) -> List[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=TOP_K)

    chunks = []
    for result in response.get("results", []):
        chunks.append({
            "text": result.get("content", ""),
            "source": result.get("url", "web"),
            "score": result.get("score", 0.5),
            "type": "web",
        })
    return chunks


def run_retriever(state: RAGState) -> RAGState:
    sub_questions = state["sub_questions"]

    for sq in sub_questions:
        if sq["approved"]:
            continue

        query = sq["rewritten_query"] or sq["question"]

        try:
            if sq["source"] == "web":
                chunks = _tavily_search(query)
            else:
                chunks = _pinecone_search(query)
        except Exception as e:
            chunks = [{
                "text": f"Retrieval error: {str(e)}",
                "source": "error",
                "score": 0.0,
                "type": sq["source"],
            }]

        sq["chunks"] = chunks
        sq["retries"] += 1

    state["sub_questions"] = sub_questions
    state["current_step"] = "retriever_done"
    return state