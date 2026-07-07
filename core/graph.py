"""
LangGraph pipeline — wires Planner → Router → Retriever → Critic → Writer.

rag_pipeline is built lazily inside run_query() instead of at module
import time. This means the graph is only constructed when a query is
actually made, at which point all API keys are already loaded in os.environ.
"""

from langgraph.graph import StateGraph, END
from core.state import RAGState
from agents.planner import run_planner
from agents.router import run_router
from agents.retriever import run_retriever
from agents.critic import run_critic
from agents.writer import run_writer


# ─── Node wrappers ────────────────────────────────────────────────────────────

def planner_node(state: RAGState) -> RAGState:
    return run_planner(state)

def router_node(state: RAGState) -> RAGState:
    return run_router(state)

def retriever_node(state: RAGState) -> RAGState:
    return run_retriever(state)

def critic_node(state: RAGState) -> RAGState:
    return run_critic(state)

def writer_node(state: RAGState) -> RAGState:
    return run_writer(state)


# ─── Conditional edges ────────────────────────────────────────────────────────

def should_retry(state: RAGState) -> str:
    for sq in state["sub_questions"]:
        if not sq["approved"] and sq["retries"] < 2:
            return "retriever"
    return "writer"


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("planner",   planner_node)
    graph.add_node("router",    router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("critic",    critic_node)
    graph.add_node("writer",    writer_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner",   "router")
    graph.add_edge("router",    "retriever")
    graph.add_edge("retriever", "critic")

    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"retriever": "retriever", "writer": "writer"},
    )

    graph.add_edge("writer", END)

    return graph.compile()


# No longer building at module level.
# Pipeline is built once per process on first query, then cached.
_pipeline_cache = None

def run_query(query: str) -> dict:
    """
    Entry point: run the full pipeline for a user query.
    Builds the pipeline on first call (after keys are loaded).
    """
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = build_graph()

    initial_state: RAGState = {
        "original_query": query,
        "sub_questions": [],
        "approved_chunks": [],
        "final_answer": "",
        "sources": [],
        "error": None,
        "current_step": "start",
    }

    result = _pipeline_cache.invoke(initial_state)

    return {
        "answer": result["final_answer"],
        "sources": result["sources"],
        "sub_questions": [
            {"question": sq["question"], "source": sq["source"]}
            for sq in result["sub_questions"]
        ],
    }