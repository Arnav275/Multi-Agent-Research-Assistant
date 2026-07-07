"""
Shared state schema for the multi-agent RAG pipeline.
All agents read from and write to this TypedDict.
"""

from typing import TypedDict, Annotated, List, Optional
import operator


class SubQuestion(TypedDict):
    id: str
    question: str
    source: str          # "vector" | "web"
    chunks: List[dict]   # retrieved chunks
    approved: bool       # Critic verdict
    retries: int         # retry counter (max 2)
    rewritten_query: Optional[str]


class RAGState(TypedDict):
    # Input
    original_query: str

    # Planner output
    sub_questions: List[SubQuestion]

    # Aggregated approved context
    approved_chunks: Annotated[List[dict], operator.add]

    # Final output
    final_answer: str
    sources: List[str]

    # Control
    error: Optional[str]
    current_step: str