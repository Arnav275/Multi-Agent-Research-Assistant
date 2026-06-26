"""
Planner Agent
─────────────
Breaks the user query into 2-3 focused sub-questions.

FIX: ChatGroq is instantiated inside run_planner(), not at module level.
This ensures the API key is read from os.environ at call time, not import time.
"""

import json
import uuid
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import RAGState, SubQuestion
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a query decomposition expert. Given a complex user question,
break it into 2-3 focused sub-questions that:
1. Can each be answered independently from documents or the web
2. Together cover the full scope of the original question
3. Are specific and self-contained

Respond ONLY with a JSON array. Example:
[
  {"question": "What is retrieval-augmented generation?"},
  {"question": "What are the main limitations of RAG systems?"},
  {"question": "How do recent papers address RAG hallucination?"}
]

No explanation, no markdown fences — raw JSON array only."""


def run_planner(state: RAGState) -> RAGState:
    # ✅ FIX: instantiate inside function so key is read at call time
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    query = state["original_query"]

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Decompose this query: {query}"),
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
    except Exception:
        parsed = [{"question": query}]

    sub_questions: list[SubQuestion] = [
        {
            "id": str(uuid.uuid4())[:8],
            "question": item["question"],
            "source": "unknown",
            "chunks": [],
            "approved": False,
            "retries": 0,
            "rewritten_query": None,
        }
        for item in parsed
    ]

    state["sub_questions"] = sub_questions
    state["current_step"] = "planner_done"
    return state