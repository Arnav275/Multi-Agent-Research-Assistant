"""
Router Agent
────────────
Decides vector vs web search for each sub-question.

"""

import json
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import RAGState
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a search routing classifier. For each question, decide whether it should be answered from:
- "vector": a local knowledge base of indexed research documents (best for technical/academic content already in the corpus)
- "web": a live web search (best for recent events, real-time data, or anything needing current information)

Respond with a JSON array matching the input order. Example:
[{"source": "vector"}, {"source": "web"}, {"source": "vector"}]

Raw JSON only — no explanation, no markdown."""


def run_router(state: RAGState) -> RAGState:
    # instantiate inside function so key is read at call time
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    sub_questions = state["sub_questions"]
    questions_list = [sq["question"] for sq in sub_questions]

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Classify these questions:\n{json.dumps(questions_list)}"),
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        routing = json.loads(raw)
    except Exception:
        routing = [{"source": "vector"}] * len(sub_questions)

    for i, sq in enumerate(sub_questions):
        if i < len(routing):
            sq["source"] = routing[i].get("source", "vector")
        else:
            sq["source"] = "vector"

    state["sub_questions"] = sub_questions
    state["current_step"] = "router_done"
    return state