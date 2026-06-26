"""
Critic Agent
────────────
Scores retrieved chunks for relevance, triggers re-retrieval if needed.

FIX: ChatGroq instantiated inside functions, not at module level.
"""

import json
import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import RAGState
from dotenv import load_dotenv

load_dotenv()

SCORE_PROMPT = """You are a retrieval quality critic. Given a question and retrieved text chunks,
score the overall relevance of the chunks to the question.

Return a JSON object:
{
  "score": <float 0.0-1.0>,
  "reason": "<one sentence>",
  "rewritten_query": "<improved query if score < 0.5, else null>"
}

Score guide:
- 0.9-1.0: Chunks directly and completely answer the question
- 0.7-0.8: Chunks are mostly relevant with minor gaps
- 0.5-0.6: Chunks partially relevant, key info missing
- 0.0-0.4: Chunks are off-topic or unhelpful

Raw JSON only."""


def _score_chunks(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {
            "score": 0.0,
            "reason": "No chunks retrieved",
            "rewritten_query": question + " overview explanation",
        }

    # ✅ FIX: instantiate inside function so key is read at call time
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    context_preview = "\n\n".join(
        f"[Chunk {i+1}] {c['text'][:400]}"
        for i, c in enumerate(chunks[:5])
    )

    response = llm.invoke([
        SystemMessage(content=SCORE_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nChunks:\n{context_preview}"),
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        return {"score": 0.6, "reason": "Parse error — defaulting to pass", "rewritten_query": None}


def run_critic(state: RAGState) -> RAGState:
    sub_questions = state["sub_questions"]
    approved_chunks = []

    for sq in sub_questions:
        if sq["approved"]:
            approved_chunks.extend(sq["chunks"])
            continue

        verdict = _score_chunks(sq["question"], sq["chunks"])
        score = verdict.get("score", 0.0)

        if score >= 0.5 or sq["retries"] >= 2:
            sq["approved"] = True
            approved_chunks.extend(sq["chunks"])
        else:
            sq["approved"] = False
            sq["rewritten_query"] = verdict.get("rewritten_query") or sq["question"]

    state["sub_questions"] = sub_questions
    state["approved_chunks"] = approved_chunks
    state["current_step"] = "critic_done"
    return state