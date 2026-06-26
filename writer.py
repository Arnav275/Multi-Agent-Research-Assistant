"""
Writer Agent
────────────
Synthesizes approved chunks into a final answer with citations.

FIX: ChatGroq instantiated inside run_writer(), not at module level.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import RAGState
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are an expert research synthesizer. You will be given:
1. An original user question
2. A list of retrieved context chunks with source identifiers

Your task:
- Write a comprehensive, well-structured answer to the original question
- Cite sources inline using [Source N] notation
- Do NOT fabricate information — only use what is in the provided context
- If context is insufficient for part of the answer, say so explicitly
- End with a "## Sources" section listing each unique source URL or identifier

Format: Markdown. Be thorough but concise."""


def run_writer(state: RAGState) -> RAGState:
    # ✅ FIX: instantiate inside function so key is read at call time
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    query = state["original_query"]
    chunks = state["approved_chunks"]

    if not chunks:
        state["final_answer"] = (
            "I could not retrieve sufficient context to answer your question. "
            "Please try rephrasing or check that documents have been indexed."
        )
        state["sources"] = []
        return state

    seen_sources = {}
    context_lines = []
    source_counter = 1

    for chunk in chunks:
        src = chunk.get("source", "Unknown")
        if src not in seen_sources:
            seen_sources[src] = source_counter
            source_counter += 1
        src_num = seen_sources[src]
        context_lines.append(f"[Source {src_num}] {chunk['text'][:600]}")

    context_block = "\n\n".join(context_lines)
    sources_index = "\n".join(
        f"[Source {num}]: {src}" for src, num in seen_sources.items()
    )

    prompt = f"""Original Question: {query}

---
Retrieved Context:
{context_block}

---
Source Index:
{sources_index}

---
Write your answer now:"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

    state["final_answer"] = response.content
    state["sources"] = list(seen_sources.keys())
    state["current_step"] = "writer_done"
    return state