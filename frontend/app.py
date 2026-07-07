"""
Streamlit Frontend
──────────────────
Run from project root: streamlit run frontend/app.py

"""

import sys
import os
from pathlib import Path

# Set project root on path FIRST, before any local imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env BEFORE importing any agents or graph
from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

import streamlit as st
import json
import time

st.set_page_config(
    page_title="Multi-Agent RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .agent-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-vector   { background: #e0f2fe; color: #0369a1; }
    .badge-web      { background: #fef9c3; color: #713f12; }
    .source-chip {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 0.75rem;
        margin: 2px;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    # Show current key status from .env
    groq_loaded   = "✅ Loaded" if os.environ.get("GROQ_API_KEY")    else "❌ Missing"
    pine_loaded   = "✅ Loaded" if os.environ.get("PINECONE_API_KEY") else "❌ Missing"
    tavily_loaded = "✅ Loaded" if os.environ.get("TAVILY_API_KEY")   else "❌ Missing"

    st.caption(f"Groq: {groq_loaded} | Pinecone: {pine_loaded} | Tavily: {tavily_loaded}")
    st.divider()

    st.markdown("**Override API Keys** *(optional — leave blank to use .env)*")
    groq_key    = st.text_input("Groq API Key",    type="password")
    pinecone_key= st.text_input("Pinecone API Key",type="password")
    tavily_key  = st.text_input("Tavily API Key",  type="password")
    index_name  = st.text_input("Pinecone Index",  value=os.environ.get("PINECONE_INDEX_NAME", "rag-index"))

    if st.button("💾 Save & Apply Keys", use_container_width=True):
        if groq_key:     os.environ["GROQ_API_KEY"]        = groq_key.strip()
        if pinecone_key: os.environ["PINECONE_API_KEY"]    = pinecone_key.strip()
        if tavily_key:   os.environ["TAVILY_API_KEY"]      = tavily_key.strip()
        if index_name:   os.environ["PINECONE_INDEX_NAME"] = index_name.strip()

        # Reset pipeline cache so next query picks up new keys
        import core.graph as g
        g._pipeline_cache = None

        st.success("Keys applied! Next query will use the new keys.")

    st.divider()

    st.subheader("📤 Ingest Documents")
    uploaded = st.file_uploader(
        "Upload PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("🚀 Ingest", use_container_width=True):
        import tempfile
        from ingestion.ingest import ingest
        progress = st.progress(0)
        for i, f in enumerate(uploaded):
            suffix = Path(f.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            with st.spinner(f"Indexing {f.name}..."):
                ingest(tmp_path, os.environ.get("PINECONE_INDEX_NAME", "rag-index"))
                os.unlink(tmp_path)
            progress.progress((i + 1) / len(uploaded))
        st.success(f"✓ Indexed {len(uploaded)} file(s)")


# ─── Main panel ───────────────────────────────────────────────────────────────
st.title("🧠 Multi-Agent RAG")
st.caption("LangGraph pipeline: Planner → Router → Retriever → Critic → Writer")

with st.expander("🔁 Pipeline Architecture", expanded=False):
    st.markdown("""
    ```
    User Query
        ↓
    🗂 Planner      — breaks query into 2-3 sub-questions
        ↓
    🔀 Router       — decides: vector DB or web search per sub-question
        ↓
    🔍 Retriever    — Pinecone (bge-m3) or Tavily web search
        ↓
    ⚖️ Critic       — scores relevance; rewrites + retries if score < 0.5
        ↓ (loop max 2×)
    ✍️ Writer       — synthesizes answer with inline citations
        ↓
    Final Answer + Sources
    ```
    """)

query = st.text_area(
    "Ask a question",
    placeholder="e.g. How does retrieval-augmented generation compare to fine-tuning?",
    height=80,
)

col1, col2 = st.columns([1, 5])
with col1:
    run_btn = st.button("▶ Run", type="primary", use_container_width=True)

if run_btn and query.strip():
    # Import run_query here (after dotenv loaded) not at file top
    from core.graph import run_query

    step_placeholder = st.empty()
    stages = [
        ("🗂 Planner",   "Decomposing query into sub-questions..."),
        ("🔀 Router",    "Classifying retrieval sources..."),
        ("🔍 Retriever", "Fetching relevant chunks..."),
        ("⚖️ Critic",    "Scoring context quality..."),
        ("✍️ Writer",    "Synthesizing final answer..."),
    ]
    for label, desc in stages:
        step_placeholder.info(f"**{label}** — {desc}")
        time.sleep(0.3)
    step_placeholder.empty()

    with st.spinner("Running pipeline..."):
        try:
            result = run_query(query)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    if result.get("sub_questions"):
        st.subheader("🗂 Sub-Questions")
        for i, sq in enumerate(result["sub_questions"], 1):
            badge_class = "badge-vector" if sq["source"] == "vector" else "badge-web"
            st.markdown(
                f"**{i}.** {sq['question']} "
                f"<span class='agent-badge {badge_class}'>{sq['source']}</span>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader("✍️ Answer")
    st.markdown(result["answer"])

    if result.get("sources"):
        st.subheader("📚 Sources")
        cols = st.columns(min(len(result["sources"]), 3))
        for i, src in enumerate(result["sources"]):
            with cols[i % 3]:
                short = src if len(src) < 60 else src[:57] + "..."
                st.markdown(f"<span class='source-chip'>{short}</span>", unsafe_allow_html=True)

elif run_btn:
    st.warning("Please enter a question.")


# ─── RAGAS metrics panel ──────────────────────────────────────────────────────
st.divider()
with st.expander("📊 RAGAS Evaluation Results"):
    results_path = PROJECT_ROOT / "eval" / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            res = json.load(f)
        metrics = res.get("metrics", {})
        cols = st.columns(4)
        metric_labels = {
            "faithfulness":       "Faithfulness",
            "answer_relevancy":   "Answer Relevancy",
            "context_recall":     "Context Recall",
            "context_precision":  "Context Precision",
        }
        for i, (key, label) in enumerate(metric_labels.items()):
            with cols[i]:
                val = metrics.get(key, 0)
                st.metric(label=label, value=f"{val:.3f}")
                st.progress(val)
        st.caption(f"Evaluated on {res.get('n_questions','?')} questions · {res.get('timestamp','')}")
    else:
        st.info("No evaluation results yet. Run `python eval/ragas_eval.py` to generate metrics.")