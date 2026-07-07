---
title: Multi Agent RAG
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🧠 Multi-Agent RAG
A production-grade Retrieval-Augmented Generation system built with LangGraph, Groq (Llama 3.3), Pinecone, and Tavily. Five specialized agents collaborate in a pipeline to decompose questions, retrieve relevant context, evaluate quality, and synthesize cited answers.

## 🏗️ Architecture
```
User Query
    ↓
🗂  Planner      — breaks query into 2-3 focused sub-questions
    ↓
🔀  Router       — classifies each sub-question: vector DB or web search
    ↓
🔍  Retriever    — Pinecone (bge-m3 embeddings) or Tavily web search
    ↓
⚖️  Critic       — scores chunk relevance; rewrites query & retries if score < 0.5
    ↓ (loop max 2×)
✍️  Writer       — synthesizes final answer with inline [Source N] citations
    ↓
Final Answer + Sources
```

## Agent Responsibilities
| Agent | Role |
|---|---|
| Planner | Decomposes the user query into 2–3 independent sub-questions using Llama 3.3 |
| Router | Classifies each sub-question as vector (Pinecone) or web (Tavily) |
| Retriever | Fetches top-5 chunks from Pinecone using bge-m3 embeddings, or from Tavily |
| Critic | Scores retrieved chunks 0–1 for relevance; triggers re-retrieval with rewritten query if score < 0.5 |
| Writer | Synthesizes all approved chunks into a Markdown answer with inline citations |

## 📁 Project Structure
```
multi-agent-rag/
│
├── agents/
│   ├── planner.py        # query decomposition
│   ├── router.py         # vector vs web routing
│   ├── retriever.py      # Pinecone + Tavily search
│   ├── critic.py         # relevance scoring + retry logic
│   └── writer.py         # answer synthesis with citations
│
├── core/
│   ├── graph.py          # LangGraph pipeline wiring
│   ├── state.py          # shared RAGState TypedDict
│   └── embeddings.py     # bge-m3 embedding wrapper
│
├── ingestion/
│   └── ingest.py         # PDF/TXT chunking + Pinecone upsert
│
├── eval/
│   └── ragas_eval.py     # RAGAS faithfulness/relevancy scoring
│
├── api/
│   └── main.py           # FastAPI REST endpoints
│
├── frontend/
│   └── app.py            # Streamlit UI
│
├── papers/               # put your PDFs here
├── .env                  # API keys (never commit this)
├── .env.example          # key template
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start (local)

### 1. Prerequisites
- Python 3.11
- Git

### 2. Clone & Setup
```
git clone https://github.com/yourusername/multi-agent-rag.git
cd multi-agent-rag

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 3. Configure API Keys
```
copy .env.example .env        # Windows
# cp .env.example .env        # Mac/Linux
```
Open `.env` and fill in your real keys:
```
GROQ_API_KEY=gsk_...          # console.groq.com/keys
PINECONE_API_KEY=...          # app.pinecone.io (UUID format)
PINECONE_INDEX_NAME=rag-index
TAVILY_API_KEY=tvly-...       # app.tavily.com
```
⚠️ All services have free tiers. No credit card required.

### 4. Ingest Documents
Download AI papers from arxiv.org and place them in `./papers/`, then:
```
python -m ingestion.ingest --source ./papers/ --index rag-index
```
First run downloads the bge-m3 model (~2GB). Subsequent runs are fast.

### 5. Run the App
```
streamlit run frontend/app.py
```
Opens at http://localhost:8501

## 🔌 API Usage (local only)
```
uvicorn api.main:app --reload --port 8000
```
```
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does RAG compare to fine-tuning for knowledge injection?"}'
```
Swagger docs: http://localhost:8000/docs

## 🐳 Hugging Face Spaces Deployment
This repo is set up for the **Docker SDK** on Spaces (see the YAML block at the top of this file).

1. Create a new Space → SDK: **Docker**.
2. Push this repo's contents to the Space's git remote.
3. In **Settings → Repository secrets**, add:
   - `GROQ_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME`
   - `TAVILY_API_KEY`
4. Do **not** commit your real `.env` — it's gitignored. The app reads the same
   `os.environ.get(...)` calls whether the values come from a local `.env`
   or from Space secrets.
5. The Space serves the **Streamlit** frontend on port 7860 (the FastAPI API
   in `api/main.py` still works locally via `uvicorn`, but isn't exposed by
   this single-container Space setup).

## 📊 RAGAS Evaluation
```
python eval/ragas_eval.py --generate-sample
python eval/ragas_eval.py --questions eval/test_questions.json --output eval/results.json
```

| Metric | What it Measures |
|---|---|
| Faithfulness | Is the answer grounded in retrieved context? |
| Answer Relevancy | Does the answer address the original question? |
| Context Recall | Did retrieval fetch the right chunks? |
| Context Precision | Were the fetched chunks actually useful? |

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | BAAI/bge-m3 (1024-dim, via sentence-transformers) |
| Vector DB | Pinecone (serverless) |
| Web Search | Tavily |
| Orchestration | LangGraph |
| Framework | LangChain |
| Evaluation | RAGAS |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF |

## 📋 Common Issues
| Error | Fix |
|---|---|
| 401 Invalid API Key | Your `.env` still has placeholder text — paste your real key |
| `ModuleNotFoundError: core` | Run with `python -m ingestion.ingest` not `python ingestion/ingest.py` |
| `langchain-pinecone` install fails | You're on Python 3.13 — downgrade to Python 3.11 |
| `torchvision` warnings in terminal | Harmless — add `fileWatcherType = 'none'` to `.streamlit/config.toml` |
| `UnicodeEncodeError` on Pinecone | Your Pinecone key is `pcsk_` format — create a standard UUID key at app.pinecone.io |
| bge-m3 download hangs | Run `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"` separately |