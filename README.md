# 🧠 Multi-Agent RAG

A production-grade Retrieval-Augmented Generation system built with **LangGraph**, **Groq (Llama 3.3)**, **Pinecone**, and **Tavily**. Five specialized agents collaborate in a pipeline to decompose questions, retrieve relevant context, evaluate quality, and synthesize cited answers.

---

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

### Agent Responsibilities

| Agent | Role |
|---|---|
| **Planner** | Decomposes the user query into 2–3 independent sub-questions using Llama 3.3 |
| **Router** | Classifies each sub-question as `vector` (Pinecone) or `web` (Tavily) |
| **Retriever** | Fetches top-5 chunks from Pinecone using bge-m3 embeddings, or from Tavily |
| **Critic** | Scores retrieved chunks 0–1 for relevance; triggers re-retrieval with rewritten query if score < 0.5 |
| **Writer** | Synthesizes all approved chunks into a Markdown answer with inline citations |

---

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

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.11
- Git

### 2. Clone & Setup

```bash
git clone https://github.com/yourusername/multi-agent-rag.git
cd multi-agent-rag

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Mac/Linux
```

Open `.env` and fill in your real keys:

```env
GROQ_API_KEY=gsk_...          # console.groq.com/keys
PINECONE_API_KEY=...          # app.pinecone.io (UUID format)
PINECONE_INDEX_NAME=rag-index
TAVILY_API_KEY=tvly-...       # app.tavily.com
```

> ⚠️ All services have **free tiers**. No credit card required.

### 4. Ingest Documents

Download AI papers from [arxiv.org](https://arxiv.org) and place them in `./papers/`, then:

```bash
python -m ingestion.ingest --source ./papers/ --index rag-index
```

First run downloads the bge-m3 model (~2GB). Subsequent runs are fast.

### 5. Run the App

```bash
streamlit run frontend/app.py
```

Opens at **http://localhost:8501**

---

## 🔌 API Usage

Start the REST API:

```bash
uvicorn api.main:app --reload --port 8000
```

**Query endpoint:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does RAG compare to fine-tuning for knowledge injection?"}'
```

**Response:**
```json
{
  "answer": "## Answer\n\nRAG and fine-tuning differ in... [Source 1]...\n\n## Sources",
  "sources": ["papers/rag_paper.pdf", "https://arxiv.org/..."],
  "sub_questions": [
    {"question": "What is RAG?", "source": "vector"},
    {"question": "What is fine-tuning?", "source": "vector"},
    {"question": "Latest benchmarks comparing RAG vs fine-tuning?", "source": "web"}
  ]
}
```

**Swagger docs:** http://localhost:8000/docs

---

## 📊 RAGAS Evaluation

Generate a sample test set and evaluate:

```bash
# Create sample questions
python eval/ragas_eval.py --generate-sample

# Run evaluation (add your own questions to eval/test_questions.json first)
python eval/ragas_eval.py --questions eval/test_questions.json --output eval/results.json
```

### Metrics Tracked

| Metric | What it Measures |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevancy** | Does the answer address the original question? |
| **Context Recall** | Did retrieval fetch the right chunks? |
| **Context Precision** | Were the fetched chunks actually useful? |

Results appear automatically in the Streamlit UI's metrics panel.

---

## 🐳 Docker

```bash
# Build
docker build -t multi-agent-rag .

# Run API
docker run -p 8000:8000 --env-file .env multi-agent-rag

# Run Streamlit
docker run -p 8501:8501 --env-file .env multi-agent-rag \
  streamlit run frontend/app.py --server.address 0.0.0.0
```

---

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

---

## ⚙️ Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key | required |
| `PINECONE_API_KEY` | Pinecone API key | required |
| `PINECONE_INDEX_NAME` | Pinecone index name | `rag-index` |
| `TAVILY_API_KEY` | Tavily search API key | required |

---

## 🧩 Extending the Pipeline

**Add a new agent:** Create `agents/my_agent.py`, add a node in `core/graph.py`, wire it with `graph.add_edge()`.

**Change the LLM:** Replace `llama-3.3-70b-versatile` with any Groq-supported model in each agent file.

**Change the embedding model:** Edit `core/embeddings.py` — update the model name and Pinecone index dimension to match.

**Add more document types:** Extend `iter_documents()` in `ingestion/ingest.py` to handle `.docx`, `.html`, etc.

---

## 📋 Common Issues

| Error | Fix |
|---|---|
| `401 Invalid API Key` | Your `.env` still has placeholder text — paste your real key |
| `ModuleNotFoundError: core` | Run with `python -m ingestion.ingest` not `python ingestion/ingest.py` |
| `langchain-pinecone` install fails | You're on Python 3.13 — downgrade to Python 3.11 |
| `torchvision` warnings in terminal | Harmless — add `fileWatcherType = 'none'` to `.streamlit/config.toml` |
| `UnicodeEncodeError` on Pinecone | Your Pinecone key is `pcsk_` format — create a standard UUID key at app.pinecone.io |
| bge-m3 download hangs | Run `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"` separately |

---

## Sample output

<img width="960" height="600" alt="Screenshot 2026-06-13 211804" src="https://github.com/user-attachments/assets/afc430a2-2848-4539-aab5-27a549dbe3b6" />

<img width="960" height="600" alt="Screenshot 2026-06-13 225754" src="https://github.com/user-attachments/assets/6a044b5d-7d1d-4d5f-bdc6-ad2e75cb20a8" />
