FROM python:3.11-slim

WORKDIR /app

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download bge-m3 model at build time (saves first-request latency)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" || true

# Hugging Face Spaces (Docker SDK) routes traffic to this port
EXPOSE 7860

# Streamlit is the user-facing app for the Space.
# (FastAPI's api/main.py still works locally via `uvicorn api.main:app`,
#  but a single Space container should expose one entrypoint.)
CMD ["streamlit", "run", "frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"]