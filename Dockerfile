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

# Pre-download bge-m3 model at build time (optional — saves first-request latency)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')" || true

EXPOSE 8000 8501

# Default: run FastAPI. Override CMD to run Streamlit instead.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]