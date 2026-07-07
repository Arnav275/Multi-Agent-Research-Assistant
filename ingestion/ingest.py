"""
Ingestion Pipeline
──────────────────
Chunks PDF/text documents, embeds with bge-m3, upserts to Pinecone.

Usage:
    python ingestion/ingest.py --source ./papers/ --index rag-index

Supports:
    - PDFs (via PyMuPDF)
    - Plain text files (.txt, .md)
    - Recursive directory traversal
"""

import os
import uuid
import argparse
from pathlib import Path
from typing import List, Generator
from dotenv import load_dotenv

load_dotenv()


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[str]:
    """
    Sliding window chunker on word boundaries.
    chunk_size and overlap are in tokens (approx words).
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:  # skip tiny fragments
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ─── File reading ──────────────────────────────────────────────────────────────

def read_pdf(path: Path) -> str:
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    return "\n".join(page.get_text() for page in doc)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def iter_documents(source: str) -> Generator[tuple[str, str], None, None]:
    """Yields (file_path_str, text_content) for all supported files."""
    p = Path(source)
    files = list(p.rglob("*.pdf")) + list(p.rglob("*.txt")) + list(p.rglob("*.md")) if p.is_dir() else [p]

    for f in files:
        print(f"  Reading: {f.name}")
        try:
            if f.suffix.lower() == ".pdf":
                text = read_pdf(f)
            else:
                text = read_text(f)
            yield str(f), text
        except Exception as e:
            print(f"  ⚠ Skipped {f.name}: {e}")


# ─── Pinecone upsert ──────────────────────────────────────────────────────────

def upsert_to_pinecone(
    records: List[dict],
    index_name: str,
    batch_size: int = 100,
):
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Create index if it doesn't exist (bge-m3 = 1024 dims)
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' (1024 dims, cosine)...")
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)

    # Batch upsert
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        vectors = [
            {
                "id": rec["id"],
                "values": rec["embedding"],
                "metadata": {"text": rec["text"], "source": rec["source"]},
            }
            for rec in batch
        ]
        index.upsert(vectors=vectors)
        print(f"  Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def ingest(source: str, index_name: str):
    from core.embeddings import embed_texts

    print(f"\n{'='*50}")
    print(f"Ingestion Pipeline")
    print(f"Source : {source}")
    print(f"Index  : {index_name}")
    print(f"{'='*50}\n")

    all_chunks = []

    for file_path, text in iter_documents(source):
        chunks = chunk_text(text)
        print(f"  → {len(chunks)} chunks from {Path(file_path).name}")
        for chunk in chunks:
            all_chunks.append({"source": file_path, "text": chunk})

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embedding with bge-m3...")

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    records = [
        {
            "id": str(uuid.uuid4()),
            "embedding": emb,
            "text": chunk["text"],
            "source": chunk["source"],
        }
        for chunk, emb in zip(all_chunks, embeddings)
    ]

    print(f"\nUploading to Pinecone index '{index_name}'...")
    upsert_to_pinecone(records, index_name)

    print(f"\n✓ Ingestion complete — {len(records)} vectors indexed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    parser.add_argument("--source", required=True, help="Path to file or folder")
    parser.add_argument("--index", default="rag-index", help="Pinecone index name")
    args = parser.parse_args()
    ingest(args.source, args.index)