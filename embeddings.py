"""
BGE-M3 embedding logic using sentence-transformers.
Cached model loader to avoid repeated downloads.
"""

from functools import lru_cache
from typing import List
import numpy as np


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    print("Loading bge-m3 model (first run only)...")
    model = SentenceTransformer("BAAI/bge-m3")
    return model


def embed_texts(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Embed a list of strings using bge-m3.
    Returns a list of float vectors (1024-dim).
    """
    model = _load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 10,
        normalize_embeddings=True,   # cosine-ready
    )
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """
    Single-query embedding with BGE instruction prefix for retrieval.
    """
    instruction = f"Represent this sentence for searching relevant passages: {query}"
    return embed_texts([instruction])[0]