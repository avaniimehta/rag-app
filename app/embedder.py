"""
Turns text chunks into vectors (embeddings) using a local, free model.
Model: sentence-transformers/all-MiniLM-L6-v2 -> 384-dimensional vectors.
Loaded once and reused (loading it fresh per call is slow).
"""
from functools import lru_cache
import numpy as np
from app import config


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: list) -> np.ndarray:
    """
    Returns a (len(texts), EMBEDDING_DIM) float32 numpy array.
    Normalized so cosine similarity == dot product (needed for FAISS IndexFlatIP).
    """
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    vectors = vectors.astype("float32")
    # L2-normalize each row
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    vectors = vectors / norms
    return vectors


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
