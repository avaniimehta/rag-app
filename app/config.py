"""
Central config. Everything tunable lives here, read from environment
variables with sane defaults. Nothing secret is hardcoded.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))       # tokens (approx, by words)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))  # tokens

# --- Embedding model ---
# 384-dim, fast, free, runs locally - good enough to defend in interview
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))

# --- Vector store ---
INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.jsonl")
HASHES_PATH = os.path.join(INDEX_DIR, "ingested_hashes.json")

# --- Retrieval ---
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", 5))
# Below this cosine similarity, we treat retrieval as "no relevant context"
# rather than force-feeding weak chunks to the LLM (this is how we avoid
# hallucinating an answer when the corpus doesn't cover the question).
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 0.3))

# --- LLM for answer generation ---
# Ollama runs locally and free - no API key needed. Must have `ollama serve`
# running and the model pulled first: `ollama pull mistral`
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- Corpus ---
CORPUS_DIR = os.getenv("CORPUS_DIR", "data/corpus")
