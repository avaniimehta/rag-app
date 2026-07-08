"""
FAISS-backed vector store with:
 - metadata stored alongside each vector (source file, chunk index, text)
 - idempotent add: each chunk is hashed (sha256 of its text + source path),
   and a chunk whose hash is already stored is skipped, so re-running
   ingestion on an unchanged corpus adds zero duplicate vectors.
"""
import os
import json
import hashlib
import numpy as np
from app import config


def _hash_chunk(source: str, text: str) -> str:
    h = hashlib.sha256()
    h.update(source.encode("utf-8"))
    h.update(b"::")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class VectorStore:
    def __init__(self):
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        self.index = None
        self.metadata = []       # list of dicts, position i == vector i in FAISS index
        self.seen_hashes = set() # for idempotent re-ingest
        self._load()

    def _load(self):
        import faiss
        if os.path.exists(config.INDEX_PATH):
            self.index = faiss.read_index(config.INDEX_PATH)
        else:
            self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)  # cosine sim via normalized vectors

        if os.path.exists(config.METADATA_PATH):
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = [json.loads(line) for line in f if line.strip()]

        if os.path.exists(config.HASHES_PATH):
            with open(config.HASHES_PATH, "r", encoding="utf-8") as f:
                self.seen_hashes = set(json.load(f))

    def save(self):
        import faiss
        faiss.write_index(self.index, config.INDEX_PATH)
        with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m) + "\n")
        with open(config.HASHES_PATH, "w", encoding="utf-8") as f:
            json.dump(sorted(self.seen_hashes), f)

    def add(self, source: str, chunks: list, vectors: np.ndarray, extra_meta: dict = None):
        """
        Adds chunks+vectors, skipping any chunk whose (source, text) hash
        was already ingested. Returns (num_added, num_skipped).
        """
        extra_meta = extra_meta or {}
        new_vectors = []
        added, skipped = 0, 0

        for i, chunk in enumerate(chunks):
            h = _hash_chunk(source, chunk)
            if h in self.seen_hashes:
                skipped += 1
                continue
            self.seen_hashes.add(h)
            new_vectors.append(vectors[i])
            self.metadata.append({
                "source": source,
                "chunk_index": i,
                "text": chunk,
                "hash": h,
                **extra_meta,
            })
            added += 1

        if new_vectors:
            self.index.add(np.vstack(new_vectors).astype("float32"))

        return added, skipped

    def search(self, query_vector: np.ndarray, k: int = 5, source_filter: str = None):
        """
        Returns list of (score, metadata_dict). If source_filter is set,
        over-fetches and filters by metadata['source'] substring match
        (the metadata filter requirement).
        """
        if self.index.ntotal == 0:
            return []

        fetch_k = k * 5 if source_filter else k  # over-fetch to allow filtering
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector.reshape(1, -1).astype("float32"), fetch_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if source_filter and source_filter not in meta["source"]:
                continue
            results.append((float(score), meta))
            if len(results) >= k:
                break
        return results

    @property
    def count(self):
        return self.index.ntotal if self.index else 0
