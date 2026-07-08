"""
Ingestion entrypoint. Ties together: load corpus -> chunk -> embed -> store.

Usage:
    python -m app.ingest
    python -m app.ingest --corpus-dir data/corpus
"""
import argparse
import datetime
from pathlib import Path

from app import config
from app.loaders import load_corpus
from app.chunker import chunk_text
from app.embedder import embed_texts
from app.vectorstore import VectorStore


def ingest(corpus_dir: str = None, chunk_size: int = None, overlap: int = None):
    corpus_dir = corpus_dir or config.CORPUS_DIR
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP

    store = VectorStore()
    total_added, total_skipped, files_seen = 0, 0, 0

    for file_path, raw_text in load_corpus(corpus_dir):
        files_seen += 1
        chunks = chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            print(f"[ingest] {file_path}: no text extracted, skipping")
            continue

        vectors = embed_texts(chunks)
        extra_meta = {
            "doc_type": Path(file_path).suffix.lower().lstrip("."),
            "ingested_at": datetime.datetime.utcnow().isoformat(),
        }
        added, skipped = store.add(file_path, chunks, vectors, extra_meta=extra_meta)
        total_added += added
        total_skipped += skipped
        print(f"[ingest] {file_path}: {len(chunks)} chunks -> {added} added, {skipped} skipped (already ingested)")

    store.save()
    print(f"\n[ingest] DONE. Files processed: {files_seen} | "
          f"Vectors added: {total_added} | Vectors skipped (dupes): {total_skipped} | "
          f"Total vectors in store: {store.count}")
    return {"files": files_seen, "added": total_added, "skipped": total_skipped, "total": store.count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    args = parser.parse_args()
    ingest(args.corpus_dir, args.chunk_size, args.overlap)
