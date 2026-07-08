"""
Splits text into overlapping chunks by word count (a simple, defensible
proxy for tokens - good enough to state as an assumption in the README).
"""
from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    chunk_size and overlap are in words.
    Example: chunk_size=500, overlap=50 means each chunk is ~500 words,
    and the next chunk starts 50 words before the previous one ended
    (so a sentence split across the boundary isn't lost entirely).
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(words):
        chunk_words = words[start:start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks
