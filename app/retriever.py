"""
Retrieval: turns a question into a vector, searches the store,
and applies the "no relevant context" cutoff.
"""
from app import config
from app.embedder import embed_query
from app.vectorstore import VectorStore


def retrieve(store: VectorStore, query: str, k: int = None, source_filter: str = None):
    """
    Returns (results, has_relevant_context).
    results: list of (score, metadata) sorted by score desc.
    has_relevant_context: False if the top result is below MIN_SIMILARITY -
    the caller should NOT ask the LLM to answer in that case.
    """
    k = k or config.DEFAULT_TOP_K
    query_vector = embed_query(query)
    results = store.search(query_vector, k=k, source_filter=source_filter)

    has_relevant_context = bool(results) and results[0][0] >= config.MIN_SIMILARITY
    return results, has_relevant_context
