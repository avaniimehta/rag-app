"""
Real IR metrics, computed from retrieved results vs a ground-truth
"relevant_sources" list per question. All metrics are binary-relevance
based (a chunk is relevant if its source file is in relevant_sources).
"""
import math


def _is_relevant(meta: dict, relevant_sources: list) -> bool:
    return any(rs in meta["source"] for rs in relevant_sources)


def recall_at_k(retrieved: list, relevant_sources: list, k: int) -> float:
    """Fraction of relevant items that appear in the top-k retrieved."""
    if not relevant_sources:
        return None  # undefined, skip in aggregation
    top_k = retrieved[:k]
    hits = sum(1 for _, meta in top_k if _is_relevant(meta, relevant_sources))
    # cap at number of unique relevant sources actually retrievable
    total_relevant = len(set(relevant_sources))
    return min(hits / total_relevant, 1.0)


def hit_rate(retrieved: list, relevant_sources: list, k: int) -> float:
    """1 if at least one relevant chunk is in top-k, else 0."""
    top_k = retrieved[:k]
    return 1.0 if any(_is_relevant(meta, relevant_sources) for _, meta in top_k) else 0.0


def mrr(retrieved: list, relevant_sources: list) -> float:
    """Reciprocal rank of the first relevant chunk (0 if none found)."""
    for rank, (_, meta) in enumerate(retrieved, start=1):
        if _is_relevant(meta, relevant_sources):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list, relevant_sources: list, k: int) -> float:
    """Binary-relevance nDCG@k."""
    top_k = retrieved[:k]
    dcg = 0.0
    for i, (_, meta) in enumerate(top_k, start=1):
        rel = 1 if _is_relevant(meta, relevant_sources) else 0
        dcg += rel / math.log2(i + 1)

    num_relevant = min(len(set(relevant_sources)), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, num_relevant + 1))
    return dcg / idcg if idcg > 0 else 0.0


def context_precision(retrieved: list, relevant_sources: list, k: int) -> float:
    """Of the top-k retrieved, what fraction are actually relevant."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for _, meta in top_k if _is_relevant(meta, relevant_sources))
    return hits / len(top_k)


def compute_all_retrieval_metrics(retrieved: list, relevant_sources: list, k: int) -> dict:
    return {
        "recall_at_k": recall_at_k(retrieved, relevant_sources, k),
        "hit_rate": hit_rate(retrieved, relevant_sources, k),
        "mrr": mrr(retrieved, relevant_sources),
        "ndcg_at_k": ndcg_at_k(retrieved, relevant_sources, k),
        "context_precision": context_precision(retrieved, relevant_sources, k),
    }
