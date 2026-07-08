"""
Runs the fixed question set (eval/questions.json) through the full RAG
pipeline, computes retrieval + answer metrics for every question, and
writes an aggregated results file (eval/results.json) plus a p50/p95
latency summary.

Usage:
    python -m eval.run_eval
"""
import json
import time
import statistics
from pathlib import Path

from app import config
from app.vectorstore import VectorStore
from app.retriever import retrieve
from app.answer import generate_answer
from eval.retrieval_metrics import compute_all_retrieval_metrics
from eval.answer_metrics import exact_match, f1_score, llm_judge

QUESTIONS_PATH = "eval/questions.json"
RESULTS_PATH = "eval/results.json"


def run_eval(k: int = None):
    k = k or config.DEFAULT_TOP_K
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    store = VectorStore()
    if store.count == 0:
        print("[eval] WARNING: vector store is empty. Run `python -m app.ingest` first.")

    per_question_results = []
    latencies_ms = []

    for q in questions:
        t0 = time.time()
        retrieved, has_context = retrieve(store, q["question"], k=k)
        chunks = [meta for _, meta in retrieved]
        answer_result = generate_answer(q["question"], chunks, has_relevant_context=has_context)
        total_latency = round((time.time() - t0) * 1000, 1)
        latencies_ms.append(total_latency)

        retrieval_metrics = compute_all_retrieval_metrics(retrieved, q["relevant_sources"], k=k)

        answer_metrics = {}
        if q.get("gold_answer"):
            answer_metrics["exact_match"] = exact_match(answer_result["answer"], q["gold_answer"])
            answer_metrics["f1"] = f1_score(answer_result["answer"], q["gold_answer"])

        context_text = "\n\n".join(c["text"] for c in chunks) if chunks else "(no context retrieved)"
        judge_result = llm_judge(q["question"], context_text, answer_result["answer"])
        answer_metrics["faithfulness"] = judge_result["faithfulness"]
        answer_metrics["relevance"] = judge_result["relevance"]
        answer_metrics["judge_parse_ok"] = judge_result["parse_ok"]

        per_question_results.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer_result["answer"],
            "chunk_count": answer_result["chunk_count"],
            "total_latency_ms": total_latency,
            "retrieval_metrics": retrieval_metrics,
            "answer_metrics": answer_metrics,
        })
        print(f"[eval] {q['id']}: recall@{k}={retrieval_metrics['recall_at_k']} "
              f"mrr={round(retrieval_metrics['mrr'],3)} "
              f"faithfulness={answer_metrics.get('faithfulness')} "
              f"latency={total_latency}ms")

    aggregated = _aggregate(per_question_results, latencies_ms, k)

    output = {"k": k, "per_question": per_question_results, "aggregated": aggregated}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[eval] DONE. Results written to {RESULTS_PATH}")
    print(json.dumps(aggregated, indent=2))
    return output


def _aggregate(results, latencies_ms, k):
    def avg(key_path):
        vals = []
        for r in results:
            v = r
            for key in key_path:
                v = v.get(key) if v else None
            if v is not None:
                vals.append(v)
        return round(statistics.mean(vals), 4) if vals else None

    latencies_sorted = sorted(latencies_ms)
    n = len(latencies_sorted)

    def percentile(p):
        if not latencies_sorted:
            return None
        idx = min(int(round(p * (n - 1))), n - 1)
        return latencies_sorted[idx]

    return {
        "num_questions": len(results),
        "avg_recall_at_k": avg(["retrieval_metrics", "recall_at_k"]),
        "avg_hit_rate": avg(["retrieval_metrics", "hit_rate"]),
        "avg_mrr": avg(["retrieval_metrics", "mrr"]),
        "avg_ndcg_at_k": avg(["retrieval_metrics", "ndcg_at_k"]),
        "avg_context_precision": avg(["retrieval_metrics", "context_precision"]),
        "avg_faithfulness": avg(["answer_metrics", "faithfulness"]),
        "avg_relevance": avg(["answer_metrics", "relevance"]),
        "avg_f1": avg(["answer_metrics", "f1"]),
        "avg_exact_match": avg(["answer_metrics", "exact_match"]),
        "p50_latency_ms": percentile(0.50),
        "p95_latency_ms": percentile(0.95),
    }


if __name__ == "__main__":
    run_eval()
