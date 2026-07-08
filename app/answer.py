"""
Takes retrieved chunks + the question, calls a local Ollama model, and
returns a grounded answer that cites which chunk(s) it used.

If retrieval found nothing relevant, we never call the LLM at all -
that's the actual mechanism that prevents hallucination here (not just
a prompt instruction, which models can ignore).
"""
import time
import requests
from app import config

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything in the corpus relevant to this question, "
    "so I'm not going to guess. Try rephrasing, or this may genuinely be "
    "outside the ingested documents."
)


def _build_prompt(question: str, chunks: list) -> str:
    context_block = "\n\n".join(
        f"[Source {i+1}: {c['source']} | chunk {c['chunk_index']}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"""You are a QA assistant. Answer the question using ONLY the context below.
Every claim in your answer must be traceable to a specific [Source N] tag.
After your answer, add a line "Cited sources: " listing which Source N numbers you used.
If the context does not actually contain the answer, say so plainly instead of guessing.

Context:
{context_block}

Question: {question}

Answer:"""


def generate_answer(question: str, retrieved_chunks: list, has_relevant_context: bool):
    """
    retrieved_chunks: list of metadata dicts (from retriever results, score stripped).
    Returns a dict with the answer, citations, latency, and token usage -
    all of which get logged per the assignment's logging requirement.
    """
    start = time.time()

    if not has_relevant_context:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "cited_sources": [],
            "chunk_count": 0,
            "latency_ms": round((time.time() - start) * 1000, 1),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    prompt = _build_prompt(question, retrieved_chunks)

    response = requests.post(
        f"{config.OLLAMA_HOST}/api/generate",
        json={
            "model": config.LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 300},  # cap output length to avoid runaway generation
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "answer": data.get("response", "").strip(),
        "cited_sources": [c["source"] for c in retrieved_chunks],
        "chunk_count": len(retrieved_chunks),
        "latency_ms": latency_ms,
        # Ollama reports these natively - real token usage, not estimated
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }
