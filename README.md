# Cost-Efficient RAG Application

A QA service over a document corpus, backed by FAISS (free, self-hosted)
instead of a managed vector DB, with full retrieval/answer/cost evaluation.

## Vector store chosen: FAISS

Managed vector DBs (Pinecone, etc.) bill for always-on pods regardless of
query volume, which is exactly the cost problem this assignment describes.
FAISS runs embedded in the app process, has zero licensing/storage cost,
and comfortably handles millions of vectors in RAM on a small VM. The
trade-off: no built-in server, no automatic replication/sharding, and you
manage persistence yourself (handled here via `faiss.write_index` /
`read_index` to disk). For a QA service that's read-heavy and doesn't need
multi-region distribution, that trade-off is worth it.

## Prerequisites

- Python 3.10+
- The model pulled once: `ollama pull qwen2:0.5b`
- ~1GB free RAM for the embedding model + qwen2:0.5b (chosen over larger
  models like Mistral because it runs far faster on CPU-only machines)


LLM choice: qwen2:0.5b via Ollama — a small (352MB) model that runs fast on CPU-only hardware. Larger local models (e.g. Mistral 7B) produced multi-minute response times without a GPU, which isn't practical for an interactive QA service.

## Install (under 10 minutes)

```bash
git clone <your-repo-url>
cd rag-app
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # defaults are fine as-is
```

## Environment variables

All in `.env.example` — no secrets required since Ollama is local and free.
Key ones: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `EMBEDDING_MODEL`, `LLM_MODEL`,
`DEFAULT_TOP_K`, `MIN_SIMILARITY` (the no-relevant-context cutoff).

## Ingest a corpus

Drop your PDF/HTML/MD files into `data/corpus/`, then:

```bash
python -m app.ingest
```

Re-running this on an unchanged corpus adds 0 new vectors (idempotent —
each chunk is hashed on `source + text`, and already-seen hashes are skipped).

## Run the service

```bash
uvicorn app.main:app --reload --port 8000
```

Then query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is database normalization?", "top_k": 5}'
```

Or trigger ingestion via the API instead of the CLI:

```bash
curl -X POST http://localhost:8000/ingest
```

## Evaluation

1. Edit `eval/questions.json` — replace the 2 example questions with your
   own **15–30 questions** based on your actual corpus. Each needs:
   - `relevant_sources`: which file(s) in your corpus actually answer it
   - `gold_answer` (optional): enables EM/F1 scoring
2. Run:
   ```bash
   python -m eval.run_eval
   ```
3. Results land in `eval/results.json` — per-question metrics plus
   aggregated Recall@k, Hit Rate, MRR, nDCG@k, context precision,
   faithfulness/relevance (LLM-judged via Ollama), F1/EM, and p50/p95 latency.

## Cost comparison

```bash
python -m eval.cost_analysis
```

Compares FAISS (self-hosted VM, sized by RAM needed) against a managed,
pod-based vector DB at 100K / 1M / 10M vectors. Assumptions (pod capacity,
VM pricing tiers) are documented in the module's docstring — these are
illustrative to show the shape of the cost curve, not vendor quotes.



**When would you switch back to a managed DB?**
Once you need multi-region low-latency replication, automatic scaling
across many concurrent write-heavy clients, or a team without infra
capacity to manage a VM/backups — the operational cost of self-hosting
starts to outweigh the dollar savings shown above.



