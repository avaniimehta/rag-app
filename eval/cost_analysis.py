"""
Cost comparison: self-hosted FAISS vs a managed vector DB, at three scales.

ASSUMPTIONS (stated explicitly, as the assignment requires):
 - Managed DB modeled as pod-based (e.g. Pinecone p1 pods), which is the
   "always-on pods" pricing the assignment describes: each pod is billed
   whether or not it's queried, and has a fixed vector capacity. We assume
   ~1M vectors of this dimensionality (384-dim) fit per pod, at ~$70/pod/month
   (illustrative, in line with publicly listed starter pod pricing). Cost
   scales in STEPS as you add pods to hold more vectors - this is the actual
   cost driver the assignment is pointing at, not a smooth per-GB rate.
 - Self-hosted FAISS: cost is just the VM needed to hold the index in RAM.
   Each vector = 384 dims * 4 bytes (float32) + ~500 bytes metadata (source
   path, chunk text, timestamps) = ~2KB/vector. VM sized to comfortably fit
   the index + OS + app overhead.
 - Both scale with vector count, but FAISS scales continuously with RAM
   (cheap, gradual) while managed scales in discrete pod jumps (expensive,
   step-function) - that difference is the actual point of this comparison.
 - These are illustrative numbers to show the SHAPE of the cost curve, not
   vendor-negotiated pricing - state this plainly in the README.
"""
import math

BYTES_PER_VECTOR = 1536 + 500  # embedding (384*4 bytes) + metadata, ~2KB

# Managed: pod-based, always-on, billed per pod regardless of query volume
VECTORS_PER_POD = 1_000_000
COST_PER_POD_MONTH = 70

# Self-hosted FAISS: cost is the VM, tiered by how much RAM is needed
FAISS_VM_TIERS = [
    (2, 15),    # up to 2GB RAM -> ~$15/month VM (t3.small)
    (4, 30),    # up to 4GB RAM -> ~$30/month VM (t3.medium)
    (16, 90),   # up to 16GB RAM -> ~$90/month VM (t3.xlarge)
    (32, 150),  # up to 32GB RAM -> ~$150/month VM (r6i.large-ish)
]


def _gb_needed(num_vectors: int) -> float:
    return (num_vectors * BYTES_PER_VECTOR) / (1024 ** 3)


def managed_cost(num_vectors: int) -> float:
    pods_needed = math.ceil(num_vectors / VECTORS_PER_POD)
    return pods_needed * COST_PER_POD_MONTH


def faiss_cost(num_vectors: int) -> float:
    gb = _gb_needed(num_vectors)
    for tier_gb, cost in FAISS_VM_TIERS:
        if gb <= tier_gb:
            return cost
    return FAISS_VM_TIERS[-1][1]


def build_cost_table():
    scales = [100_000, 1_000_000, 10_000_000]
    rows = []
    for n in scales:
        f_cost = faiss_cost(n)
        m_cost = managed_cost(n)
        rows.append({
            "num_vectors": n,
            "estimated_gb": round(_gb_needed(n), 3),
            "faiss_self_hosted_monthly_usd": f_cost,
            "managed_vector_db_monthly_usd": m_cost,
            "faiss_savings_pct": round((1 - f_cost / m_cost) * 100, 1),
        })
    return rows


if __name__ == "__main__":
    import json
    print(json.dumps(build_cost_table(), indent=2))
