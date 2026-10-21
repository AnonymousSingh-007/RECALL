"""Isolate reranker timing from everything else."""
import time
import torch
from backend.reranker import load_reranker

model = load_reranker()
print("Model device:", model.model.device)

query = "How great of an impact did Oppenheimer have in project Manhattan?"
chunks = [
    "However, the man who is generally regarded as the father of the A-bomb is J. Robert Oppenheimer, who led the Manhattan Project.",
    "The history of the atomic bomb began with the Manhattan Project in 1939.",
    "The atomic fission bomb was invented by Leo Szilard in London in 1933.",
]
pairs = [[query, c] for c in chunks]

# warm-up call (first GPU call always has overhead)
_ = model.predict(pairs)

# timed call
start = time.perf_counter()
scores = model.predict(pairs)
elapsed = (time.perf_counter() - start) * 1000
print(f"Rerank time (warm): {elapsed:.1f}ms")
print("Scores:", scores)