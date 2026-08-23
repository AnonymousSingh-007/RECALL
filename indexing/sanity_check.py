"""Sanity check: query the sliding_window index and eyeball results."""
import json
import faiss
from sentence_transformers import SentenceTransformer

index = faiss.read_index("indexing/indices/sliding_window.faiss")
index.nprobe = 10  # search more clusters for better recall on IVF index

meta = []
with open("indexing/indices/sliding_window_meta.jsonl", encoding="utf-8") as f:
    for line in f:
        meta.append(json.loads(line))

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

query = "what was the immediate impact of the manhattan project"
q_vec = model.encode([query], convert_to_numpy=True).astype("float32")

scores, ids = index.search(q_vec, k=5)

print(f"Query: {query}\n")
for score, idx in zip(scores[0], ids[0]):
    print(f"score={score:.3f}  is_selected={meta[idx]['is_selected']}")
    print(f"  {meta[idx]['text'][:150]}...\n")