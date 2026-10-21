"""Build the Strategy 2 (sliding window) FAISS index from the sampled corpus.

This is the first end-to-end index: chunk -> embed -> FAISS -> save.
Proves the pipeline before scaling to all 4 strategies.
"""
import json
import sys
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.chunker.sliding_window import chunk_sliding_window

CORPUS_PATH = "indexing/corpus_sample.jsonl"
INDEX_DIR = "indexing/indices"
os.makedirs(INDEX_DIR, exist_ok=True)

print("Loading corpus...")
records = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
print(f"Loaded {len(records)} passages")

print("Chunking...")
chunks = []
chunk_meta = []
for rec in records:
    sub_chunks = chunk_sliding_window(rec["text"], size=256, overlap=64)
    for sc in sub_chunks:
        chunks.append(sc["text"])
        chunk_meta.append({
            "query_id": rec["query_id"],
            "passage_idx": rec["passage_idx"],
            "is_selected": rec["is_selected"],
            "query_type": rec["query_type"],
            "start_token": sc["start_token"],
            "end_token": sc["end_token"],
        })
print(f"Generated {len(chunks)} chunks")

print("Loading embedding model (GPU)...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

print("Embedding...")
embeddings = model.encode(
    chunks,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True,
)
embeddings = embeddings.astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

print("Building FAISS index...")
dim = embeddings.shape[1]
nlist = min(256, len(chunks) // 40)  # avoid nlist > training points
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.train(embeddings)
index.add(embeddings)

faiss.write_index(index, f"{INDEX_DIR}/sliding_window.faiss")

with open(f"{INDEX_DIR}/sliding_window_meta.jsonl", "w", encoding="utf-8") as f:
    for chunk_text, meta in zip(chunks, chunk_meta):
        entry = {**meta, "text": chunk_text}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Saved index to {INDEX_DIR}/sliding_window.faiss")
print(f"Saved metadata to {INDEX_DIR}/sliding_window_meta.jsonl")
print(f"Total vectors in index: {index.ntotal}")