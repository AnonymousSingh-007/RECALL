"""Build the Strategy 4 (metadata-aware) FAISS index.

Respects passage boundaries entirely — no sub-chunking, one embedding
per whole passage. All available metadata is attached for filtered retrieval
(query_type, is_selected, language).
"""
import json
import os
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "indexing/corpus_sample.jsonl"
INDEX_DIR = "indexing/indices"
os.makedirs(INDEX_DIR, exist_ok=True)

print("Loading corpus...")
records = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
print(f"Loaded {len(records)} passages")

print("Preparing chunks (whole passages, boundary-respecting)...")
chunks = []
chunk_meta = []

for rec in records:
    chunks.append(rec["text"])
    chunk_meta.append({
        "query_id": rec["query_id"],
        "passage_idx": rec["passage_idx"],
        "is_selected": rec["is_selected"],
        "query_type": rec["query_type"],
        "language": "en",  # English_passages field — extend later for Hindi if needed
    })

print(f"Total passages (no sub-splitting): {len(chunks)}")

print("Loading embedding model (GPU)...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

print("Embedding...")
embeddings = model.encode(
    chunks,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

print("Building FAISS index...")
dim = embeddings.shape[1]
nlist = min(256, len(chunks) // 40)
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.train(embeddings)
index.add(embeddings)

faiss.write_index(index, f"{INDEX_DIR}/metadata_aware.faiss")

with open(f"{INDEX_DIR}/metadata_aware_meta.jsonl", "w", encoding="utf-8") as f:
    for chunk_text, meta in zip(chunks, chunk_meta):
        entry = {**meta, "text": chunk_text}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Saved index to {INDEX_DIR}/metadata_aware.faiss")
print(f"Saved metadata to {INDEX_DIR}/metadata_aware_meta.jsonl")
print(f"Total vectors in index: {index.ntotal}")