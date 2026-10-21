"""Build the Strategy 1 (semantic) FAISS index.

Splits text at semantic boundary shifts — where consecutive sentence
embeddings drop below a similarity threshold — instead of fixed token counts.
"""
import json
import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "indexing/corpus_sample.jsonl"
INDEX_DIR = "indexing/indices"
os.makedirs(INDEX_DIR, exist_ok=True)

SIMILARITY_THRESHOLD = 0.85
MIN_CHUNK_SENTENCES = 2


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter — good enough for MSMARCO-style passages."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]


def chunk_semantic(text: str, model: SentenceTransformer,
                    threshold: float = SIMILARITY_THRESHOLD) -> list[str]:
    """Group sentences into chunks, splitting where meaning shifts."""
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-8, None)

    chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = float(np.dot(normed[i - 1], normed[i]))
        if sim < threshold and len(current) >= MIN_CHUNK_SENTENCES:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(" ".join(current))

    return chunks


print("Loading corpus...")
records = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
print(f"Loaded {len(records)} passages")

print("Loading embedding model (GPU)...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

print("Chunking semantically (this pass embeds sentences per-passage)...")
chunks = []
chunk_meta = []

for i, rec in enumerate(records):
    sub_chunks = chunk_semantic(rec["text"], model)
    for sc in sub_chunks:
        chunks.append(sc)
        chunk_meta.append({
            "query_id": rec["query_id"],
            "passage_idx": rec["passage_idx"],
            "is_selected": rec["is_selected"],
            "query_type": rec["query_type"],
        })
    if (i + 1) % 10000 == 0:
        print(f"  processed {i + 1}/{len(records)} passages...")

print(f"Generated {len(chunks)} semantic chunks")

print("Embedding final chunks...")
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

faiss.write_index(index, f"{INDEX_DIR}/semantic.faiss")

with open(f"{INDEX_DIR}/semantic_meta.jsonl", "w", encoding="utf-8") as f:
    for chunk_text, meta in zip(chunks, chunk_meta):
        entry = {**meta, "text": chunk_text}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Saved index to {INDEX_DIR}/semantic.faiss")
print(f"Saved metadata to {INDEX_DIR}/semantic_meta.jsonl")
print(f"Total vectors in index: {index.ntotal}")