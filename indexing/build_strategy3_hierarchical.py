"""Build the Strategy 3 (hierarchical parent-child) FAISS index.

Parent chunks: 512 tokens, provide full context.
Child chunks: 64 tokens, get embedded and indexed for precise retrieval.
Retrieval hits a child, but returns its parent for generation context.
"""
import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "indexing/corpus_sample.jsonl"
INDEX_DIR = "indexing/indices"
os.makedirs(INDEX_DIR, exist_ok=True)

PARENT_SIZE = 512
CHILD_SIZE = 64


def make_parent_child_chunks(text: str, passage_meta: dict):
    """Split text into parent chunks, then each parent into child chunks."""
    tokens = text.split()
    parents = []
    parent_start = 0
    parent_id = 0

    while parent_start < len(tokens):
        parent_end = min(parent_start + PARENT_SIZE, len(tokens))
        parent_text = " ".join(tokens[parent_start:parent_end])

        children = []
        child_start = parent_start
        while child_start < parent_end:
            child_end = min(child_start + CHILD_SIZE, parent_end)
            child_text = " ".join(tokens[child_start:child_end])
            children.append(child_text)
            child_start = child_end

        parents.append({
            "parent_id": f"{passage_meta['query_id']}_{passage_meta['passage_idx']}_p{parent_id}",
            "parent_text": parent_text,
            "children": children,
        })
        parent_id += 1
        parent_start = parent_end

    return parents


print("Loading corpus...")
records = []
with open(CORPUS_PATH, encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))
print(f"Loaded {len(records)} passages")

print("Building parent-child chunks...")
child_texts = []
child_meta = []
parent_lookup = {}

for rec in records:
    parents = make_parent_child_chunks(rec["text"], rec)
    for parent in parents:
        parent_lookup[parent["parent_id"]] = {
            "parent_text": parent["parent_text"],
            "query_id": rec["query_id"],
            "passage_idx": rec["passage_idx"],
            "is_selected": rec["is_selected"],
            "query_type": rec["query_type"],
        }
        for child_text in parent["children"]:
            child_texts.append(child_text)
            child_meta.append({"parent_id": parent["parent_id"]})

print(f"Generated {len(child_texts)} child chunks across {len(parent_lookup)} parents")

print("Loading embedding model (GPU)...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

print("Embedding children...")
embeddings = model.encode(
    child_texts,
    batch_size=512,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

print("Building FAISS index...")
dim = embeddings.shape[1]
nlist = min(256, len(child_texts) // 40)
quantizer = faiss.IndexFlatL2(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist)
index.train(embeddings)
index.add(embeddings)

faiss.write_index(index, f"{INDEX_DIR}/hierarchical.faiss")

with open(f"{INDEX_DIR}/hierarchical_child_meta.jsonl", "w", encoding="utf-8") as f:
    for meta in child_meta:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

with open(f"{INDEX_DIR}/hierarchical_parent_lookup.jsonl", "w", encoding="utf-8") as f:
    for parent_id, data in parent_lookup.items():
        entry = {"parent_id": parent_id, **data}
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Saved index to {INDEX_DIR}/hierarchical.faiss")
print(f"Saved child metadata to {INDEX_DIR}/hierarchical_child_meta.jsonl")
print(f"Saved parent lookup to {INDEX_DIR}/hierarchical_parent_lookup.jsonl")
print(f"Total vectors in index: {index.ntotal}")