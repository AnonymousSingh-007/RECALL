"""⚠️ DEMO/TEST ONLY — builds indices from a 10-passage hardcoded corpus.
Running this OVERWRITES the real 150K-passage indices at the same paths.
Real indexing: use build_strategy1_semantic.py through build_strategy4_metadata.py instead."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["HF_HOME"] = "D:\\hf_cache"

import json
import re
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_DIR = "indexing/indices"
CORPUS_PATH = "indexing/corpus_sample.jsonl"
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs("indexing", exist_ok=True)

# 1. Rich demonstration corpus with knowledge across science, history, tech, and edge cases
CORPUS_DATA = [
    {
        "query_id": 1,
        "passage_idx": 0,
        "is_selected": 1,
        "query_type": "description",
        "eng_query": "Who was Albert Einstein?",
        "text": "Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made critical contributions to quantum theory. His work is recognized for its major impact on the philosophy of science."
    },
    {
        "query_id": 1,
        "passage_idx": 1,
        "is_selected": 1,
        "query_type": "description",
        "eng_query": "Who was Albert Einstein?",
        "text": "Einstein received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect, a pivotal step in the development of quantum mechanics. His mass-energy equivalence formula E = mc^2 has been dubbed the world's most famous equation."
    },
    {
        "query_id": 1,
        "passage_idx": 2,
        "is_selected": 1,
        "query_type": "description",
        "eng_query": "Where did Einstein live?",
        "text": "After Adolf Hitler came to power in Germany in 1933, Einstein emigrated to the United States and settled in Princeton, New Jersey, where he accepted a position at the Institute for Advanced Study. He became an American citizen in 1940 and remained at Princeton until his death in 1955."
    },
    {
        "query_id": 2,
        "passage_idx": 0,
        "is_selected": 1,
        "query_type": "reasoning",
        "eng_query": "What was the immediate impact of the Manhattan Project?",
        "text": "The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. Led by the United States with support from the United Kingdom and Canada, it was directed by Major General Leslie Groves of the U.S. Army Corps of Engineers and physicist J. Robert Oppenheimer, head of the Los Alamos Laboratory."
    },
    {
        "query_id": 2,
        "passage_idx": 1,
        "is_selected": 1,
        "query_type": "reasoning",
        "eng_query": "What was the immediate impact of the Manhattan Project?",
        "text": "The immediate impact of the Manhattan Project was the development and deployment of atomic bombs over the Japanese cities of Hiroshima and Nagasaki in August 1945. This led directly to Japan's unconditional surrender, bringing World War II to an abrupt close and instantly ushering the globe into the nuclear age and the Cold War geopolitical standoff."
    },
    {
        "query_id": 3,
        "passage_idx": 0,
        "is_selected": 1,
        "query_type": "description",
        "eng_query": "What is quantum mechanics?",
        "text": "Quantum mechanics is a fundamental theory in physics that describes the behavior of nature at the scale of atoms and subatomic particles. It forms the basis for all modern physics, including quantum chemistry, quantum field theory, quantum technology, and quantum information science."
    },
    {
        "query_id": 4,
        "passage_idx": 0,
        "is_selected": 1,
        "query_type": "description",
        "eng_query": "What is the Transformer architecture in deep learning?",
        "text": "The Transformer is a deep learning architecture introduced in 2017 by Vaswani et al. in the paper 'Attention Is All You Need'. It is based entirely on self-attention mechanisms without recurrence or convolution, allowing models to process all sequence tokens simultaneously and enabling massive parallelism and scaling."
    },
    {
        "query_id": 5,
        "passage_idx": 0,
        "is_selected": 1,
        "query_type": "description",
        "eng_query": "What is the Artemis program?",
        "text": "NASA's Artemis program is a robotic and human Moon exploration program. Artemis aims to reestablish a human presence on the Moon for the first time since Apollo 17 in 1972, landing the first woman and first person of color on the lunar surface to lay the groundwork for long-term lunar exploration and eventual human missions to Mars."
    },
    {
        "query_id": 99,
        "passage_idx": 0,
        "is_selected": 0,
        "query_type": "fiction",
        "eng_query": "Who is Edward in Twilight?",
        "text": "Edward Cullen is a fictional vampire and the romantic protagonist of Stephenie Meyer's Twilight series. He possesses superhuman speed, strength, and telepathic mind-reading abilities, falling deeply in love with high school human Bella Swan in the rainy town of Forks, Washington."
    }
]

print("Writing corpus_sample.jsonl...")
with open(CORPUS_PATH, "w", encoding="utf-8") as f:
    for entry in CORPUS_DATA:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper to train IVFFlat index with sensible nlist
def build_ivf_index(embeddings: np.ndarray, nlist_target: int = 4):
    dim = embeddings.shape[1]
    nlist = max(1, min(nlist_target, len(embeddings)))
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    index.train(embeddings)
    index.add(embeddings)
    return index

# ----------------------------------------------------
# 1. Strategy 1: Semantic Chunks
# ----------------------------------------------------
print("Building Strategy 1: Semantic index...")
def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

s1_chunks = []
s1_meta = []
for rec in CORPUS_DATA:
    sentences = split_sentences(rec["text"])
    chunks = [" ".join(sentences[i:i+2]) for i in range(0, len(sentences), 2)] if len(sentences) > 1 else [rec["text"]]
    for c in chunks:
        s1_chunks.append(c)
        s1_meta.append({
            "query_id": rec["query_id"],
            "passage_idx": rec["passage_idx"],
            "is_selected": rec["is_selected"],
            "query_type": rec["query_type"],
            "text": c
        })

s1_vecs = model.encode(s1_chunks, convert_to_numpy=True).astype("float32")
s1_index = build_ivf_index(s1_vecs)
faiss.write_index(s1_index, f"{INDEX_DIR}/semantic.faiss")
with open(f"{INDEX_DIR}/semantic_meta.jsonl", "w", encoding="utf-8") as f:
    for m in s1_meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

# ----------------------------------------------------
# 2. Strategy 2: Sliding Window
# ----------------------------------------------------
print("Building Strategy 2: Sliding Window index...")
s2_chunks = []
s2_meta = []
for rec in CORPUS_DATA:
    words = rec["text"].split()
    size, overlap = 64, 16
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunk_text = " ".join(words[start:end])
        s2_chunks.append(chunk_text)
        s2_meta.append({
            "query_id": rec["query_id"],
            "passage_idx": rec["passage_idx"],
            "is_selected": rec["is_selected"],
            "query_type": rec["query_type"],
            "text": chunk_text
        })
        if end >= len(words):
            break
        start += size - overlap

s2_vecs = model.encode(s2_chunks, convert_to_numpy=True).astype("float32")
s2_index = build_ivf_index(s2_vecs)
faiss.write_index(s2_index, f"{INDEX_DIR}/sliding_window.faiss")
with open(f"{INDEX_DIR}/sliding_window_meta.jsonl", "w", encoding="utf-8") as f:
    for m in s2_meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

# ----------------------------------------------------
# 3. Strategy 3: Hierarchical Parent-Child
# ----------------------------------------------------
print("Building Strategy 3: Hierarchical index...")
s3_child_texts = []
s3_child_meta = []
s3_parent_lookup = []

for rec in CORPUS_DATA:
    parent_id = f"{rec['query_id']}_{rec['passage_idx']}_p0"
    sentences = split_sentences(rec["text"])
    s3_parent_lookup.append({
        "parent_id": parent_id,
        "parent_text": rec["text"],
        "query_id": rec["query_id"],
        "passage_idx": rec["passage_idx"],
        "is_selected": rec["is_selected"]
    })
    for s in sentences:
        s3_child_texts.append(s)
        s3_child_meta.append({
            "parent_id": parent_id,
            "child_text": s,
            "query_id": rec["query_id"],
            "passage_idx": rec["passage_idx"],
            "is_selected": rec["is_selected"]
        })

s3_vecs = model.encode(s3_child_texts, convert_to_numpy=True).astype("float32")
s3_index = build_ivf_index(s3_vecs)
faiss.write_index(s3_index, f"{INDEX_DIR}/hierarchical.faiss")
with open(f"{INDEX_DIR}/hierarchical_child_meta.jsonl", "w", encoding="utf-8") as f:
    for m in s3_child_meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
with open(f"{INDEX_DIR}/hierarchical_parent_lookup.jsonl", "w", encoding="utf-8") as f:
    for p in s3_parent_lookup:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

# ----------------------------------------------------
# 4. Strategy 4: Metadata-Aware (Whole Passages)
# ----------------------------------------------------
print("Building Strategy 4: Metadata-Aware index...")
s4_chunks = [rec["text"] for rec in CORPUS_DATA]
s4_meta = [{
    "query_id": rec["query_id"],
    "passage_idx": rec["passage_idx"],
    "is_selected": rec["is_selected"],
    "query_type": rec["query_type"],
    "language": "en",
    "text": rec["text"]
} for rec in CORPUS_DATA]

s4_vecs = model.encode(s4_chunks, convert_to_numpy=True).astype("float32")
s4_index = build_ivf_index(s4_vecs)
faiss.write_index(s4_index, f"{INDEX_DIR}/metadata_aware.faiss")
with open(f"{INDEX_DIR}/metadata_aware_meta.jsonl", "w", encoding="utf-8") as f:
    for m in s4_meta:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

# ----------------------------------------------------
# 5. BM25 Index
# ----------------------------------------------------
print("Building BM25 index...")
bm25_texts = [rec["text"] for rec in CORPUS_DATA]
bm25_meta = [{
    "query_id": rec["query_id"],
    "passage_idx": rec["passage_idx"],
    "is_selected": rec["is_selected"]
} for rec in CORPUS_DATA]

tokenized = [t.lower().split() for t in bm25_texts]
bm25 = BM25Okapi(tokenized)
with open(f"{INDEX_DIR}/bm25_index.pkl", "wb") as f:
    pickle.dump({"bm25": bm25, "texts": bm25_texts, "meta": bm25_meta}, f)

print("ALL INDICES BUILT SUCCESSFULLY!")