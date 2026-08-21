"""Run once on Samratth's machine (RTX 5050). Builds all 4 FAISS indices + BM25."""
import yaml

with open("indexing/index_config.yaml") as f:
    config = yaml.safe_load(f)

# TODO Day 1-2:
# 1. Load MSMARCO-XI via datasets.load_dataset("ai4bharat/MSMARCO-XI")
# 2. For each strategy, chunk the corpus
# 3. Embed with sentence-transformers on CUDA
# 4. Build faiss.IndexIVFFlat, save to indexing/indices/{strategy}.faiss
# 5. Build BM25 corpus from sliding_window chunks, pickle to indexing/indices/bm25.pkl