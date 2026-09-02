# RECALL

Voice-enabled RAG pipeline — HH Goa 2026, Task 2.

## Architecture

Voice → Sarvam STT → Query Classifier → Chunking Strategy Router →
Hybrid Retrieval (FAISS + BM25, parallel) → RRF Fusion → Cross-Encoder Rerank →
Guardrails → Claude Haiku Generation → Structured Output

Four chunking strategies, selected at query time rather than applied uniformly:
- **Semantic** — splits at embedding-similarity shifts between sentences
- **Sliding window** — fixed 256-token chunks, 64-token overlap
- **Hierarchical** — parent-child chunks (512/64 tokens), retrieves child, returns parent context
- **Metadata-aware** — whole-passage chunks, tagged with query_type/is_selected/language for filtered retrieval

## Dataset

[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) — bilingual (Hindi/English) MS MARCO translation dataset. Each row contains an English query (`Eng_Query`), 10 candidate English passages (`passages.English_passages`), and gold relevance labels (`passages.is_selected`).

Working corpus: a stratified 15,000-query sample (~150K passages), flattened and saved locally as `indexing/corpus_sample.jsonl` (gitignored — regenerate via `indexing/sample_corpus.py`).

## Setup

### Common (both machines)

\`\`\`bash
py -3.11 -m venv venv
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate         # macOS/Linux

pip install -r requirements.txt
cp .env.example .env             # fill in SARVAM_API_KEY and ANTHROPIC_API_KEY
\`\`\`

### GPU machine only (indexing, embeddings)

Blackwell-series GPUs (RTX 50xx) need a nightly PyTorch build with CUDA 12.8 support — stable/cu121/cu124 builds don't recognize `sm_120`:

\`\`\`bash
pip uninstall torch -y
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
\`\`\`

Verify:
\`\`\`bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
\`\`\`
Should print `True` with no `sm_120 is not compatible` warning.

### Run the server

\`\`\`bash
uvicorn main:app --reload
\`\`\`

## Building the corpus + indices (GPU machine)

\`\`\`bash
python indexing/inspect_dataset.py               # confirm schema
python indexing/build_corpus.py                  # flatten full dataset -> corpus_english.jsonl
python indexing/sample_corpus.py                 # stratified sample -> corpus_sample.jsonl
python indexing/build_strategy2_index.py         # sliding window FAISS index
python indexing/build_strategy3_hierarchical.py  # hierarchical parent-child FAISS index
python indexing/build_strategy1_semantic.py      # semantic FAISS index
python indexing/build_strategy4_metadata.py      # metadata-aware FAISS index
\`\`\`

Corpus and index files are gitignored (too large for GitHub). Shared via Drive between collaborators rather than committed.

### Index sizes (15K query / ~150K passage sample)

| Strategy | Chunks | Notes |
|---|---|---|
| Sliding window | 149,793 | 256 tokens, 64 overlap |
| Hierarchical | 189,910 children | across 149,784 parents |
| Semantic | 298,409 | variable length, similarity-split |
| Metadata-aware | 149,784 | whole passages, no sub-splitting |

## BM25

\`\`\`bash
python -m backend.retriever.bm25_store   # builds indexing/indices/bm25_index.pkl
\`\`\`

## Team

- **Samratth Singh** — retrieval, chunking strategies, FAISS indexing, GPU embedding pipeline
- **Snehal** — Sarvam STT integration, harness orchestration, guardrails, frontend, benchmarking

## Status

✅ All 4 chunking strategies built and indexed
✅ BM25 index built
✅ Guardrails Layers 1 & 2 live
✅ STT confirmed working end-to-end
🚧 Next: wire `faiss_store.py` to load all 4 indices, finish `hybrid.py` RRF fusion, full harness integration