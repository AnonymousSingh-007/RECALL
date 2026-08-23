# RECALL

Voice-enabled RAG pipeline — HH Goa 2026, Task 2.

## Architecture

Voice → Sarvam STT → Query Classifier → Chunking Strategy Router →
Hybrid Retrieval (FAISS + BM25, parallel) → RRF Fusion → Cross-Encoder Rerank →
Guardrails → Claude Haiku Generation → Structured Output

Four chunking strategies, selected at query time rather than applied uniformly:
- **Semantic** — splits at embedding-similarity shifts
- **Sliding window** — fixed 256-token chunks, 64-token overlap
- **Hierarchical** — parent-child chunks (512/64 tokens), retrieves child, returns parent context
- **Metadata-aware** — respects passage boundaries, filters by language/query_type

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

Blackwell-series GPUs (RTX 50xx) need a nightly PyTorch build with CUDA 12.8 support — the stable/cu121/cu124 builds don't recognize `sm_120` yet:

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
python indexing/inspect_dataset.py       # confirm schema
python indexing/build_corpus.py          # flatten full dataset -> corpus_english.jsonl
python indexing/sample_corpus.py         # stratified sample -> corpus_sample.jsonl
python indexing/build_strategy2_index.py # build sliding-window FAISS index
\`\`\`

Corpus and index files are gitignored (too large for GitHub). Share via Drive if the other collaborator needs them directly rather than rebuilding.

## Team

- **Samratth Singh** — retrieval, chunking strategies, FAISS indexing, GPU embedding pipeline
- **Snehal** — Sarvam STT integration, harness orchestration, guardrails, frontend, benchmarking

## Status

🚧 Day 1–2 — corpus built and sampled, first FAISS index (sliding window) in progress. STT pipeline confirmed working end-to-end.