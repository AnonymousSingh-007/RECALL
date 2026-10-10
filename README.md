# RECALL

Voice-enabled RAG pipeline — HH Goa 2026, Task 2.

## Architecture

Voice → Sarvam STT → Guardrails (input safety + language check) → Query Classifier →
Chunking Strategy Router → Hybrid Retrieval (FAISS + BM25, parallel) → RRF Fusion →
Cross-Encoder Rerank → Guardrails (relevance + grounding) → LLM Generation → Structured Output

Four chunking strategies, selected at query time rather than applied uniformly:
- **Semantic** — splits at embedding-similarity shifts between sentences
- **Sliding window** — fixed 256-token chunks, 64-token overlap
- **Hierarchical** — parent-child chunks (512/64 tokens), retrieves child, returns parent context
- **Metadata-aware** — whole-passage chunks, tagged with query_type/is_selected/language

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
python -m spacy download en_core_web_sm
cp .env.example .env             # fill in SARVAM_API_KEY, GROQ_API_KEY
\`\`\`

### GPU machine only (indexing, embeddings, reranking)

Blackwell-series GPUs (RTX 50xx) need a nightly PyTorch build with CUDA 12.8 support:

\`\`\`bash
pip uninstall torch -y
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
\`\`\`

Verify:
\`\`\`bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
\`\`\`

### Run the server

\`\`\`bash
uvicorn main:app --reload
\`\`\`
Indices load once at startup (printed to console: "RECALL: all indices loaded, harness ready."). Check `/health` for `harness_ready: true` before testing queries.

## Building the corpus + indices (GPU machine)

\`\`\`bash
python indexing/inspect_dataset.py
python indexing/build_corpus.py
python indexing/sample_corpus.py
python indexing/build_strategy2_index.py
python indexing/build_strategy3_hierarchical.py
python indexing/build_strategy1_semantic.py
python indexing/build_strategy4_metadata.py
python -m backend.retriever.bm25_store
\`\`\`

All index/corpus files are gitignored (too large for GitHub) — share via Drive, rebuild locally, or regenerate from scratch.

### Index sizes (15K query / ~150K passage sample)

| Strategy | Chunks |
|---|---|
| Sliding window | 149,793 |
| Hierarchical | 189,910 children / 149,784 parents |
| Semantic | 298,409 |
| Metadata-aware | 149,784 |
| BM25 | 149,784 |

## Guardrails

- **Layer 1** — input safety (blocklist)
- **Layer 2** — relevance gate (RRF score threshold)
- **Layer 3** — language mismatch detection
- **Layer 4** — grounding check (spaCy noun-phrase overlap)

## Generation

Currently wired for Anthropic Claude; swapping to Groq's free tier (Llama 3.1 8B) to avoid API costs during testing.

## Team

- **Samratth Singh** — retrieval, chunking strategies, FAISS indexing, GPU embedding pipeline, harness wiring, reranker
- **Snehal** — Sarvam STT integration, guardrails Layer 3/4, frontend, benchmark script

## Status

✅ All 4 chunking strategies built and indexed
✅ BM25 index built
✅ Guardrails Layers 1–4 live and tested
✅ Full harness wired end-to-end — voice in, answer out, confirmed working
✅ Frontend built (mic input, strategy badges, latency breakdown)
✅ Benchmark script built
🚧 Next: swap generator to Groq (free tier), real latency benchmarking on warm server, frontend polish