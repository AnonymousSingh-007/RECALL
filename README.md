# RECALL 🎙️

> Voice-enabled RAG pipeline — built for **HH Goa 2026, Task 2**

Ask it a question out loud or type it. It retrieves across 150K+ passages using four different chunking strategies, reranks, checks its own answer is actually grounded, and refuses rather than makes things up.

---

## ✨ Highlights

| | |
|---|---|
| 🧩 **4 chunking strategies** | Semantic · Sliding Window · Hierarchical · Metadata-aware — picked per query |
| 🔀 **Hybrid retrieval** | FAISS (dense) + BM25 (lexical), fused with Reciprocal Rank Fusion |
| 🎯 **Cross-encoder rerank** | GPU-accelerated, re-scores top candidates before generation |
| 🛡️ **4-layer guardrails** | Input safety → relevance gate → language check → grounding check |
| 🗣️ **Voice-native** | Sarvam STT in, Groq LLM out, sub-second retrieval |
| 📚 **150K+ passages** | MSMARCO-XI (bilingual) + Sherlock Holmes + Merchant of Venice |

---

## 🧭 How a query flows

```
🎤 Voice / ⌨️ Text
        │
        ▼
  Sarvam STT
        │
        ▼
  Guardrails (safety + language)
        │
        ▼
  Query Classifier → Strategy Router
        │
        ▼
  Hybrid Retrieval  (FAISS ∥ BM25)
        │
        ▼
  RRF Fusion
        │
        ▼
  Cross-Encoder Rerank
        │
        ▼
  Guardrails (relevance + grounding)
        │
        ▼
  Groq Generation
        │
        ▼
  📋 Structured Answer
```

---

## 🧩 Chunking strategies

<details>
<summary><b>Semantic</b> — splits where meaning shifts</summary>

Splits text at embedding-similarity drops between consecutive sentences, rather than at a fixed length. Produces the most "meaning-shaped" chunks.
</details>

<details>
<summary><b>Sliding window</b> — fixed-size, overlapping</summary>

256-token chunks, 64-token overlap. The reliable baseline.
</details>

<details>
<summary><b>Hierarchical</b> — parent/child</summary>

Retrieves on small 64-token child chunks for precision, returns the full 512-token parent for context.
</details>

<details>
<summary><b>Metadata-aware</b> — whole passage</summary>

No sub-splitting — keeps `query_type`, `is_selected`, and language tags attached to the full passage.
</details>

---

## 📊 Dataset

**[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)** — bilingual (Hindi/English) MS MARCO translation, plus two public-domain literary works via Project Gutenberg (*The Adventures of Sherlock Holmes*, *The Merchant of Venice*) merged in to stress-test retrieval across factual and narrative text.

**~150K passages** sampled and flattened into `indexing/corpus_sample.jsonl` *(gitignored — regenerate locally, see below)*.

---

## 🛡️ Guardrails

| Layer | Checks |
|---|---|
| 1 | Input safety (blocklist) |
| 2 | Relevance gate (RRF score threshold) |
| 3 | Language mismatch |
| 4 | Grounding (spaCy noun-phrase overlap vs. retrieved context) |

If retrieval comes back irrelevant, RECALL says so instead of confidently making something up — verified against adversarial and out-of-corpus queries.

---

## 🚀 Setup

### Common (both machines)

```bash
py -3.11 -m venv venv
venv\Scripts\Activate.ps1        # Windows
source venv/bin/activate         # macOS/Linux

pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env             # fill in SARVAM_API_KEY, GROQ_API_KEY
```

### GPU machine only (indexing / embeddings / reranking)

Blackwell GPUs (RTX 50-series) need a nightly PyTorch build with CUDA 12.8 support:

```bash
pip uninstall torch -y
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

Verify:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Run the server

```bash
uvicorn main:app --reload
```

Indices load once at startup. Check `/health` → `harness_ready: true` before querying.

---

## 🏗️ Building the corpus + indices (GPU machine)

```bash
python indexing/inspect_dataset.py
python indexing/build_corpus.py
python indexing/sample_corpus.py
python indexing/add_literary_corpus.py
python indexing/build_strategy2_index.py
python indexing/build_strategy3_hierarchical.py
python indexing/build_strategy1_semantic.py
python indexing/build_strategy4_metadata.py
python -m backend.retriever.bm25_store
```

> ⚠️ `indexing/build_all_indices.py` is a **demo-only** script with a hardcoded 10-passage corpus. Do not run it against the real indices — same output paths.

All index/corpus files are gitignored — share via Drive or regenerate locally.

---

## 🧠 Generation

Groq (`openai/gpt-oss-20b`) — OpenAI-compatible API, free tier.

---

## 👥 Team

| | |
|---|---|
| **Samratth Singh** | Retrieval, chunking strategies, FAISS indexing, GPU embedding pipeline, harness, reranker, corpus curation |
| **Snehal** | Sarvam STT, guardrails L3/L4, frontend (glassmorphism UI, 3D orb), benchmarking |

---

## ✅ Status

- [x] All 4 chunking strategies built and indexed
- [x] BM25 index built
- [x] Guardrails L1–L4 live and tested
- [x] Full harness wired end-to-end — voice and text in, grounded answer out
- [x] Real generation via Groq
- [x] Frontend complete — glassmorphism, 3D orb, strategy/confidence/latency display
- [x] Literary corpus added and verified

---

<p align="center"><sub>Built for HH Goa 2026 — Task 2</sub></p>