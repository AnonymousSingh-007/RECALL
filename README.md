# RECALL

Voice-enabled RAG pipeline — HH Goa 2026, Task 2.

## Architecture

Voice → Sarvam STT → Query Classifier → Chunking Strategy Router →
Hybrid Retrieval (FAISS + BM25, parallel) → RRF Fusion → Cross-Encoder Rerank →
Guardrails → Claude Haiku Generation → Structured Output

## Setup

\`\`\`bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
uvicorn main:app --reload
\`\`\`

## Team

Samratth Singh — retrieval, indexing, GPU pipeline
Snehal — STT, harness, guardrails, frontend

## Status

🚧 Day 1 — skeleton in place, wiring in progress.