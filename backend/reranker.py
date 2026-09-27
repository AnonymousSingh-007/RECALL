"""Cross-encoder reranker — reorders fused RRF results by actual relevance."""
import torch
from sentence_transformers import CrossEncoder

_model: CrossEncoder | None = None


def load_reranker():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-2-v2", device=device)
    return _model


def rerank(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    """Score each chunk against the query with a cross-encoder, return top_k."""
    if not chunks:
        return []

    model = load_reranker()
    pairs = [[query, c["text"]] for c in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]