"""Quick test: does reranking actually reorder results sensibly?"""
import asyncio
from backend.retriever.faiss_store import FaissStore
from backend.retriever.bm25_store import BM25Store
from backend.retriever.hybrid import retrieve
from backend.classifier import classify
from backend.reranker import rerank


async def main():
    faiss_store = FaissStore()
    faiss_store.load()
    bm25_store = BM25Store("indexing/indices/bm25_index.pkl")
    bm25_store.load()

    query = "what was the immediate impact of the manhattan project"
    strategies = classify(query)
    chunks = await retrieve(query, faiss_store, bm25_store, strategies, k=10)

    print("Before rerank (top 3 by RRF):")
    for c in chunks[:3]:
        print(f"  rrf={c['rrf_score']:.4f}  {c['text'][:100]}...")

    reranked = rerank(query, chunks, top_k=3)
    print("\nAfter rerank (top 3 by cross-encoder):")
    for c in reranked:
        print(f"  rerank_score={c['rerank_score']:.4f}  {c['text'][:100]}...")


if __name__ == "__main__":
    asyncio.run(main())