"""Test hybrid retrieval end to end before wiring into the harness."""
import asyncio
from backend.retriever.faiss_store import FaissStore
from backend.retriever.bm25_store import BM25Store
from backend.retriever.hybrid import retrieve
from backend.classifier import classify


async def main():
    faiss_store = FaissStore()
    faiss_store.load()

    bm25_store = BM25Store("indexing/indices/bm25_index.pkl")
    bm25_store.load()

    query = "what was the immediate impact of the manhattan project"
    strategies = classify(query)
    print(f"Query: {query}")
    print(f"Classifier selected strategies: {strategies}\n")

    results = await retrieve(query, faiss_store, bm25_store, strategies, k=10)

    print(f"Fused results: {len(results)}\n")
    for r in results[:5]:
        print(f"rrf_score={r['rrf_score']:.4f}  strategy={r.get('strategy', 'bm25')}  is_selected={r['is_selected']}")
        print(f"  {r['text'][:150]}...\n")


if __name__ == "__main__":
    asyncio.run(main())