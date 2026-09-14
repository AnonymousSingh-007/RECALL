"""First full end-to-end harness test: audio in, answer out."""
import asyncio
from backend.harness import RAGHarness
from backend.retriever.faiss_store import FaissStore
from backend.retriever.bm25_store import BM25Store


async def main():
    faiss_store = FaissStore()
    faiss_store.load()

    bm25_store = BM25Store("indexing/indices/bm25_index.pkl")
    bm25_store.load()

    harness = RAGHarness(faiss_store, bm25_store)

    with open("path/to/your/test_audio.mp3", "rb") as f:
        audio_bytes = f.read()

    response = await harness.run(audio_bytes)

    print("Answer:", response.answer)
    print("\nStrategy used:", response.strategy_used)
    print("Grounded:", response.grounded)
    print("Confidence:", response.confidence)
    print("\nLatency breakdown:")
    print(f"  STT: {response.latency_breakdown.stt_ms:.1f}ms")
    print(f"  Classify: {response.latency_breakdown.classify_ms:.1f}ms")
    print(f"  Retrieve: {response.latency_breakdown.retrieve_ms:.1f}ms")
    print(f"  Rerank: {response.latency_breakdown.rerank_ms:.1f}ms")
    print(f"  Generate: {response.latency_breakdown.generate_ms:.1f}ms")
    print(f"  TOTAL: {response.latency_breakdown.total_ms:.1f}ms")

    print("\nSources:")
    for s in response.sources:
        print(f"  [{s.strategy}] score={s.score:.3f} {s.text[:100]}...")


if __name__ == "__main__":
    asyncio.run(main())