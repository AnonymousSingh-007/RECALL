from backend.schemas import RAGResponse, ChunkSource, LatencyBreakdown
from backend import stt, classifier, guardrails, generator
from backend.retriever import hybrid
from backend.reranker import rerank
from backend.latency import LatencyTracker


class RAGHarness:
    def __init__(self, faiss_store, bm25_store):
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store

    async def run(self, audio_bytes: bytes) -> RAGResponse:
        tracker = LatencyTracker()

        with tracker.track("stt"):
            stt_result = await stt.transcribe(audio_bytes)
            query = stt_result["transcript"]
            detected_language = stt_result.get("language_code", "unknown")

        is_safe, reason = guardrails.check_input(query)
        if not is_safe:
            raise ValueError(reason)

        with tracker.track("classify"):
            strategies = classifier.classify(query)

        with tracker.track("retrieve"):
            chunks = await hybrid.retrieve(query, self.faiss_store, self.bm25_store, strategies, k=10)

        top_faiss_score = chunks[0]["rrf_score"] if chunks else 0.0
        if not guardrails.check_relevance(top_faiss_score, top_faiss_score):
            raise ValueError("Not enough context in my knowledge base.")

        with tracker.track("rerank"):
            top_chunks = rerank(query, chunks, top_k=3)

        with tracker.track("generate"):
            answer_text = await generator.generate(query, [
                {"chunk_id": f"{c['query_id']}_{c['passage_idx']}", "strategy": c.get("strategy", "bm25"), "text": c["text"]}
                for c in top_chunks
            ])

        grounded = guardrails.check_grounding(answer_text, [c["text"] for c in top_chunks])

        return RAGResponse(
            answer=answer_text,
            sources=[
                ChunkSource(
                    chunk_id=f"{c['query_id']}_{c['passage_idx']}",
                    text=c["text"],
                    strategy=c.get("strategy", "bm25"),
                    score=c.get("rerank_score", c["rrf_score"]),
                )
                for c in top_chunks
            ],
            strategy_used=strategies,
            confidence=top_faiss_score,
            grounded=grounded,
            latency_breakdown=LatencyBreakdown(
                stt_ms=tracker.stages.get("stt", 0),
                classify_ms=tracker.stages.get("classify", 0),
                retrieve_ms=tracker.stages.get("retrieve", 0),
                rerank_ms=tracker.stages.get("rerank", 0),
                guardrail_ms=0,
                generate_ms=tracker.stages.get("generate", 0),
                total_ms=tracker.total_ms,
            ),
            query_language=detected_language,
            query_text=query,
        )