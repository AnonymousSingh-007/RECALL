from backend.schemas import RAGResponse
from backend import stt, classifier, guardrails, generator
from backend.retriever import hybrid


class RAGHarness:
    async def run(self, audio_bytes: bytes) -> RAGResponse:
        # TODO Day 1-2: wire stages in order
        # query = await stt.transcribe(audio_bytes)
        # guardrails.check_input(query)
        # strategy = classifier.classify(query)
        # chunks = await hybrid.retrieve(query, strategy)
        # chunks = rerank(query, chunks)
        # guardrails.check_grounding(query, chunks)
        # answer = await generator.generate(query, chunks)
        # guardrails.check_output(answer, chunks)
        raise NotImplementedError