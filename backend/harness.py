from backend.schemas import RAGResponse
from backend import stt, classifier, guardrails, generator
from backend.retriever import hybrid


class RAGHarness:
    async def run(self, audio_bytes: bytes) -> RAGResponse:
        query = await stt.transcribe(audio_bytes)

        is_safe, reason = guardrails.check_input(query)
        if not is_safe:
            raise ValueError(reason)

        strategy = classifier.classify(query)
        chunks = await hybrid.retrieve(query, None, None, strategy)  # stores wired next
        # TODO: rerank, grounding check, generate, output guardrail
        raise NotImplementedError