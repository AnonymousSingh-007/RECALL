import os
from anthropic import AsyncAnthropic

client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a precise, grounded question-answering assistant.
Answer ONLY using the provided context chunks. If the context does not
contain enough information to answer, say exactly:
'I don't have sufficient context to answer this question.'
Never invent facts. Cite the source chunk ID in your answer."""


async def generate(query: str, chunks: list[dict]) -> str:
    context = "\n".join(
        f"[{c['chunk_id']}] (strategy: {c['strategy']}) {c['text']}"
        for c in chunks
    )
    msg = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
    )
    return msg.content[0].text