"""Generation via Groq's free API (OpenAI-compatible format)."""
import os
import httpx

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """You are a precise, grounded question-answering assistant.
Answer ONLY using the provided context chunks. If the context does not
contain enough information to answer, say exactly:
'I don't have sufficient context to answer this question.'
Never invent facts. Cite the source chunk ID in your answer."""


async def generate(query: str, chunks: list[dict]) -> str:
    if not GROQ_API_KEY:
        if chunks:
            return f"[NO API KEY] Top match: {chunks[0]['text'][:300]}"
        return "I don't have sufficient context to answer this question."

    context = "\n".join(
        f"[{c['chunk_id']}] (strategy: {c['strategy']}) {c['text']}"
        for c in chunks
    )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
                    ],
                    "max_tokens": 512,
                },
            )
            resp.raise_for_status()
            body = resp.json()
            return body["choices"][0]["message"]["content"]
    except httpx.HTTPError as exc:
        if chunks:
            return f"[Generation failed: {exc}] Top passage (strategy: {chunks[0]['strategy']}): {chunks[0]['text'][:300]}"
        return "I don't have sufficient context to answer this question."