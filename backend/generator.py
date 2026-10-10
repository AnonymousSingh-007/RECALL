"""Answer generator — Claude Haiku with grounding instructions.

Uses Anthropic's Claude 3 Haiku for fast, grounded answers.
Falls back to a stub if ANTHROPIC_API_KEY is not set (free-tier dev mode).

Designed to be swapped to Groq's free tier later — the prompt template
stays the same, only the client call changes.
"""

import os

_client = None
_USE_STUB = False

# Model to use — claude-3-haiku for speed + cost
MODEL = "claude-3-haiku-20240307"
MAX_TOKENS = 300


def _get_client():
    """Lazy-init the Anthropic client. Falls back to stub if no API key."""
    global _client, _USE_STUB

    if _client is not None:
        return _client

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        _USE_STUB = True
        return None

    try:
        import anthropic
        _client = anthropic.AsyncAnthropic(api_key=api_key)
        return _client
    except ImportError:
        _USE_STUB = True
        return None


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are RECALL, a precise question-answering assistant. Your answers MUST be grounded in the provided context passages.

Rules:
1. ONLY use information from the provided context passages to answer.
2. If the context does not contain enough information, say: "I don't have sufficient context to answer this question."
3. Keep answers concise — 1-3 sentences for factual questions, up to a short paragraph for explanations.
4. Do NOT speculate, infer, or add information beyond what the passages state.
5. If multiple passages provide relevant information, synthesize them into a coherent answer.
6. Cite which passage(s) you drew from when possible (e.g., "According to the retrieved context...")."""


def _build_user_prompt(query: str, chunks: list[dict]) -> str:
    """Build the user prompt with query and numbered context passages."""
    passages = []
    for i, chunk in enumerate(chunks, 1):
        strategy = chunk.get("strategy", "unknown")
        text = chunk.get("text", "").strip()
        passages.append(f"[Passage {i} — strategy: {strategy}]\n{text}")

    context_block = "\n\n".join(passages)

    return f"""Context passages:
{context_block}

Question: {query}

Answer based ONLY on the context above:"""


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

async def generate(query: str, chunks: list[dict]) -> str:
    """Generate an answer from the query and retrieved chunks.

    Args:
        query: The user's question (text, post-STT).
        chunks: List of dicts with keys: chunk_id, strategy, text.

    Returns:
        The generated answer string.
    """
    if not chunks:
        return "I don't have sufficient context to answer this question."

    client = _get_client()

    if _USE_STUB or client is None:
        # Stub mode — return top chunk directly (no API cost)
        top = chunks[0]
        return (
            f"[STUB — no API key] Based on strategy '{top['strategy']}': "
            f"{top['text'][:300]}"
        )

    # Real Anthropic call
    user_prompt = _build_user_prompt(query, chunks)

    try:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text from response
        answer = response.content[0].text.strip()
        return answer

    except Exception as exc:
        # Degrade gracefully — return stub answer rather than crashing
        top = chunks[0]
        return (
            f"[Generation failed: {exc}] "
            f"Top passage (strategy: {top['strategy']}): {top['text'][:300]}"
        )