"""TEMP STUB — no API key required. Returns top chunk directly.
Swap back to the real Anthropic call once billing is sorted.
"""


async def generate(query: str, chunks: list[dict]) -> str:
    if not chunks:
        return "I don't have sufficient context to answer this question."
    top = chunks[0]
    return f"[STUBBED ANSWER] Based on strategy '{top['strategy']}': {top['text'][:300]}"