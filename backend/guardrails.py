"""Guardrail layers. No dependency on retrieval/indexing work."""

BLOCKLIST = {
    "kill", "bomb", "weapon", "hack password",
    "suicide", "self harm", "make explosive",
}


def check_input(query: str) -> tuple[bool, str | None]:
    """Layer 1: input safety. Returns (is_safe, rejection_reason)."""
    lower = query.lower()
    for term in BLOCKLIST:
        if term in lower:
            return False, "I can't help with that request."
    return True, None


def check_relevance(top_score: float, threshold: float = 0.01) -> bool:
    """Layer 2: relevance gate. Works with RRF scores (0.001 - 0.15 range).
    Returns True if enough context exists to answer."""
    return top_score >= threshold


def check_grounding(answer: str, retrieved_chunks: list[str],
                     min_overlap: float = 0.3) -> bool:
    """Layer 4: word-overlap grounding check.
    Will be replaced with spaCy noun-phrase overlap later."""
    context = " ".join(retrieved_chunks).lower()
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words:
        return True
    overlap = sum(1 for w in words if w in context) / len(words)
    return overlap >= min_overlap