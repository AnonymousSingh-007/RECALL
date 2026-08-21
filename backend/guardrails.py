BLOCKLIST = {"kill", "bomb", "weapon", "hack password"}


def check_input(query: str) -> None:
    lower = query.lower()
    if any(term in lower for term in BLOCKLIST):
        raise ValueError("blocked: unsafe input")


def check_relevance(faiss_score: float, bm25_score: float) -> bool:
    return faiss_score >= 0.35 or bm25_score >= 0.20


def check_grounding(answer: str, chunks: list[str]) -> bool:
    # TODO: replace with spaCy noun-phrase overlap
    context = " ".join(chunks).lower()
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words:
        return True
    overlap = sum(1 for w in words if w in context) / len(words)
    return overlap >= 0.3