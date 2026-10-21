import re

FACTOID_MAX_TOKENS = 8
REASONING_WORDS = {"how", "why", "explain", "describe", "compare"}


def classify(query: str) -> list[str]:
    tokens = query.strip().split()
    lower = query.lower()

    non_ascii = bool(re.search(r"[^\x00-\x7F]", query))
    if non_ascii:
        return ["metadata_aware"]

    if len(tokens) <= FACTOID_MAX_TOKENS:
        return ["sliding_window"]

    if any(w in lower for w in REASONING_WORDS):
        return ["semantic", "hierarchical"]

    return ["semantic", "sliding_window", "hierarchical", "metadata_aware"]