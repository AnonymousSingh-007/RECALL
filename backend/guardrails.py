"""Guardrail layers 1-4. No dependency on retrieval/indexing work.

Layer 1: Input safety — blocklist filter
Layer 2: Relevance gate — RRF score threshold
Layer 3: Language mismatch — reject non-English queries (corpus is English)
Layer 4: Grounding check — spaCy noun-phrase overlap between answer & context
"""

import re

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

BLOCKLIST = {
    "kill", "bomb", "weapon", "hack password",
    "suicide", "self harm", "make explosive",
}

# Regex patterns for non-Latin scripts that indicate a non-English query.
# Covers Devanagari (Hindi), CJK, Arabic, Cyrillic, Tamil, Telugu, Bengali,
# Kannada, Malayalam, Gurmukhi, Thai, Georgian, and Korean Hangul.
_NON_LATIN_RE = re.compile(
    r"[\u0900-\u097F"    # Devanagari (Hindi)
    r"\u4E00-\u9FFF"     # CJK Unified Ideographs
    r"\u0600-\u06FF"     # Arabic
    r"\u0400-\u04FF"     # Cyrillic
    r"\u0B80-\u0BFF"     # Tamil
    r"\u0C00-\u0C7F"     # Telugu
    r"\u0980-\u09FF"     # Bengali
    r"\u0C80-\u0CFF"     # Kannada
    r"\u0D00-\u0D7F"     # Malayalam
    r"\u0A00-\u0A7F"     # Gurmukhi (Punjabi)
    r"\u0E00-\u0E7F"     # Thai
    r"\u10A0-\u10FF"     # Georgian
    r"\uAC00-\uD7AF]"    # Korean Hangul
)

# Proportion of characters that must be non-Latin to trigger the gate.
_LANG_MISMATCH_RATIO = 0.30

# ---------------------------------------------------------------------------
# spaCy lazy loader for Layer 4
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    """Load spaCy English model lazily. Falls back to sm if md unavailable."""
    global _nlp
    if _nlp is None:
        import spacy  # type: ignore  # installed at runtime; IDE may not detect lazy import
        for model_name in ("en_core_web_md", "en_core_web_sm"):
            try:
                _nlp = spacy.load(model_name, disable=["ner", "parser", "textcat"])
                break
            except OSError:
                continue
        if _nlp is None:
            raise RuntimeError(
                "No spaCy English model found. Run: "
                "python -m spacy download en_core_web_sm"
            )
    return _nlp


# ---------------------------------------------------------------------------
# Layer 1: Input safety
# ---------------------------------------------------------------------------

def check_input(query: str) -> tuple[bool, str | None]:
    """Layer 1: input safety. Returns (is_safe, rejection_reason)."""
    lower = query.lower()
    for term in BLOCKLIST:
        if term in lower:
            return False, "I can't help with that request."
    return True, None


# ---------------------------------------------------------------------------
# Layer 2: Relevance gate
# ---------------------------------------------------------------------------

def check_relevance(top_score: float, threshold: float = 0.01) -> bool:
    """Layer 2: relevance gate. Works with RRF scores (0.001 - 0.15 range).
    Returns True if enough context exists to answer."""
    return top_score >= threshold


# ---------------------------------------------------------------------------
# Layer 3: Language mismatch detection
# ---------------------------------------------------------------------------

def check_language(query: str, corpus_language: str = "en") -> tuple[bool, str | None]:
    """Layer 3: language mismatch detection.

    Returns (is_compatible, rejection_reason).
    The corpus is English-only, so queries dominated by non-Latin script
    characters are flagged as mismatched.  Pure-ASCII queries (even if
    transliterated Hindi) pass through — retrieval will handle the
    relevance filtering via Layer 2.
    """
    if corpus_language != "en":
        # Future: support other corpus languages
        return True, None

    if not query or not query.strip():
        return True, None

    # Count non-Latin script characters vs total alpha characters
    alpha_chars = [c for c in query if c.isalpha()]
    if not alpha_chars:
        return True, None  # pure numbers / symbols — let retrieval handle it

    non_latin_count = len(_NON_LATIN_RE.findall(query))
    ratio = non_latin_count / len(alpha_chars)

    if ratio >= _LANG_MISMATCH_RATIO:
        return False, (
            "Your question appears to be in a non-English language. "
            "This knowledge base currently only supports English queries."
        )

    return True, None


# ---------------------------------------------------------------------------
# Layer 4: Grounding check (spaCy noun-phrase overlap)
# ---------------------------------------------------------------------------

def check_grounding(answer: str, retrieved_chunks: list[str],
                    min_overlap: float = 0.3) -> bool:
    """Layer 4: grounding check via spaCy noun-phrase overlap.

    Extracts noun phrases from the answer and checks what proportion
    appear in the retrieved context.  This replaces the crude word-overlap
    heuristic with linguistically meaningful units (proper nouns, compound
    nouns, key entities) that better indicate factual grounding.

    Falls back to the simpler word-overlap method if spaCy fails to load.
    """
    if not answer or not retrieved_chunks:
        return True

    context = " ".join(retrieved_chunks).lower()

    try:
        nlp = _get_nlp()

        # Extract noun phrases from the answer
        answer_doc = nlp(answer)
        noun_phrases = [
            chunk.text.lower().strip()
            for chunk in answer_doc.noun_chunks
            if len(chunk.text.strip()) > 2  # skip trivial chunks like "I", "it"
        ]

        if not noun_phrases:
            # No noun phrases found — fall back to content-word overlap
            return _word_overlap_fallback(answer, context, min_overlap)

        grounded_count = sum(1 for np_ in noun_phrases if np_ in context)
        overlap = grounded_count / len(noun_phrases)
        return overlap >= min_overlap

    except (RuntimeError, ImportError):
        # spaCy not available — degrade gracefully to word overlap
        return _word_overlap_fallback(answer, context, min_overlap)


def _word_overlap_fallback(answer: str, context: str,
                           min_overlap: float) -> bool:
    """Fallback: simple content-word overlap (original Layer 4 logic)."""
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words:
        return True
    overlap = sum(1 for w in words if w in context) / len(words)
    return overlap >= min_overlap