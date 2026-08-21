def chunk_sliding_window(text: str, size: int = 256, overlap: int = 64) -> list[dict]:
    """Fixed-size token chunks with overlap. Returns chunks with position metadata."""
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append({
            "text": " ".join(tokens[start:end]),
            "start_token": start,
            "end_token": end,
        })
        start += size - overlap
    return chunks