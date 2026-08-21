def chunk_hierarchical(text: str, parent_size: int = 512, child_size: int = 64) -> list[dict]:
    """Parent-child chunking: index child embeddings, return parent context."""
    # TODO Day 2 (Samratth, GPU): build parent chunks, then child chunks with parent_id links
    raise NotImplementedError