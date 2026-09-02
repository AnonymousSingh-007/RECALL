import asyncio


def reciprocal_rank_fusion(faiss_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """Fuse FAISS and BM25 result lists using Reciprocal Rank Fusion.

    Both inputs are lists of dicts with at least a unique identifier field.
    We key on (query_id, passage_idx) since that's the common identifier
    across both FAISS metadata and BM25 results.
    """
    scores: dict[tuple, float] = {}
    doc_lookup: dict[tuple, dict] = {}

    for rank, item in enumerate(faiss_results):
        key = (item["query_id"], item["passage_idx"])
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        doc_lookup[key] = item

    for rank, item in enumerate(bm25_results):
        key = (item["query_id"], item["passage_idx"])
        scores[key] = scores.get(key, 0) + 1 / (k + rank)
        doc_lookup.setdefault(key, item)

    ranked_keys = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_lookup[key], "rrf_score": score} for key, score in ranked_keys]


async def retrieve(query: str, faiss_store, bm25_store, strategies: list[str], k: int = 20) -> list[dict]:
    """Run FAISS and BM25 retrieval in parallel, fuse with RRF."""
    loop = asyncio.get_event_loop()

    async def run_faiss():
        # TODO: once faiss_store.search() is wired to return dicts with
        # query_id/passage_idx/text (matching your metadata jsonl format),
        # call it here per selected strategy and merge results across strategies
        return []

    async def run_bm25():
        return await loop.run_in_executor(None, bm25_store.search, query, k)

    faiss_results, bm25_results = await asyncio.gather(run_faiss(), run_bm25())
    return reciprocal_rank_fusion(faiss_results, bm25_results)