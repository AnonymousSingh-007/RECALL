import asyncio


def reciprocal_rank_fusion(faiss_results: list, bm25_results: list, k: int = 60) -> list:
    scores: dict[str, float] = {}
    for rank, item in enumerate(faiss_results):
        scores[item["id"]] = scores.get(item["id"], 0) + 1 / (k + rank)
    for rank, item in enumerate(bm25_results):
        scores[item["id"]] = scores.get(item["id"], 0) + 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def retrieve(query: str, strategies: list[str]) -> list:
    # TODO Day 2: run faiss_store.search and bm25_store.search in parallel
    # via asyncio.gather, then fuse with reciprocal_rank_fusion
    raise NotImplementedError