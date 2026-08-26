"""Build and query a BM25 index from the corpus sample."""
import json
import pickle
from rank_bm25 import BM25Okapi


def build_bm25_index(corpus_path: str, output_path: str):
    """One-time build step — run locally, saves a pickle."""
    texts = []
    meta = []

    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            meta.append({
                "query_id": rec["query_id"],
                "passage_idx": rec["passage_idx"],
                "is_selected": rec["is_selected"],
            })

    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    with open(output_path, "wb") as f:
        pickle.dump({"bm25": bm25, "texts": texts, "meta": meta}, f)

    print(f"Built BM25 index: {len(texts)} documents -> {output_path}")


class BM25Store:
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.bm25 = None
        self.texts = []
        self.meta = []

    def load(self):
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.texts = data["texts"]
            self.meta = data["meta"]

    def search(self, query: str, k: int = 20):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = scores.argsort()[::-1][:k]
        return [
            {"text": self.texts[i], "score": float(scores[i]), **self.meta[i]}
            for i in top_idx
        ]


if __name__ == "__main__":
    build_bm25_index(
        corpus_path="indexing/corpus_sample.jsonl",
        output_path="indexing/indices/bm25_index.pkl",
    )