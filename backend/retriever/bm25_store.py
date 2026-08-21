import pickle
from rank_bm25 import BM25Okapi


class BM25Store:
    def __init__(self, path: str):
        self.path = path
        self.bm25: BM25Okapi | None = None
        self.corpus: list[str] = []

    def load(self):
        with open(self.path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.corpus = data["corpus"]

    def search(self, query: str, k: int = 20):
        tokenized = query.lower().split()
        scores = self.bm25.get_scores(tokenized)
        top_idx = scores.argsort()[::-1][:k]
        return [(self.corpus[i], scores[i]) for i in top_idx]