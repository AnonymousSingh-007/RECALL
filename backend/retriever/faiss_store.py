import faiss
import numpy as np

STRATEGIES = ["semantic", "sliding_window", "hierarchical", "metadata_aware"]


class FaissStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.indices: dict[str, faiss.Index] = {}
        self.metadata: dict[str, list[dict]] = {}

    def load(self):
        for strategy in STRATEGIES:
            path = f"{self.index_dir}/{strategy}.faiss"
            # TODO Day 2: self.indices[strategy] = faiss.read_index(path)
            pass

    def search(self, strategy: str, query_vec: np.ndarray, k: int = 20):
        index = self.indices[strategy]
        scores, ids = index.search(query_vec.reshape(1, -1), k)
        return scores[0], ids[0]