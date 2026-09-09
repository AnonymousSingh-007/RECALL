"""Load all 4 FAISS indices and expose a unified search interface."""
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

STRATEGIES = ["semantic", "sliding_window", "hierarchical", "metadata_aware"]

STRATEGY_META_FILES = {
    "semantic": "semantic_meta.jsonl",
    "sliding_window": "sliding_window_meta.jsonl",
    "hierarchical": "hierarchical_child_meta.jsonl",  # needs parent lookup join
    "metadata_aware": "metadata_aware_meta.jsonl",
}


class FaissStore:
    def __init__(self, index_dir: str = "indexing/indices"):
        self.index_dir = index_dir
        self.indices: dict[str, faiss.Index] = {}
        self.metadata: dict[str, list[dict]] = {}
        self.hierarchical_parents: dict[str, dict] = {}
        self.model: SentenceTransformer | None = None

    def load(self):
        print("Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # CPU fine at inference

        for strategy in STRATEGIES:
            index_path = f"{self.index_dir}/{strategy}.faiss"
            print(f"Loading {strategy} index...")
            self.indices[strategy] = faiss.read_index(index_path)
            self.indices[strategy].nprobe = 10

            meta_path = f"{self.index_dir}/{STRATEGY_META_FILES[strategy]}"
            with open(meta_path, encoding="utf-8") as f:
                self.metadata[strategy] = [json.loads(line) for line in f]

        # hierarchical needs the parent lookup joined in separately
        parent_path = f"{self.index_dir}/hierarchical_parent_lookup.jsonl"
        with open(parent_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                self.hierarchical_parents[entry["parent_id"]] = entry

        print("All indices loaded.")

    def search(self, strategy: str, query: str, k: int = 20) -> list[dict]:
        if strategy not in self.indices:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        scores, ids = self.indices[strategy].search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            meta = self.metadata[strategy][idx]

            if strategy == "hierarchical":
                parent = self.hierarchical_parents.get(meta["parent_id"], {})
                results.append({
                    "text": parent.get("parent_text", ""),
                    "score": float(score),
                    "strategy": strategy,
                    "query_id": parent.get("query_id"),
                    "passage_idx": parent.get("passage_idx"),
                    "is_selected": parent.get("is_selected"),
                })
            else:
                results.append({
                    "text": meta["text"],
                    "score": float(score),
                    "strategy": strategy,
                    "query_id": meta["query_id"],
                    "passage_idx": meta["passage_idx"],
                    "is_selected": meta["is_selected"],
                })

        return results

    def search_multi(self, strategies: list[str], query: str, k: int = 20) -> list[dict]:
        """Search multiple strategies and merge results."""
        all_results = []
        for strategy in strategies:
            all_results.extend(self.search(strategy, query, k))
        return all_results