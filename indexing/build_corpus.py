"""Flatten MSMARCO-XI English passages into a single corpus for chunking.

Filters out queries with no answer. Keeps is_selected as relevance metadata.
"""
import pyarrow.parquet as pq
import pandas as pd
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="ai4bharat/MSMARCO-XI",
    filename="train/hintrain.parquet",
    repo_type="dataset",
)

pf = pq.ParquetFile(path)

corpus = []
skipped_no_answer = 0

for batch in pf.iter_batches(batch_size=1000):
    df = batch.to_pandas()
    for _, row in df.iterrows():
        if row["Eng_Answer"] == "No Answer Present.":
            skipped_no_answer += 1
            continue

        passages = row["passages"]["English_passages"]
        selected = row["passages"]["is_selected"]

        for idx, (passage_text, is_sel) in enumerate(zip(passages, selected)):
            corpus.append({
                "query_id": int(row["query_id"]),
                "passage_idx": idx,
                "text": passage_text,
                "is_selected": int(is_sel),
                "query_type": row["query_type"],
                "eng_query": row["Eng_Query"],
            })

print(f"Skipped {skipped_no_answer} queries with no answer")
print(f"Total passages in corpus: {len(corpus)}")

with open("indexing/corpus_english.jsonl", "w", encoding="utf-8") as f:
    for entry in corpus:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Saved to indexing/corpus_english.jsonl")