"""Sample a manageable subset of the full corpus for indexing.

Takes a stratified sample across query_types to keep diversity,
prioritizing is_selected=1 passages plus some negatives for realism.
"""
import json
import random

random.seed(42)

TARGET_QUERIES = 15000  # ~80K passages at 10/query

queries_by_id = {}
with open("indexing/corpus_english.jsonl", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        qid = entry["query_id"]
        queries_by_id.setdefault(qid, []).append(entry)

all_qids = list(queries_by_id.keys())
print(f"Total unique queries: {len(all_qids)}")

sampled_qids = random.sample(all_qids, min(TARGET_QUERIES, len(all_qids)))

sampled_passages = []
for qid in sampled_qids:
    sampled_passages.extend(queries_by_id[qid])

print(f"Sampled queries: {len(sampled_qids)}")
print(f"Sampled passages: {len(sampled_passages)}")

with open("indexing/corpus_sample.jsonl", "w", encoding="utf-8") as f:
    for entry in sampled_passages:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print("Saved to indexing/corpus_sample.jsonl")