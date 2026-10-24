"""Download public-domain literary texts and append them to corpus_sample.jsonl
as new passages, tagged distinctly so they merge cleanly into the real corpus.

Run this BEFORE re-running the four build_strategyN scripts, since those
scripts read corpus_sample.jsonl fresh each time.
"""
import httpx
import json
import re

CORPUS_PATH = "indexing/corpus_sample.jsonl"

SOURCES = {
    "sherlock_holmes": "https://www.gutenberg.org/files/1661/1661-0.txt",
    "merchant_of_venice": "https://www.gutenberg.org/ebooks/1515.txt.utf-8",
}

# query_id range reserved for literary additions, well clear of the
# real MSMARCO-XI query_ids (which run into the hundreds of thousands)
LITERARY_QID_START = 9_000_000


def strip_gutenberg_boilerplate(text: str) -> str:
    start_marker = re.search(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
    end_marker = re.search(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, re.IGNORECASE)
    start = start_marker.end() if start_marker else 0
    end = end_marker.start() if end_marker else len(text)
    return text[start:end].strip()


def chunk_into_passages(text: str, words_per_passage: int = 200) -> list[str]:
    """Split raw text into ~200-word passages, similar size to MSMARCO passages."""
    words = text.split()
    passages = []
    for i in range(0, len(words), words_per_passage):
        chunk = " ".join(words[i:i + words_per_passage]).strip()
        if len(chunk) > 50:  # skip tiny trailing fragments
            passages.append(chunk)
    return passages


new_entries = []
qid = LITERARY_QID_START

for name, url in SOURCES.items():
    print(f"Downloading {name}...")
    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    raw_text = resp.text

    clean_text = strip_gutenberg_boilerplate(raw_text)
    passages = chunk_into_passages(clean_text)
    print(f"  {len(passages)} passages extracted")

    for idx, passage_text in enumerate(passages):
        new_entries.append({
            "query_id": qid,
            "passage_idx": idx,
            "is_selected": 0,  # no MSMARCO gold label applies; neutral default
            "query_type": "literary",
            "eng_query": "",  # literary passages have no associated query
            "text": passage_text,
            "source": name,  # extra tag, harmless if unused downstream
        })
    qid += 1

print(f"Total new literary passages: {len(new_entries)}")

with open(CORPUS_PATH, "a", encoding="utf-8") as f:
    for entry in new_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Appended to {CORPUS_PATH}")