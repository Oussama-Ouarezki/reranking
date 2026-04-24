"""
Split candidates.jsonl into 4 separate files:
  queries.jsonl          — qid, query, n_gold
  gold_passages.jsonl    — qid, docid, title, text
  hard_negatives.jsonl   — qid, docid, title, text, ce_score, bm25_rank
  easy_negatives.jsonl   — qid, docid, title, text, bm25_rank
"""

import json
from pathlib import Path

IN_FILE = Path(__file__).parent / "candidates.jsonl"
OUT_DIR = Path(__file__).parent / "split"
OUT_DIR.mkdir(exist_ok=True)

queries_file       = OUT_DIR / "queries.jsonl"
gold_file          = OUT_DIR / "gold_passages.jsonl"
hard_file          = OUT_DIR / "hard_negatives.jsonl"
easy_file          = OUT_DIR / "easy_negatives.jsonl"

n_queries = n_gold = n_hard = n_easy = 0

with (
    IN_FILE.open()           as src,
    queries_file.open("w")   as fq,
    gold_file.open("w")      as fg,
    hard_file.open("w")      as fh,
    easy_file.open("w")      as fe,
):
    for line in src:
        rec = json.loads(line)
        qid   = rec["qid"]
        query = rec["query"]

        fq.write(json.dumps({"qid": qid, "query": query, "n_gold": rec["n_gold"]}, ensure_ascii=False) + "\n")
        n_queries += 1

        for doc in rec["gold_passages"]:
            fg.write(json.dumps({"qid": qid, "docid": doc["docid"], "title": doc["title"], "text": doc["text"]}, ensure_ascii=False) + "\n")
            n_gold += 1

        for doc in rec["hard_negatives"]:
            fh.write(json.dumps({"qid": qid, "docid": doc["docid"], "title": doc["title"], "text": doc["text"], "ce_score": doc["ce_score"], "bm25_rank": doc["bm25_rank"]}, ensure_ascii=False) + "\n")
            n_hard += 1

        for doc in rec.get("easy_negatives", []):
            fe.write(json.dumps({"qid": qid, "docid": doc["docid"], "title": doc["title"], "text": doc["text"], "bm25_rank": doc["bm25_rank"]}, ensure_ascii=False) + "\n")
            n_easy += 1

print(f"queries:        {n_queries:>7,}  → {queries_file}")
print(f"gold_passages:  {n_gold:>7,}  → {gold_file}")
print(f"hard_negatives: {n_hard:>7,}  → {hard_file}")
print(f"easy_negatives: {n_easy:>7,}  → {easy_file}")
