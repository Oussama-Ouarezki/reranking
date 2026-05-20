"""Cache BM25 top-50 hits for the BioASQ Task13BGoldenEnriched test queries.

Uses the existing Pyserini/Lucene index built with k1=0.7, b=0.9 on the full
corpus (257 907 docs). Output mirrors qwen4b_uncertainty/data/bm25_top50_test.jsonl.

Writes: qwen3_0.6b/data/bm25_top50_test.jsonl
  one record per query: {qid, type, query, hits: [{rank, docid, bm25_score, contents}]}
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import json
from pathlib import Path

from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher

BASE = Path(__file__).resolve().parents[1]
QUERIES_F = BASE / "data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl"
INDEX_DIR = BASE / "data/bm25_indexing_full/corpus_full/lucene_index"
OUT = BASE / "qwen3_0.6b/data/bm25_top50_test.jsonl"

K = 50
BM25_K1 = 0.7
BM25_B = 0.9


def load_queries() -> list[dict]:
    rows = []
    with QUERIES_F.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append({"qid": r["_id"], "type": r["type"], "query": r["text"]})
    return rows


def main() -> None:
    queries = load_queries()
    print(f"{len(queries)} test queries")

    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for q in tqdm(queries, unit="q"):
            hits = searcher.search(q["query"], k=K)
            recs = []
            for rank, h in enumerate(hits, start=1):
                raw = json.loads(searcher.doc(h.docid).raw())
                recs.append({
                    "rank": rank,
                    "docid": h.docid,
                    "bm25_score": float(h.score),
                    "contents": raw["contents"],
                })
            f.write(json.dumps({
                "qid": q["qid"],
                "type": q["type"],
                "query": q["query"],
                "hits": recs,
            }) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
