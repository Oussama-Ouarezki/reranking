"""Cache BM25 top-50 (Pyserini, k1=0.7, b=0.9) for the 500 selected queries.

Reads:
  qwen4b_uncertainty/data/queries_500.jsonl
  data/bm25_indexing_full/corpus_full_processed/lucene_index
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl  (for title+text)

Writes:
  qwen4b_uncertainty/data/bm25_top50.jsonl
    one record per query: {qid, type, query, hits: [{rank, docid, bm25_score, contents}]}
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import json
import time
from pathlib import Path

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

BASE = Path(__file__).resolve().parents[1]
QUERIES = BASE / "qwen4b_uncertainty/data/queries_2000.jsonl"
INDEX_DIR = BASE / "data/bm25_indexing_full/corpus_full_processed/lucene_index"
CORPUS = BASE / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
OUT = BASE / "qwen4b_uncertainty/data/bm25_top2000.jsonl"

TOP_K = 200
BM25_K1, BM25_B = 0.7, 0.9


def load_queries() -> list[dict]:
    with QUERIES.open() as f:
        return [json.loads(l) for l in f]


def load_corpus() -> dict[str, dict]:
    print("Loading corpus texts …")
    corpus: dict[str, dict] = {}
    with CORPUS.open() as f:
        for line in tqdm(f, total=257907, unit="doc"):
            d = json.loads(line)
            corpus[d["_id"]] = {"title": d.get("title", ""), "text": d.get("text", "")}
    print(f"  {len(corpus):,} docs loaded")
    return corpus


def main() -> None:
    queries = load_queries()
    print(f"{len(queries)} queries loaded")

    print(f"Opening Lucene index (k1={BM25_K1}, b={BM25_B}) …")
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)

    qids = [q["_id"] for q in queries]
    qtexts = [q["text"] for q in queries]

    t0 = time.time()
    print(f"Searching top-{TOP_K} …")
    batch = searcher.batch_search(queries=qtexts, qids=qids, k=TOP_K, threads=4)
    print(f"  done in {time.time()-t0:.1f}s")

    corpus = load_corpus()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    short = 0
    with OUT.open("w") as out:
        for q in tqdm(queries, unit="q"):
            qid = q["_id"]
            hits = batch.get(qid, [])
            recs = []
            for rank, h in enumerate(hits, start=1):
                doc = corpus.get(h.docid)
                if doc is None:
                    missing += 1
                    continue
                contents = (doc["title"] + " " + doc["text"]).strip()
                recs.append({
                    "rank": rank,
                    "docid": h.docid,
                    "bm25_score": float(h.score),
                    "contents": contents,
                })
            if len(recs) < TOP_K:
                short += 1
            out.write(json.dumps({
                "qid": qid,
                "type": q["type"],
                "query": q["text"],
                "hits": recs,
            }, ensure_ascii=False) + "\n")

    print(f"Wrote {OUT}")
    if missing:
        print(f"  warn: {missing} hits missing in corpus jsonl")
    if short:
        print(f"  warn: {short} queries with fewer than {TOP_K} hits")


if __name__ == "__main__":
    main()
