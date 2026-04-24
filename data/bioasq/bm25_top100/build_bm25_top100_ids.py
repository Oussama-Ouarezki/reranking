"""
2000 most-recent BioASQ queries (≥3 gold docs) → BM25 top-100 docids + scores.

Output: data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  {"qid": "...", "query": "...", "top100": [{"docid": "...", "score": 12.34}, ...]}

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/build_bm25_top100_ids.py
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import json
import time
from collections import defaultdict
from pathlib import Path

from pyserini.search.lucene import LuceneSearcher

BASE      = Path(__file__).resolve().parents[3]
QUERIES   = BASE / "data/bioasq/processed/queries.jsonl"
QRELS     = BASE / "data/bioasq/processed/qrels.tsv"
INDEX_DIR = BASE / "data/bm25_indexing_full/corpus_full_processed/lucene_index"
OUT_FILE  = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"

MIN_GOLD  = 3
N_QUERIES = 2000
TOP_K     = 100
BM25_K1, BM25_B = 0.7, 0.9


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            if int(score) > 0:
                qrels[qid].add(did)
    return qrels


def main() -> None:
    print("Loading qrels …")
    qrels = load_qrels(QRELS)

    print("Loading and filtering queries …")
    queries = []
    with QUERIES.open() as f:
        for line in f:
            q = json.loads(line)
            if len(qrels.get(q["_id"], set())) >= MIN_GOLD:
                queries.append(q)

    queries.sort(key=lambda q: q.get("created_at", ""), reverse=True)
    selected = queries[:N_QUERIES]
    print(f"  {len(selected):,} queries selected "
          f"({selected[-1]['created_at'][:10]} → {selected[0]['created_at'][:10]})")

    print(f"\nLoading Pyserini index (k1={BM25_K1}, b={BM25_B}) …")
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)

    qids_list   = [q["_id"]  for q in selected]
    qtexts_list = [q["text"] for q in selected]

    print(f"Retrieving top-{TOP_K} …")
    t0 = time.time()
    batch_hits = searcher.batch_search(
        queries=qtexts_list, qids=qids_list, k=TOP_K, threads=4
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"Writing {OUT_FILE} …")
    with OUT_FILE.open("w") as out:
        for q in selected:
            qid  = q["_id"]
            hits = batch_hits.get(qid, [])
            record = {
                "qid":   qid,
                "query": q["text"],
                "top100": [{"docid": hit.docid, "score": round(float(hit.score), 4)} for hit in hits],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    file_kb = OUT_FILE.stat().st_size / 1_000
    print(f"  {len(selected):,} queries written — {file_kb:.0f} KB → {OUT_FILE}")


if __name__ == "__main__":
    main()
