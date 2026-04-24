"""
Select the 2000 most-recent BioASQ training queries with ≥3 gold docs and
retrieve BM25 top-100 from the full processed corpus using Pyserini/Lucene.

Reads:
  data/bioasq/processed/queries.jsonl
  data/bioasq/processed/qrels.tsv
  data/bm25_indexing_full/corpus_full_processed/lucene_index
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl   (for passage text)

Writes:
  data/bioasq/bm25_top100/bm25_top100.jsonl

Output format — one JSON record per query:
  {
    "qid":         "<id>",
    "query":       "<question text>",
    "created_at":  "2024-...",
    "gold_docids": ["pmid", ...],         # ≥ 3 entries
    "bm25_top100": [
      {"rank": 1, "docid": "...", "score": 12.34, "title": "...", "text": "..."},
      ...
    ]
  }

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/build_bm25_top100.py
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import json
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher

BASE       = Path(__file__).resolve().parents[3]
QUERIES    = BASE / "data/bioasq/processed/queries.jsonl"
QRELS      = BASE / "data/bioasq/processed/qrels.tsv"
INDEX_DIR  = BASE / "data/bm25_indexing_full/corpus_full_processed/lucene_index"
CORPUS     = BASE / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
OUT_FILE   = BASE / "data/bioasq/bm25_top100/bm25_top100.jsonl"

MIN_GOLD = 3
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


def load_corpus_texts(path: Path) -> dict[str, dict]:
    print("Loading corpus texts …")
    corpus: dict[str, dict] = {}
    with path.open() as f:
        for line in tqdm(f, total=257907, unit="doc"):
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text":  doc["text"],
            }
    print(f"  {len(corpus):,} documents loaded")
    return corpus


def main() -> None:
    # ── Load qrels ─────────────────────────────────────────────────────────────
    print("Loading qrels …")
    qrels = load_qrels(QRELS)
    print(f"  {len(qrels):,} queries with at least 1 gold doc")

    # ── Load queries, filter ≥3 gold, sort most-recent first ──────────────────
    print("Loading queries …")
    queries = []
    with QUERIES.open() as f:
        for line in f:
            q = json.loads(line)
            if len(qrels.get(q["_id"], set())) >= MIN_GOLD:
                queries.append(q)

    queries.sort(key=lambda q: q.get("created_at", ""), reverse=True)
    selected = queries[:N_QUERIES]

    print(f"  {len(queries):,} queries with ≥{MIN_GOLD} gold docs")
    print(f"  Selected {len(selected):,} most-recent")
    print(f"  Date range: {selected[-1]['created_at']} → {selected[0]['created_at']}")

    # ── BM25 retrieval ─────────────────────────────────────────────────────────
    print(f"\nLoading Pyserini index (k1={BM25_K1}, b={BM25_B}) …")
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)

    qids_list   = [q["_id"]  for q in selected]
    qtexts_list = [q["text"] for q in selected]

    print(f"Retrieving top-{TOP_K} for {len(selected):,} queries …")
    t0 = time.time()
    batch_hits = searcher.batch_search(
        queries=qtexts_list, qids=qids_list, k=TOP_K, threads=4
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Load corpus for passage text ───────────────────────────────────────────
    corpus = load_corpus_texts(CORPUS)

    # ── Write output ───────────────────────────────────────────────────────────
    print(f"\nWriting {OUT_FILE} …")
    missing_docs = 0
    with OUT_FILE.open("w") as out:
        for q in tqdm(selected, unit="query"):
            qid  = q["_id"]
            hits = batch_hits.get(qid, [])

            top100 = []
            for rank, hit in enumerate(hits, start=1):
                doc = corpus.get(hit.docid)
                if doc is None:
                    missing_docs += 1
                    continue
                top100.append({
                    "rank":   rank,
                    "docid":  hit.docid,
                    "score":  round(float(hit.score), 4),
                    "title":  doc["title"],
                    "text":   doc["text"],
                })

            record = {
                "qid":         qid,
                "query":       q["text"],
                "created_at":  q.get("created_at", ""),
                "gold_docids": sorted(qrels[qid]),
                "bm25_top100": top100,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    if missing_docs:
        print(f"  Warning: {missing_docs} retrieved docids not found in corpus text")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n  Queries written : {len(selected):,}")
    print(f"  Output          : {OUT_FILE}")
    file_mb = OUT_FILE.stat().st_size / 1_000_000
    print(f"  File size       : {file_mb:.1f} MB")


if __name__ == "__main__":
    main()
