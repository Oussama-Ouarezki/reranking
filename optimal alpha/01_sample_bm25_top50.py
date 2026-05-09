"""Sample 500 BioASQ training queries (with qrels) and cache BM25 top-50.

Uses the same Pyserini/Lucene index as the test set (k1=0.7, b=0.9).

Writes: optimal alpha/data/bm25_top50_train500.jsonl
        optimal alpha/data/qrels_train500.tsv
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import json
import random
from pathlib import Path

from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher

BASE = Path(__file__).resolve().parents[1]
QUERIES_F = BASE / "data/bioasq/processed/queries.jsonl"
QRELS_F   = BASE / "data/bioasq/processed/qrels.tsv"
INDEX_DIR = BASE / "data/bm25_indexing_full/corpus_full/lucene_index"
OUT_DIR   = BASE / "optimal alpha/data"
OUT_HITS  = OUT_DIR / "bm25_top50_train500.jsonl"
OUT_QRELS = OUT_DIR / "qrels_train500.tsv"

K = 50
BM25_K1, BM25_B = 0.7, 0.9
N_SAMPLE = 500
SEED = 42


def main() -> None:
    # Load qrels & queries
    qid_to_qrels: dict[str, list[tuple[str, int]]] = {}
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qid_to_qrels.setdefault(p[0], []).append((p[1], int(p[2])))

    queries = []
    with QUERIES_F.open() as f:
        for line in f:
            r = json.loads(line)
            qid = r["_id"]
            if qid in qid_to_qrels:
                queries.append({"qid": qid, "type": r["type"], "query": r["text"]})
    print(f"{len(queries)} train queries with qrels")

    rng = random.Random(SEED)
    sample = rng.sample(queries, N_SAMPLE)
    sample_qids = {q["qid"] for q in sample}
    print(f"sampled {len(sample)} queries")

    # Write qrels for the sample
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_QRELS.open("w") as f:
        f.write("query-id\tcorpus-id\tscore\n")
        for q in sample:
            for d, s in qid_to_qrels[q["qid"]]:
                f.write(f"{q['qid']}\t{d}\t{s}\n")
    print(f"wrote {OUT_QRELS}")

    # Type breakdown
    from collections import Counter
    types = Counter(q["type"] for q in sample)
    print(f"type breakdown: {dict(types)}")

    # BM25 retrieval
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)

    with OUT_HITS.open("w") as f:
        for q in tqdm(sample, unit="q"):
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
    print(f"wrote {OUT_HITS}")


if __name__ == "__main__":
    main()
