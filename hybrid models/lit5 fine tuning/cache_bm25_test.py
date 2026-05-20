"""Cache BM25 top-100 retrieval on the BioASQ Task13B golden test set.

Uses the pre-built Lucene index (k1=0.7, b=0.9 — standard BioASQ params).
Output is consumed by lr_sweep.py for fast repeated evaluation without
rebuilding/querying BM25 each run.

Output: bm25_top100_test.jsonl
  Each line: {qid, query, candidates: [{docid, text}, ...]}
"""

import json
import os
from pathlib import Path

# Pyserini needs Java 11+; system default is 8.
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

ROOT = Path("/home/oussama/Desktop/reranking_project")
INDEX = ROOT / "data/bm25_indexing_full/corpus_full/lucene_index"
TEST_DIR = ROOT / "data/bioasq/raw/Task13BGoldenEnriched"
BATCHES = ["13B1", "13B2", "13B3", "13B4"]
OUT = ROOT / "lit5 fine tuning/bm25_top100_test.jsonl"
TOP_K = 100


def load_test_queries():
    queries = {}
    for batch in BATCHES:
        with (TEST_DIR / batch / "queries.jsonl").open() as f:
            for line in f:
                q = json.loads(line)
                queries[q["_id"]] = q["text"]
    return queries


def main():
    queries = load_test_queries()
    print(f"Loaded {len(queries)} test queries")

    print(f"Opening Lucene index: {INDEX}")
    searcher = LuceneSearcher(str(INDEX))
    searcher.set_bm25(k1=0.7, b=0.9)

    with OUT.open("w") as f:
        for qid, qtext in tqdm(queries.items(), desc="BM25 retrieve"):
            hits = searcher.search(qtext, k=TOP_K)
            cands = []
            for h in hits:
                raw = json.loads(searcher.doc(h.docid).raw())
                cands.append({"docid": h.docid, "text": raw.get("contents", "")})
            f.write(json.dumps({"qid": qid, "query": qtext, "candidates": cands}) + "\n")

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
