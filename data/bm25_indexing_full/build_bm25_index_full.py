"""
Build Pyserini/Lucene BM25 indexes for both full merged corpora.

Why Pyserini (Lucene) for maximum query speed:
  - Inverted index on disk — queries touch only the posting lists for query terms,
    not the full corpus. Scales to millions of docs with sub-millisecond latency.
  - Lucene's BM25 is the industry standard: thread-safe, SIMD-optimised internals.
  - Index persists on disk; LuceneSearcher loads in <1 s and streams results.
  - Compared to rank_bm25 (O(N) per query) or bm25s (sparse matrix in RAM),
    Lucene is I/O-efficient and has no RAM footprint for the index itself.

Requires Java 11. The script sets JAVA_HOME automatically.

Inputs:
  data/bioasq/pubmed_full/full/corpus_full.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Outputs (one sub-folder per corpus):
  data/bm25_indexing_full/corpus_full/lucene_index/      ← Lucene index
  data/bm25_indexing_full/corpus_full/index_meta.json
  data/bm25_indexing_full/corpus_full_processed/lucene_index/
  data/bm25_indexing_full/corpus_full_processed/index_meta.json

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python data/bm25_indexing_full/build_bm25_index_full.py

Querying a saved index:
    import os
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher("data/bm25_indexing_full/corpus_full/lucene_index")
    searcher.set_bm25(k1=0.7, b=0.9)
    hits = searcher.search("your query here", k=100)
    for hit in hits:
        print(hit.docid, hit.score)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Java 11 required by Pyserini ─────────────────────────────────────────────
JAVA_HOME = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = f"{JAVA_HOME}/bin:" + os.environ.get("PATH", "")

BASE     = Path(__file__).resolve().parents[2]   # reranking_project/
FULL_DIR = BASE / "data" / "bioasq" / "pubmed_full" / "full"
OUT_BASE = BASE / "data" / "bm25_indexing_full"
PYTHON   = sys.executable

K1      = 0.7
B       = 0.9
THREADS = 4

JOBS = [
    {
        "corpus":  FULL_DIR / "corpus_full.jsonl",
        "out_dir": OUT_BASE / "corpus_full",
    },
    {
        "corpus":  FULL_DIR / "corpus_full_processed.jsonl",
        "out_dir": OUT_BASE / "corpus_full_processed",
    },
]


def convert_to_pyserini_format(corpus_path: Path, pyserini_input_dir: Path) -> int:
    """
    Pyserini's JsonCollection expects one JSONL file where each line has:
        {"id": "<doc_id>", "contents": "<title + text>"}
    """
    pyserini_input_dir.mkdir(parents=True, exist_ok=True)
    out_file = pyserini_input_dir / "docs.jsonl"
    n = 0
    with corpus_path.open(encoding="utf-8") as src, out_file.open("w", encoding="utf-8") as dst:
        for line in src:
            doc: dict[str, str] = json.loads(line)
            contents = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            dst.write(json.dumps({"id": doc["_id"], "contents": contents}, ensure_ascii=False) + "\n")
            n += 1
    return n


def build_lucene_index(pyserini_input_dir: Path, lucene_index_dir: Path) -> None:
    cmd = [
        PYTHON, "-m", "pyserini.index.lucene",
        "--collection",  "JsonCollection",
        "--input",       str(pyserini_input_dir),
        "--index",       str(lucene_index_dir),
        "--generator",   "DefaultLuceneDocumentGenerator",
        "--threads",     str(THREADS),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Pyserini indexing failed (exit {result.returncode})")


def build_index(corpus_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pyserini_input_dir = out_dir / "_pyserini_input"
    lucene_index_dir   = out_dir / "lucene_index"
    meta_path          = out_dir / "index_meta.json"

    print(f"\n{'='*60}")
    print(f"Corpus : {corpus_path.name}")
    print(f"Output : {out_dir}")
    print(f"{'='*60}")

    # ── Step 1: convert to Pyserini JSONL format ──────────────────────────────
    print("Converting to Pyserini format …")
    t0 = time.time()
    n_docs = convert_to_pyserini_format(corpus_path, pyserini_input_dir)
    print(f"  {n_docs:,} documents written in {time.time() - t0:.1f}s")

    # ── Step 2: build Lucene inverted index ───────────────────────────────────
    print(f"\nBuilding Lucene index  (k1={K1}, b={B}, threads={THREADS}) …")
    t0 = time.time()
    build_lucene_index(pyserini_input_dir, lucene_index_dir)
    elapsed = time.time() - t0
    print(f"  Indexed in {elapsed:.1f}s")

    index_size_mb = sum(p.stat().st_size for p in lucene_index_dir.rglob("*") if p.is_file()) / 1e6
    print(f"  Index size : {index_size_mb:.1f} MB  →  {lucene_index_dir}")

    # ── Step 3: clean up temporary conversion files ───────────────────────────
    shutil.rmtree(pyserini_input_dir)
    print(f"  Cleaned up {pyserini_input_dir.name}/")

    # ── Step 4: metadata ──────────────────────────────────────────────────────
    meta = {
        "engine":        "pyserini/lucene",
        "k1":            K1,
        "b":             B,
        "threads":       THREADS,
        "n_docs":        n_docs,
        "index_size_mb": round(index_size_mb, 1),
        "corpus_path":   str(corpus_path),
        "lucene_index":  str(lucene_index_dir),
        "build_time_s":  round(elapsed, 2),
        "built_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
        "java_home":     JAVA_HOME,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Metadata   → {meta_path}")


def main() -> None:
    for job in JOBS:
        build_index(job["corpus"], job["out_dir"])

    print("\n\nAll Lucene indexes built successfully.")
    print("\nTo query:")
    print("  import os; os.environ['JAVA_HOME'] = '/usr/lib/jvm/java-21-openjdk-amd64'")
    print("  from pyserini.search.lucene import LuceneSearcher")
    print("  searcher = LuceneSearcher('data/bm25_indexing_full/corpus_full/lucene_index')")
    print(f"  searcher.set_bm25(k1={K1}, b={B})")
    print("  hits = searcher.search('your query', k=100)")


if __name__ == "__main__":
    main()
