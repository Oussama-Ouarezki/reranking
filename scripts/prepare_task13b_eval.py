"""
Convert Task13B golden files into evaluation format matching data/bioasq/processed/:
  queries.jsonl  — {"_id": qid, "text": body}
  qrels.tsv      — query-id \t corpus-id \t score

One subfolder per batch:
  data/bioasq/raw/Task13BGoldenEnriched/13B{1-4}/queries.jsonl
  data/bioasq/raw/Task13BGoldenEnriched/13B{1-4}/qrels.tsv

Queries whose documents are not in the corpus are dropped and reported.
"""

import json
import re
from pathlib import Path

BASE_DIR    = Path("data/bioasq/raw/Task13BGoldenEnriched")
CORPUS_FILE = BASE_DIR / "corpus.jsonl"


def extract_pmid(url: str) -> str | None:
    m = re.search(r"pubmed/(\d+)", url)
    return m.group(1) if m else None


def load_corpus_ids(corpus_file: Path) -> set:
    ids = set()
    with corpus_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["_id"])
    return ids


def process_golden(path: Path, corpus_ids: set, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    queries_out = out_dir / "queries.jsonl"
    qrels_out   = out_dir / "qrels.tsv"

    with path.open(encoding="utf-8") as f:
        questions = json.load(f)["questions"]

    kept, dropped = 0, 0

    with queries_out.open("w", encoding="utf-8") as qf, \
         qrels_out.open("w",   encoding="utf-8") as rf:

        rf.write("query-id\tcorpus-id\tscore\n")

        for q in questions:
            qid  = q["id"]
            body = q["body"]

            # collect relevant PMIDs from documents field
            pmids = set()
            for url in q.get("documents", []):
                pmid = extract_pmid(url)
                if pmid:
                    pmids.add(pmid)

            if not pmids:
                print(f"  DROP {qid}: no document URLs")
                dropped += 1
                continue

            missing = pmids - corpus_ids
            if missing:
                print(f"  DROP {qid}: {len(missing)} PMIDs not in corpus {missing}")
                dropped += 1
                continue

            # write query
            json.dump({"_id": qid, "text": body}, qf, ensure_ascii=False)
            qf.write("\n")

            # write qrels (all referenced docs are relevant, score=1)
            for pmid in sorted(pmids):
                rf.write(f"{qid}\t{pmid}\t1\n")

            kept += 1

    print(f"  {path.name}: kept {kept}, dropped {dropped} → {out_dir}")


def main():
    print("Loading corpus IDs …")
    corpus_ids = load_corpus_ids(CORPUS_FILE)
    print(f"  {len(corpus_ids):,} documents in corpus\n")

    for i in range(1, 5):
        golden_path = BASE_DIR / f"13B{i}_golden.json"
        out_dir     = BASE_DIR / f"13B{i}"
        process_golden(golden_path, corpus_ids, out_dir)

    print("\nDone. Output structure:")
    for i in range(1, 5):
        out_dir = BASE_DIR / f"13B{i}"
        q_lines = sum(1 for _ in (out_dir / "queries.jsonl").open())
        r_lines = sum(1 for _ in (out_dir / "qrels.tsv").open()) - 1  # minus header
        print(f"  13B{i}/  queries: {q_lines}  qrels: {r_lines}")


if __name__ == "__main__":
    main()
