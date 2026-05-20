"""Build a sliding-window training dataset for LiT5-Distill fine-tuning.

Source: data/bioasq/reranked/deepseek_oracle_hybrid.jsonl (2000 queries, 50 docs each)
Steps:
  1. Pick the 1000 most recent queries (sort qids descending — BioASQ qids are
     MongoDB ObjectIds whose first 4 bytes encode a unix timestamp).
  2. For each query, slide a window of size 20 with stride 10 across the BM25
     order (windows starting at 0, 10, 20, 30 -> 4 windows of 20 docs each).
  3. Within each window, present passages bottom-to-top (reversed BM25 order):
     the model sees the worst-ranked candidate first.
  4. Target = the deepseek "permutation" restricted to the window's docids,
     keeping the deepseek relative order (gold ranking among those 20 docs).
  5. Hydrate query text and passage (title + text) from the project's queries
     and full corpus files.
"""

import json
from pathlib import Path

ROOT = Path("/home/oussama/Desktop/reranking_project")
SRC = ROOT / "data/bioasq/reranked/deepseek_oracle_hybrid.jsonl"
QUERIES = ROOT / "data/bioasq/processed/queries.jsonl"
CORPUS = ROOT / "data/bioasq/pubmed_full/full/corpus_full.jsonl"
OUT_DIR = ROOT / "lit5 fine tuning"
OUT = OUT_DIR / "windowed_train.jsonl"

WINDOW_SIZE = 20
STRIDE = 10
N_QUERIES = 1000


def load_queries():
    out = {}
    with open(QUERIES) as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d["text"]
    return out


def load_corpus(needed_ids):
    out = {}
    with open(CORPUS) as f:
        for line in f:
            d = json.loads(line)
            if d["_id"] in needed_ids:
                title = d.get("title", "") or ""
                text = d.get("text", "") or ""
                out[d["_id"]] = (title.strip() + " " + text.strip()).strip()
                if len(out) == len(needed_ids):
                    break
    return out


def main():
    rows = [json.loads(l) for l in open(SRC)]
    rows.sort(key=lambda r: r["qid"], reverse=True)
    rows = rows[:N_QUERIES]
    print(f"Selected {len(rows)} most recent queries")

    queries = load_queries()
    needed = set()
    for r in rows:
        needed.update(r["bm25_order"])
    print(f"Need {len(needed)} unique passages from corpus")

    corpus = load_corpus(needed)
    print(f"Loaded {len(corpus)} / {len(needed)} passages")
    missing = needed - set(corpus.keys())
    if missing:
        print(f"WARNING: {len(missing)} passages missing from corpus (first few: {list(missing)[:5]})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_windows = 0
    with open(OUT, "w") as f:
        for r in rows:
            qid = r["qid"]
            qtext = queries.get(qid, "")
            bm25 = r["bm25_order"]
            perm = r["permutation"]
            perm_rank = {d: i for i, d in enumerate(perm)}

            for start in range(0, len(bm25) - WINDOW_SIZE + 1, STRIDE):
                window = bm25[start:start + WINDOW_SIZE]
                input_docs = list(reversed(window))  # bottom -> top
                target_docs = sorted(window, key=lambda d: perm_rank.get(d, 1_000_000))

                f.write(json.dumps({
                    "qid": qid,
                    "query": qtext,
                    "window_start": start,
                    "input_docids": input_docs,
                    "input_passages": [corpus.get(d, "") for d in input_docs],
                    "target_docids": target_docs,
                }) + "\n")
                n_windows += 1

    print(f"Wrote {n_windows} windows to {OUT}")


if __name__ == "__main__":
    main()
