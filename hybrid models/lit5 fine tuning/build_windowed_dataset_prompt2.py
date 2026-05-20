"""Build a sliding-window LiT5 training set from the prompt-2 DeepSeek teacher.

Source : data/bioasq/reranked/deepseek_sliding_reranked_prompt_2_1000.jsonl
         999 queries, each with a bm25_order (50 docids) and a DeepSeek
         permutation (50 docids re-ranked by the teacher; no gold-pushing).
Output : lit5 fine tuning/windowed_train_prompt2_2400.jsonl  (2400 windows)

Steps:
  1. Pick the 600 most recent queries (sort qids descending — BioASQ qids are
     MongoDB ObjectIds whose first 4 bytes encode a unix timestamp).
  2. For each query, slide a window of size 20 with stride 10 across the BM25
     order. With 50 docs that is windows starting at 0, 10, 20, 30 → exactly
     4 windows of 20 docs per query → 600 × 4 = 2400 rows.
  3. Within each window, present passages bottom-to-top (reversed BM25 order):
     the model sees the worst-ranked candidate first.
  4. Target = the deepseek permutation restricted to the window's docids,
     keeping the deepseek relative order (the teacher's ranking among those
     20 docs).
  5. Hydrate query text and passage (title + text) from the project's queries
     and full corpus files.

Schema matches the existing windowed_train.jsonl so the same kaggle/train.py
(or train_local.py) consumes it without changes.
"""

import json
from pathlib import Path

ROOT = Path("/home/oussama/Desktop/reranking_project")
SRC = ROOT / "data/bioasq/reranked/deepseek_sliding_reranked_prompt_2_1000.jsonl"
QUERIES = ROOT / "data/bioasq/processed/queries.jsonl"
CORPUS = ROOT / "data/bioasq/pubmed_full/full/corpus_full.jsonl"
OUT_DIR = ROOT / "lit5 fine tuning"
OUT = OUT_DIR / "windowed_train_prompt2_2400.jsonl"

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
    print(f"Selected {len(rows)} most recent queries from {SRC.name}")

    queries = load_queries()
    needed = set()
    for r in rows:
        needed.update(r["bm25_order"])
    print(f"Need {len(needed)} unique passages from corpus")

    corpus = load_corpus(needed)
    print(f"Loaded {len(corpus)} / {len(needed)} passages")
    missing = needed - set(corpus.keys())
    if missing:
        print(f"WARNING: {len(missing)} passages missing from corpus "
              f"(first few: {list(missing)[:5]})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_windows = 0
    n_skipped_short = 0
    with open(OUT, "w") as f:
        for r in rows:
            qid = r["qid"]
            qtext = queries.get(qid, "")
            bm25 = r["bm25_order"]
            perm = r["permutation"]
            perm_rank = {d: i for i, d in enumerate(perm)}

            if len(bm25) < WINDOW_SIZE:
                n_skipped_short += 1
                continue

            for start in range(0, len(bm25) - WINDOW_SIZE + 1, STRIDE):
                window = bm25[start:start + WINDOW_SIZE]
                input_docs = list(reversed(window))  # bottom -> top
                target_docs = sorted(window,
                                     key=lambda d: perm_rank.get(d, 1_000_000))

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
    if n_skipped_short:
        print(f"Skipped {n_skipped_short} queries with <{WINDOW_SIZE} bm25 docs")


if __name__ == "__main__":
    main()
