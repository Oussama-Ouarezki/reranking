"""Build a sliding-window training dataset for LiT5-Distill fine-tuning.

Source: data/bioasq/reranked/deepseek_oracle_hybrid.jsonl (2000 queries, 50 docs each)
Steps:
  1. Pick the 1000 most recent queries (sort qids descending — BioASQ qids are
     MongoDB ObjectIds whose first 4 bytes encode a unix timestamp).
  2. For each query, create exactly TWO windows: [0:20] and [10:30].
  3. Within each window, present passages bottom-to-top (reversed BM25 order):
     the model sees the worst-ranked candidate first.
  4. Target = position-based permutation like "[5] > [3] > [7] > ..." where
     numbers are 1-based positions in the INPUT order (reversed BM25).
  5. Hydrate query text and passage (title + text) from the project's queries
     and full corpus files.
  6. Input formatted exactly as LiT5 expects:
     "Query: {q} Document: [1] {p1} [2] {p2} ... [20] {p20} Relevant Document:"
"""

import json
from pathlib import Path

ROOT    = Path("/home/oussama/Desktop/reranking_project")
SRC     = ROOT / "data/bioasq/reranked/deepseek_oracle_hybrid.jsonl"
QUERIES = ROOT / "data/bioasq/processed/queries.jsonl"
CORPUS  = ROOT / "data/bioasq/pubmed_full/full/corpus_full.jsonl"
OUT_DIR = ROOT / "lit5 fine tuning"
OUT     = OUT_DIR / "windowed_train.jsonl"

WINDOW_SIZE = 20
N_QUERIES   = 2000
WINDOWS     = [(0, 20), (10, 30)]   # exactly two windows per query


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
                text  = d.get("text",  "") or ""
                out[d["_id"]] = (title.strip() + " " + text.strip()).strip()
                if len(out) == len(needed_ids):
                    break
    return out


def build_input_string(query, passages):
    """
    Format input exactly as LiT5 expects:
      "Query: {q} Document: [1] {p1} [2] {p2} ... [N] {pN} Relevant Document:"
    Passages are already in INPUT order (reversed BM25 = worst first).
    [1] = worst BM25 doc in the window, [20] = best BM25 doc in the window.
    """
    doc_parts = " ".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
    return f"Query: {query} Document: {doc_parts} Relevant Document:"


def build_target(window, perm_rank):
    """
    Build position-based permutation target string.

    Input order is reversed BM25 (worst first), so:
      prompt [1]  <->  BM25 index (WINDOW_SIZE - 1)  (worst)
      prompt [20] <->  BM25 index 0                  (best)

    We sort window positions by DeepSeek rank, then convert each BM25 position
    to its 1-based prompt position: prompt_pos = WINDOW_SIZE - bm25_pos

    Returns e.g. "[5] > [3] > [20] > ..."
    """
    sorted_bm25_positions = sorted(
        range(len(window)),
        key=lambda i: perm_rank.get(window[i], 1_000_000)
    )
    return " > ".join(f"[{WINDOW_SIZE - p}]" for p in sorted_bm25_positions)


def main():
    rows = [json.loads(l) for l in open(SRC)]
    rows.sort(key=lambda r: r["qid"], reverse=True)
    rows = rows[:N_QUERIES]
    print(f"Selected {len(rows)} most recent queries")

    queries = load_queries()

    needed = set()
    for r in rows:
        for start, end in WINDOWS:
            needed.update(r["bm25_order"][start:end])
    print(f"Need {len(needed)} unique passages from corpus")

    corpus = load_corpus(needed)
    print(f"Loaded {len(corpus)} / {len(needed)} passages")
    missing = needed - set(corpus.keys())
    if missing:
        print(f"WARNING: {len(missing)} passages missing from corpus "
              f"(first few: {list(missing)[:5]})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_windows = 0

    with open(OUT, "w") as f:
        for r in rows:
            qid   = r["qid"]
            qtext = queries.get(qid, "")
            bm25  = r["bm25_order"]
            perm  = r["permutation"]

            perm_rank = {d: i for i, d in enumerate(perm)}

            for start, end in WINDOWS:
                window     = bm25[start:end]
                input_docs = list(reversed(window))   # worst first -> [1]...[20]
                passages   = [corpus.get(d, "") for d in input_docs]

                input_str = build_input_string(qtext, passages)
                target    = build_target(window, perm_rank)

                f.write(json.dumps({
                    "qid":          qid,
                    "query":        qtext,
                    "window_start": start,
                    "input_docids": input_docs,
                    "input":        input_str,
                    "target":       target,
                }) + "\n")
                n_windows += 1

    print(f"Wrote {n_windows} windows to {OUT}")
    print(f"Expected: {len(rows)} queries x {len(WINDOWS)} windows = {len(rows) * len(WINDOWS)}")


if __name__ == "__main__":
    main()