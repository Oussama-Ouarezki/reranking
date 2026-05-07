"""
Build an oracle listwise dataset from deepseek_sliding_reranked_prompt_2.jsonl.

For each query the permutation becomes:
  1. Relevant docs   (appear in BM25 top-50 AND in qrels), kept in their BM25 order
  2. Irrelevant docs (appear in BM25 top-50 but NOT in qrels), kept in their BM25 order

This gives the model a clean, noise-free training signal: always push what is
known to be relevant to the top, without relying on DeepSeek's ordering.

Output format matches the fine-tuning script's expected schema:
  {"qid": ..., "bm25_order": [...50 doc ids...], "permutation": [...50 doc ids...]}

Stats printed at end so you can verify coverage.

Usage:
    python data/bioasq/reranked/create_oracle_listwise.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RERANKED_IN  = ROOT / "data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl"
QRELS_FILE   = ROOT / "data/bioasq/processed/qrels.tsv"
OUT_FILE     = ROOT / "data/bioasq/reranked/oracle_listwise.jsonl"


def load_qrels(path: Path) -> dict[str, list[tuple[str, int]]]:
    """Returns {qid: [(docid, score), ...]} sorted by score desc."""
    raw: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open() as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                raw[qid][did] = score
    return {qid: sorted(docs.items(), key=lambda x: -x[1]) for qid, docs in raw.items()}


def build_oracle_permutation(
    bm25_order: list[str],
    relevant: dict[str, int],
) -> list[str]:
    """
    Stable oracle sort: relevant docs (by relevance score desc, then BM25 position)
    followed by irrelevant docs (in original BM25 order).
    """
    rel_in_list    = [(i, did, relevant[did]) for i, did in enumerate(bm25_order) if did in relevant]
    nonrel_in_list = [(i, did) for i, did in enumerate(bm25_order) if did not in relevant]

    # Sort relevant: higher score first; break ties by original BM25 rank (lower = earlier)
    rel_sorted    = sorted(rel_in_list, key=lambda x: (-x[2], x[0]))
    nonrel_sorted = sorted(nonrel_in_list, key=lambda x: x[0])  # BM25 order preserved

    return [did for _, did, *_ in rel_sorted] + [did for _, did in nonrel_sorted]


def main():
    print("Loading qrels …")
    qrels = load_qrels(QRELS_FILE)
    print(f"  {len(qrels):,} queries with relevant docs")

    entries_in = []
    with RERANKED_IN.open() as f:
        for line in f:
            entries_in.append(json.loads(line))
    print(f"  {len(entries_in):,} entries in source file")

    # ── Stats tracking ────────────────────────────────────────────────────────
    n_no_qrels     = 0   # qid not in qrels at all
    n_no_rel_in50  = 0   # qid has qrels but none in BM25 top-50
    n_partial      = 0   # some relevant docs found
    n_full         = 0   # all relevant docs found
    rel_found_sum  = 0
    rel_total_sum  = 0

    written = 0
    with OUT_FILE.open("w") as fout:
        for entry in entries_in:
            qid        = entry["qid"]
            bm25_order = entry["bm25_order"]   # list of 50 doc ids

            relevant = dict(qrels.get(qid, []))  # {docid: score}

            if not relevant:
                n_no_qrels += 1
                # Fallback: use BM25 order as-is (no oracle signal available)
                permutation = list(bm25_order)
            else:
                rel_in_50 = {d: s for d, s in relevant.items() if d in bm25_order}
                rel_found_sum  += len(rel_in_50)
                rel_total_sum  += len(relevant)

                if not rel_in_50:
                    n_no_rel_in50 += 1
                    permutation = list(bm25_order)
                else:
                    if len(rel_in_50) == len(relevant):
                        n_full += 1
                    else:
                        n_partial += 1
                    permutation = build_oracle_permutation(bm25_order, relevant)

            fout.write(json.dumps({
                "qid":        qid,
                "bm25_order": bm25_order,
                "permutation": permutation,
            }) + "\n")
            written += 1

    # ── Report ────────────────────────────────────────────────────────────────
    has_oracle = n_partial + n_full
    print(f"\n{'─'*52}")
    print(f"  Written         : {written:,} entries → {OUT_FILE.name}")
    print(f"  Oracle signal   : {has_oracle:,} / {written:,} queries ({has_oracle/written*100:.1f}%)")
    print(f"    all rel in 50 : {n_full:,}  ({n_full/written*100:.1f}%)")
    print(f"    partial       : {n_partial:,}  ({n_partial/written*100:.1f}%)")
    print(f"    no rel in 50  : {n_no_rel_in50:,}  (BM25 order used as fallback)")
    print(f"    no qrels      : {n_no_qrels:,}  (BM25 order used as fallback)")
    if rel_total_sum:
        print(f"  Avg recall@50   : {rel_found_sum/rel_total_sum:.3f}")
    print(f"{'─'*52}")
    print("Done.")


if __name__ == "__main__":
    main()
