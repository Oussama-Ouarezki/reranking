"""
Build a hybrid listwise dataset from deepseek_sliding_reranked_prompt_2.jsonl.

For each query:
  - bm25_order  : unchanged (model input order)
  - permutation : take DeepSeek's permutation as the base, then
                  stable-partition it so relevant docs (from qrels) come
                  first — preserving their relative order within DeepSeek —
                  followed by irrelevant docs in their DeepSeek order.

Example
-------
DeepSeek permutation : [A, C, E, B, D]
Gold relevant docs   : {B, C}
New permutation      : [C, B, A, E, D]
                        ↑──↑  ↑──────↑
                        rel   non-rel (both in DeepSeek order)

Usage:
    python data/bioasq/reranked/create_deepseek_oracle_hybrid.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT         = Path(__file__).resolve().parents[3]
SOURCE_FILE  = ROOT / "data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl"
QRELS_FILE   = ROOT / "data/bioasq/processed/qrels.tsv"
OUT_FILE     = ROOT / "data/bioasq/reranked/deepseek_oracle_hybrid.jsonl"


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Returns {qid: {docid: score}} for score > 0."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                qrels[parts[0]][parts[1]] = int(parts[2])
    return dict(qrels)


def hybrid_permutation(deepseek_perm: list[str], relevant: dict[str, int]) -> list[str]:
    """
    Stable-partition deepseek_perm: relevant docs first (DeepSeek order),
    then non-relevant (DeepSeek order).
    """
    rel    = [did for did in deepseek_perm if did in relevant]
    nonrel = [did for did in deepseek_perm if did not in relevant]
    return rel + nonrel


def main():
    print("Loading qrels …")
    qrels = load_qrels(QRELS_FILE)
    print(f"  {len(qrels):,} queries with relevant docs")

    entries = []
    with SOURCE_FILE.open() as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  {len(entries):,} entries in source file")

    n_no_qrels    = 0
    n_no_rel_in50 = 0
    n_partial     = 0
    n_full        = 0
    rel_found = rel_total = 0

    with OUT_FILE.open("w") as fout:
        for entry in entries:
            qid       = entry["qid"]
            bm25_order = entry["bm25_order"]
            deepseek_perm = entry["permutation"]
            relevant = qrels.get(qid, {})

            if not relevant:
                n_no_qrels += 1
                permutation = list(deepseek_perm)
            else:
                rel_in_50 = {d: s for d, s in relevant.items() if d in set(bm25_order)}
                rel_found += len(rel_in_50)
                rel_total += len(relevant)

                if not rel_in_50:
                    n_no_rel_in50 += 1
                    permutation = list(deepseek_perm)
                else:
                    if len(rel_in_50) == len(relevant):
                        n_full += 1
                    else:
                        n_partial += 1
                    permutation = hybrid_permutation(deepseek_perm, rel_in_50)

            fout.write(json.dumps({
                "qid":        qid,
                "bm25_order": bm25_order,
                "permutation": permutation,
            }) + "\n")

    has_signal = n_partial + n_full
    print(f"\n{'─'*54}")
    print(f"  Written         : {len(entries):,}  →  {OUT_FILE.name}")
    print(f"  Oracle signal   : {has_signal:,} / {len(entries):,} ({has_signal/len(entries)*100:.1f}%)")
    print(f"    all rel in 50 : {n_full:,}  ({n_full/len(entries)*100:.1f}%)")
    print(f"    partial       : {n_partial:,}  ({n_partial/len(entries)*100:.1f}%)")
    print(f"    no rel in 50  : {n_no_rel_in50:,}  (DeepSeek order kept)")
    print(f"    no qrels      : {n_no_qrels:,}  (DeepSeek order kept)")
    if rel_total:
        print(f"  Avg recall@50   : {rel_found/rel_total:.3f}")
    print(f"{'─'*54}")


if __name__ == "__main__":
    main()
