"""
Build a gold-front listwise dataset from deepseek_sliding_reranked_prompt_2.jsonl.

For each query:
  1. Take DeepSeek's top-20 permutation as the base (NOT BM25 order)
  2. Find gold docs (in qrels) within those top 20
  3. Push all gold docs to the FRONT, preserving their relative DeepSeek order
  4. Non-gold docs follow after, also preserving their relative DeepSeek order
  5. Docs ranked 21-50 from DeepSeek are appended unchanged

Example:
  DeepSeek top-20: [A, B*, C, D*, E, F, ...]   (* = gold)
  Output target:   [B*, D*, A, C, E, F, ...]

This gives LiT5 a strong, unambiguous training signal for gold docs
while preserving DeepSeek's distillation knowledge for non-gold ordering.

Output format:
  {"qid": ..., "bm25_order": [...50 doc ids...], "permutation": [...50 doc ids...]}

Usage:
    python create_gold_front_listwise.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

RERANKED_IN = ROOT / "data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl"
QRELS_FILE  = ROOT / "data/bioasq/processed/qrels.tsv"
OUT_FILE    = ROOT / "data/bioasq/reranked/gold_front_listwise.jsonl"


def load_qrels(path: Path) -> dict[str, set[str]]:
    """Returns {qid: set(relevant_docids)} — only docs with score > 0."""
    raw: dict[str, set[str]] = defaultdict(set)
    with path.open() as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            if score > 0:
                raw[qid].add(did)
    return dict(raw)


def gold_front_permutation(top20: list[str], gold: set[str]) -> list[str]:
    """
    Reorder top20 so that:
      - Gold docs come first, in their original DeepSeek relative order
      - Non-gold docs follow, in their original DeepSeek relative order

    Relative order within each group is always preserved.
    """
    gold_docs    = [d for d in top20 if d in gold]
    nongold_docs = [d for d in top20 if d not in gold]
    return gold_docs + nongold_docs


def main():
    print("Loading qrels …")
    qrels = load_qrels(QRELS_FILE)
    print(f"  {len(qrels):,} queries with relevant docs")

    entries_in = []
    with RERANKED_IN.open() as f:
        for line in f:
            entries_in.append(json.loads(line))
    print(f"  {len(entries_in):,} entries in source file")

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_no_qrels      = 0
    n_no_gold_in20  = 0
    n_reordered     = 0
    gold_found_sum  = 0
    gold_total_sum  = 0

    written = 0
    with OUT_FILE.open("w") as fout:
        for entry in entries_in:
            qid           = entry["qid"]
            bm25_order    = entry["bm25_order"]
            deepseek_perm = entry["permutation"]

            top20 = deepseek_perm[:20]
            rest  = deepseek_perm[20:]

            gold = qrels.get(qid, set())

            if not gold:
                n_no_qrels += 1
                permutation = list(deepseek_perm)

            else:
                gold_in_20 = [d for d in top20 if d in gold]
                gold_found_sum += len(gold_in_20)
                gold_total_sum += len(gold)

                if not gold_in_20:
                    n_no_gold_in20 += 1
                    permutation = list(deepseek_perm)

                else:
                    n_reordered += 1
                    reordered_top20 = gold_front_permutation(top20, gold)
                    permutation = reordered_top20 + list(rest)

            fout.write(json.dumps({
                "qid":        qid,
                "bm25_order": bm25_order,
                "permutation": permutation,
            }) + "\n")
            written += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*54}")
    print(f"  Written             : {written:,} entries → {OUT_FILE.name}")
    print(f"  Queries reordered   : {n_reordered:,} / {written:,} ({n_reordered/written*100:.1f}%)")
    print(f"  No gold in top-20   : {n_no_gold_in20:,}  (DeepSeek order kept)")
    print(f"  No qrels            : {n_no_qrels:,}  (DeepSeek order kept)")
    if gold_total_sum:
        recall = gold_found_sum / gold_total_sum
        print(f"  Avg gold recall@20  : {recall:.3f}  ({gold_found_sum:,} / {gold_total_sum:,})")
    print(f"{'─'*54}")
    print("Done.")


if __name__ == "__main__":
    main()