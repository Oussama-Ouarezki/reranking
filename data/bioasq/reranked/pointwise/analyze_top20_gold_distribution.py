"""
Analyze gold vs non-gold doc distribution in the top-20 of the
DeepSeek sliding-window listwise reranker output.

For each query reports:
  - how many of the top-20 reranked docs are gold
  - % of top-20 that are gold  (gold density)
  - % of top-20 that are non-gold
  - recall@20  (fraction of the query's total gold docs captured in top-20)

Reads:
  data/bioasq/processed/qrels.tsv
  data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2.jsonl

Writes:
  data/bioasq/pointwise_pairwise/top20_gold_distribution.txt

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/pointwise_pairwise/analyze_top20_gold_distribution.py
"""

import json
from collections import defaultdict
from pathlib import Path

BASE       = Path(__file__).resolve().parents[3]
QRELS      = BASE / "data/bioasq/processed/qrels.tsv"
RERANKED   = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2.jsonl"
OUT_FILE   = BASE / "data/bioasq/pointwise_pairwise/top20_gold_distribution.txt"

TOP_K = 20


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            if int(score) > 0:
                qrels[qid].add(did)
    return dict(qrels)


def load_reranked(path: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            ranked = r.get("permutation") or r.get("reranked") or []
            results[r["qid"]] = ranked
    return results


def main() -> None:
    print("Loading qrels …")
    qrels = load_qrels(QRELS)

    print("Loading reranked results …")
    reranked = load_reranked(RERANKED)

    qids = [qid for qid in reranked if qrels.get(qid)]
    print(f"{len(qids)} queries with gold labels\n")

    rows = []
    sum_pct_gold    = 0.0
    sum_pct_nongold = 0.0
    sum_recall      = 0.0

    for qid in qids:
        gold      = qrels[qid]
        top20     = reranked[qid][:TOP_K]
        n_gold    = sum(1 for d in top20 if d in gold)
        n_nongold = TOP_K - n_gold
        pct_gold    = n_gold    / TOP_K * 100
        pct_nongold = n_nongold / TOP_K * 100
        recall      = n_gold / len(gold) * 100 if gold else 0.0

        sum_pct_gold    += pct_gold
        sum_pct_nongold += pct_nongold
        sum_recall      += recall

        rows.append((qid, len(gold), n_gold, n_nongold, pct_gold, pct_nongold, recall))

    n = len(qids)
    avg_pct_gold    = sum_pct_gold    / n
    avg_pct_nongold = sum_pct_nongold / n
    avg_recall      = sum_recall      / n

    # ── Print summary ──────────────────────────────────────────────────────────
    summary = (
        f"DeepSeek Sliding Window (prompt_2) — top-{TOP_K} gold distribution\n"
        f"Queries evaluated : {n}\n"
        f"\n"
        f"  Avg % gold in top-{TOP_K}     : {avg_pct_gold:.2f}%\n"
        f"  Avg % non-gold in top-{TOP_K}  : {avg_pct_nongold:.2f}%\n"
        f"  Avg recall@{TOP_K}             : {avg_recall:.2f}%\n"
        f"    (= avg fraction of each query's gold docs captured in top-{TOP_K})\n"
    )
    print(summary)

    # ── Distribution of gold count in top-20 ──────────────────────────────────
    from collections import Counter
    gold_count_dist = Counter(r[2] for r in rows)
    print(f"  Distribution of #gold docs in top-{TOP_K}:")
    for k in sorted(gold_count_dist):
        pct = gold_count_dist[k] / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {k:>2} gold docs : {gold_count_dist[k]:>5} queries  ({pct:5.1f}%)  {bar}")

    # ── Write per-query TSV ────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w") as f:
        f.write(summary + "\n")
        f.write(f"Distribution of #gold docs in top-{TOP_K}:\n")
        for k in sorted(gold_count_dist):
            pct = gold_count_dist[k] / n * 100
            f.write(f"  {k} gold: {gold_count_dist[k]} queries ({pct:.1f}%)\n")
        f.write("\n")
        header = (f"{'qid':<20}  {'total_gold':>10}  {'gold@20':>7}  "
                  f"{'nongold@20':>10}  {'%gold':>6}  {'%nongold':>9}  {'recall@20':>9}\n")
        f.write(header)
        f.write("─" * len(header) + "\n")
        for qid, total_gold, n_gold, n_nongold, pct_gold, pct_nongold, recall in rows:
            f.write(
                f"{qid:<20}  {total_gold:>10}  {n_gold:>7}  "
                f"{n_nongold:>10}  {pct_gold:>5.1f}%  {pct_nongold:>8.1f}%  {recall:>8.1f}%\n"
            )

    print(f"\nPer-query breakdown saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
