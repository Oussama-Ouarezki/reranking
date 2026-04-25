"""
Compare BM25 vs DeepSeek sliding-window reranker on nDCG@10 and nDCG@20.

Reads:
  data/bioasq/processed/qrels.tsv
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/reranked/compare_bm25_vs_deepseek.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

BASE      = Path(__file__).resolve().parents[3]
QRELS     = BASE / "data/bioasq/processed/qrels.tsv"
BM25_FILE = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
DS_FILE   = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2.jsonl"
OUT_FILE  = BASE / "data/bioasq/reranked/per_query_results.txt"


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


def load_bm25(path: Path, top_n: int = 50) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            results[r["qid"]] = [h["docid"] for h in r["top100"][:top_n]]
    return results


def load_reranked(path: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            ranked = r.get("permutation") or r.get("reranked") or []
            results[r["qid"]] = ranked
    return results


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(ranked[:k])
        if d in gold
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0



def main() -> None:
    qrels    = load_qrels(QRELS)
    bm25     = load_bm25(BM25_FILE)
    reranked = load_reranked(DS_FILE)

    qids = [qid for qid in reranked if qrels.get(qid)]
    ks   = [10, 20]

    header = (f"{'qid':<15}  "
              f"{'BM25@10':>8}  {'DS@10':>7}  {'Δ@10':>7}  "
              f"{'BM25@20':>8}  {'DS@20':>7}  {'Δ@20':>7}")
    sep = "─" * len(header)

    lines_out = [sep, header, sep]
    sums: dict[str, float] = defaultdict(float)
    n = 0

    for qid in qids:
        gold = qrels.get(qid, set())
        if not gold:
            continue
        bm25_ranked = bm25.get(qid, [])
        ds_ranked   = reranked.get(qid, [])

        scores = {}
        for k in ks:
            scores[f"bm25_{k}"] = ndcg_at_k(bm25_ranked, gold, k) * 100
            scores[f"ds_{k}"]   = ndcg_at_k(ds_ranked,   gold, k) * 100
            sums[f"bm25_{k}"] += scores[f"bm25_{k}"]
            sums[f"ds_{k}"]   += scores[f"ds_{k}"]

        d10 = scores["ds_10"] - scores["bm25_10"]
        d20 = scores["ds_20"] - scores["bm25_20"]

        lines_out.append(
            f"{qid:<15}  "
            f"{scores['bm25_10']:>7.1f}%  {scores['ds_10']:>6.1f}%  "
            f"{'+' if d10>=0 else ''}{d10:>5.1f}%  "
            f"{scores['bm25_20']:>7.1f}%  {scores['ds_20']:>6.1f}%  "
            f"{'+' if d20>=0 else ''}{d20:>5.1f}%"
        )
        n += 1

    avgs = {k: sums[k] / n for k in sums}
    d10 = avgs["ds_10"] - avgs["bm25_10"]
    d20 = avgs["ds_20"] - avgs["bm25_20"]
    lines_out.append(sep)
    lines_out.append(
        f"{'AVERAGE ('+str(n)+'q)':<15}  "
        f"{avgs['bm25_10']:>7.1f}%  {avgs['ds_10']:>6.1f}%  "
        f"{'+' if d10>=0 else ''}{d10:>5.1f}%  "
        f"{avgs['bm25_20']:>7.1f}%  {avgs['ds_20']:>6.1f}%  "
        f"{'+' if d20>=0 else ''}{d20:>5.1f}%"
    )
    lines_out.append(sep)

    output = "\n".join(lines_out) + "\n"
    print(output)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(output)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
