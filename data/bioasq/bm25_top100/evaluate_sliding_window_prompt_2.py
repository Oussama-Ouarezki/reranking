"""
Evaluate BM25 (top-50) vs DeepSeek reranked (top-50) on BioASQ.

Metrics: nDCG@1, nDCG@5, nDCG@10, nDCG@20, nDCG@50, P@10, P@20, R@1, R@5, R@10, R@20, R@50

Reads:
  data/bioasq/processed/qrels.tsv
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2.jsonl

Writes:
  data/bioasq/bm25_top100/images/evaluation_bm25_vs_sliding_window_prompt_2.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/evaluate_deepseek_50.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE        = Path(__file__).resolve().parents[3]
QRELS       = BASE / "data/bioasq/processed/qrels.tsv"
BM25_FILE   = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
DS_SW_FILE   = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2.jsonl"
IMG_DIR     = BASE / "data/bioasq/bm25_top100/images"

TOP_N     = 50
NDCG_KS   = [1, 5, 10, 20, 50]
PREC_KS   = [1, 5, 10, 20, 50]
RECALL_KS = [1, 5, 10, 20, 50]
MRR_KS    = [1, 5, 10, 20, 50]
MAP_KS    = [1, 5, 10, 20, 50]


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in gold) / k


def map_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    hits = 0
    score = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in gold:
            hits += 1
            score += hits / (i + 1)
    return score / len(gold) if gold else 0.0


def mrr_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in gold) / len(gold) if gold else 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(ranked[:k])
        if d in gold
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ── Loaders ───────────────────────────────────────────────────────────────────

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
    return qrels


def load_bm25(path: Path, top_n: int = TOP_N) -> dict[str, list[str]]:
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
            ranked = r.get("permutation") or r.get("reranked")
            if ranked:
                results[r["qid"]] = ranked
    return results


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(results: dict[str, list[str]], qrels: dict[str, set[str]],
             qids: list[str]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    n = 0
    for qid in qids:
        gold = qrels.get(qid, set())
        if not gold:
            continue
        ranked = results.get(qid, [])
        n += 1
        for k in NDCG_KS:
            sums[f"nDCG@{k}"] += ndcg_at_k(ranked, gold, k)
        for k in PREC_KS:
            sums[f"P@{k}"] += precision_at_k(ranked, gold, k)
        for k in RECALL_KS:
            sums[f"R@{k}"] += recall_at_k(ranked, gold, k)
        for k in MRR_KS:
            sums[f"MRR@{k}"] += mrr_at_k(ranked, gold, k)
        for k in MAP_KS:
            sums[f"MAP@{k}"] += map_at_k(ranked, gold, k)
    return {m: 100 * v / n for m, v in sums.items()}


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(ax: plt.Axes, metrics: list[str], bm25: dict[str, float],
                ds: dict[str, float], title: str, show_legend: bool) -> None:
    x = np.arange(len(metrics))
    w = 0.35
    bm25_vals = [bm25[m] for m in metrics]
    ds_vals   = [ds[m]   for m in metrics]

    bars_b = ax.bar(x - w / 2, bm25_vals, w, label="BM25 top-50",
                    color="#4C72B0", alpha=0.88)
    bars_d = ax.bar(x + w / 2, ds_vals,   w, label="DeepSeek-Chat Sliding Prompt v2b (w=20/s=10)",
                    color="#55A868", alpha=0.88)

    for bars, color, vals in [(bars_b, "#4C72B0", bm25_vals),
                               (bars_d, "#55A868", ds_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, max(max(bm25_vals), max(ds_vals)) * 1.3 + 5)
    ax.set_title(title, fontsize=11, pad=8)
    if show_legend:
        ax.legend(fontsize=10)


def plot(bm25: dict[str, float], ds: dict[str, float], n_queries: int) -> None:
    panels = [
        ([f"nDCG@{k}" for k in NDCG_KS],   "nDCG"),
        ([f"P@{k}"    for k in PREC_KS],    "Precision"),
        ([f"R@{k}"    for k in RECALL_KS],  "Recall"),
        ([f"MRR@{k}"  for k in MRR_KS],     "MRR"),
        ([f"MAP@{k}"  for k in MAP_KS],     "MAP"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle(
        f"BM25 vs DeepSeek-Chat Sliding Window (Prompt v2b) — BioASQ  ({n_queries} queries, top-50, w=20/s=10)",
        fontsize=13, y=1.01,
    )

    for ax, (metrics, title), show_legend in zip(axes, panels, [True, False, False, False, False]):
        _draw_panel(ax, metrics, bm25, ds, title, show_legend)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "evaluation_bm25_vs_sliding_window_prompt_2.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading qrels …")
    qrels = load_qrels(QRELS)

    print("Loading BM25 results …")
    bm25_results = load_bm25(BM25_FILE)

    print(f"Loading DeepSeek reranked results from {DS_SW_FILE.name} …")
    ds_results = load_reranked(DS_SW_FILE)

    qids = [qid for qid in ds_results if qrels.get(qid)]
    print(f"Evaluating {len(qids)} queries with gold labels …")

    bm25_scores = evaluate(bm25_results, qrels, qids)
    ds_scores   = evaluate(ds_results,   qrels, qids)

    metrics = ([f"nDCG@{k}" for k in NDCG_KS]
               + [f"P@{k}"   for k in PREC_KS]
               + [f"R@{k}"   for k in RECALL_KS]
               + [f"MRR@{k}" for k in MRR_KS]
               + [f"MAP@{k}" for k in MAP_KS])
    header  = f"  {'Metric':<12}  {'BM25':>8}  {'DeepSeek':>10}  {'Δ':>8}"
    print(f"\n{header}")
    print("  " + "─" * 46)
    for m in metrics:
        delta = ds_scores[m] - bm25_scores[m]
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:<12}  {bm25_scores[m]:>7.2f}%  {ds_scores[m]:>9.2f}%  "
              f"{sign}{delta:>6.2f}%")

    print("\nGenerating plot …")
    plot(bm25_scores, ds_scores, len(qids))
    print("Done.")


if __name__ == "__main__":
    main()
