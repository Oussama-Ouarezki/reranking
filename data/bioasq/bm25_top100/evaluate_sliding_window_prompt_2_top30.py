"""
Evaluate BM25 (top-30) vs DeepSeek sliding-window reranked (top-30, window=20, step=10).

Reads:
  data/bioasq/processed/qrels.tsv
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2_top30_w20_s10.jsonl

Writes:
  data/bioasq/bm25_top100/images/evaluation_sliding_window_prompt_2_top30.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/evaluate_sliding_window_prompt_2_top30.py
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

BASE      = Path(__file__).resolve().parents[3]
QRELS     = BASE / "data/bioasq/processed/qrels.tsv"
BM25_FILE = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
DS_FILE   = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2_top30_w20_s10.jsonl"
IMG_DIR   = BASE / "data/bioasq/bm25_top100/images"

KS = [1, 5, 10, 20, 50]


# ── Metrics ───────────────────────────────────────────────────────────────────
def ndcg_at_k(r, g, k):
    dcg   = sum(1 / math.log2(i + 2) for i, d in enumerate(r[:k]) if d in g)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(g), k)))
    return dcg / ideal if ideal else 0.0

def precision_at_k(r, g, k): return sum(1 for d in r[:k] if d in g) / k
def recall_at_k(r, g, k):    return sum(1 for d in r[:k] if d in g) / len(g) if g else 0.0
def mrr_at_k(r, g, k):       return next((1 / (i + 1) for i, d in enumerate(r[:k]) if d in g), 0.0)
def map_at_k(r, g, k):
    h = s = 0
    for i, d in enumerate(r[:k]):
        if d in g: h += 1; s += h / (i + 1)
    return s / len(g) if g else 0.0


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_qrels(path):
    qrels = defaultdict(set)
    with path.open() as f:
        next(f)
        for line in f:
            p = line.strip().split("\t")
            qid, did, score = (p[0], p[1], p[2]) if len(p) == 3 else (p[0], p[2], p[3])
            if int(score) > 0: qrels[qid].add(did)
    return qrels

def load_bm25(path):
    results = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            results[r["qid"]] = [h["docid"] for h in r["top100"][:50]]
    return results

def load_reranked(path):
    results = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            ranked = r.get("permutation") or r.get("reranked")
            if ranked: results[r["qid"]] = ranked
    return results


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(results, qrels, qids):
    sums = defaultdict(float); n = 0
    for qid in qids:
        g = qrels.get(qid, set())
        if not g: continue
        r = results.get(qid, []); n += 1
        for k in KS:
            sums[f"nDCG@{k}"] += ndcg_at_k(r, g, k)
            sums[f"P@{k}"]    += precision_at_k(r, g, k)
            sums[f"R@{k}"]    += recall_at_k(r, g, k)
            sums[f"MRR@{k}"]  += mrr_at_k(r, g, k)
            sums[f"MAP@{k}"]  += map_at_k(r, g, k)
    return {m: 100 * v / n for m, v in sums.items()}


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(bm25, ds, n_queries):
    panels = [
        ([f"nDCG@{k}" for k in KS], "nDCG"),
        ([f"P@{k}"    for k in KS], "Precision"),
        ([f"R@{k}"    for k in KS], "Recall"),
        ([f"MRR@{k}"  for k in KS], "MRR"),
        ([f"MAP@{k}"  for k in KS], "MAP"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle(
        f"BM25 vs DeepSeek Sliding-Window top-30 (w=20, s=10, 512 tok) — BioASQ ({n_queries} queries)",
        fontsize=13, y=1.01,
    )
    for ax, (metrics, title), show_legend in zip(axes, panels, [True, False, False, False, False]):
        x = np.arange(len(metrics)); w = 0.35
        bv = [bm25[m] for m in metrics]; dv = [ds[m] for m in metrics]
        bars_b = ax.bar(x - w/2, bv, w, label="BM25 top-50",       color="#4C72B0", alpha=0.88)
        bars_d = ax.bar(x + w/2, dv, w, label="DeepSeek top-30 w=20 s=10", color="#55A868", alpha=0.88)
        for bars, color, vals in [(bars_b, "#4C72B0", bv), (bars_d, "#55A868", dv)]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)
        ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylabel("Score (%)", fontsize=11)
        ax.set_ylim(0, max(max(bv), max(dv)) * 1.3 + 5)
        ax.set_title(title, fontsize=11, pad=8)
        if show_legend: ax.legend(fontsize=10)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "evaluation_sliding_window_prompt_2_top30.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading qrels …")
    qrels = load_qrels(QRELS)
    print("Loading BM25 results …")
    bm25_results = load_bm25(BM25_FILE)
    print(f"Loading DeepSeek results from {DS_FILE.name} …")
    ds_results = load_reranked(DS_FILE)

    qids = [qid for qid in ds_results if qrels.get(qid)]
    print(f"Evaluating {len(qids)} queries with gold labels …")

    bm25_scores = evaluate(bm25_results, qrels, qids)
    ds_scores   = evaluate(ds_results,   qrels, qids)

    metrics = [f"{t}@{k}" for t in ("nDCG", "P", "R", "MRR", "MAP") for k in KS]

    print(f"\ntop_n=30 | window=20 | step=10 | max_tokens=512 | n_queries={len(qids)}")
    print(f"  {'Metric':<12}  {'BM25':>8}  {'DeepSeek':>10}  {'Δ':>8}")
    print("  " + "─" * 46)
    for m in metrics:
        delta = ds_scores[m] - bm25_scores[m]
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:<12}  {bm25_scores[m]:>7.2f}%  {ds_scores[m]:>9.2f}%  {sign}{delta:>6.2f}%")

    print("\nGenerating plot …")
    plot(bm25_scores, ds_scores, len(qids))
    print("Done.")


if __name__ == "__main__":
    main()
