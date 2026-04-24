"""
Evaluate BM25 vs Ollama reranking on:
  - Precision@10
  - nDCG@1, nDCG@5, nDCG@10

Reads:
  data/bioasq/processed/qrels.tsv
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/bm25_top100/ollama_reranked.jsonl

Writes:
  data/bioasq/bm25_top100/images/evaluation_bm25_vs_ollama.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/evaluate_reranking.py
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

BASE       = Path(__file__).resolve().parents[3]
QRELS      = BASE / "data/bioasq/processed/qrels.tsv"
BM25_FILE  = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
OLLAMA_FILE= BASE / "data/bioasq/bm25_top100/ollama_reranked.jsonl"
IMG_DIR    = BASE / "data/bioasq/bm25_top100/images"

CUTOFFS = [1, 5, 10]


# ── Metrics ───────────────────────────────────────────────────────────────────
def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    hits = sum(1 for did in ranked[:k] if did in gold)
    return hits / k


def dcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return sum(
        1.0 / math.log2(i + 2)
        for i, did in enumerate(ranked[:k])
        if did in gold
    )


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    actual  = dcg_at_k(ranked, gold, k)
    ideal   = dcg_at_k(sorted(ranked, key=lambda d: d in gold, reverse=True), gold, k)
    # ideal DCG uses the best possible ordering of the *retrieved* set
    # but nDCG is normalised against the perfect ranking over all relevant docs
    n_rel   = min(len(gold), k)
    ideal_g = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return actual / ideal_g if ideal_g > 0 else 0.0


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


def load_bm25(path: Path) -> dict[str, list[str]]:
    results = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            results[r["qid"]] = r["top100"]
    return results


def load_reranked(path: Path) -> dict[str, list[str]]:
    results = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            ranked = r.get("permutation") or r.get("reranked")
            if ranked:
                results[r["qid"]] = ranked
    return results


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(
    results: dict[str, list[str]],
    qrels:   dict[str, set[str]],
    qids:    list[str],
) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    n = 0
    for qid in qids:
        gold = qrels.get(qid, set())
        if not gold:
            continue
        ranked = results.get(qid, [])
        n += 1
        sums["P@10"] += precision_at_k(ranked, gold, 10)
        for k in CUTOFFS:
            sums[f"nDCG@{k}"] += ndcg_at_k(ranked, gold, k)
    return {metric: 100 * v / n for metric, v in sums.items()}


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(bm25_scores: dict[str, float], ollama_scores: dict[str, float],
         n_queries: int, model_name: str) -> None:
    metrics   = ["nDCG@1", "nDCG@5", "nDCG@10", "P@10"]
    x         = np.arange(len(metrics))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    bm25_vals   = [bm25_scores[m]   for m in metrics]
    ollama_vals = [ollama_scores[m] for m in metrics]

    bars_b = ax.bar(x - bar_width / 2, bm25_vals,   bar_width,
                    label="BM25 (Pyserini)", color="#4C72B0", alpha=0.88)
    bars_o = ax.bar(x + bar_width / 2, ollama_vals, bar_width,
                    label=f"Ollama ({model_name})", color="#DD8452", alpha=0.88)

    for bars, color, vals in [(bars_b, "#4C72B0", bm25_vals),
                               (bars_o, "#DD8452", ollama_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, max(max(bm25_vals), max(ollama_vals)) * 1.2 + 5)
    ax.set_title(
        f"BM25 vs Ollama Reranking — BioASQ\n"
        f"({n_queries} queries, top-20 candidates)",
        fontsize=13, pad=12,
    )
    ax.legend(fontsize=11)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "evaluation_bm25_vs_ollama.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading qrels …")
    qrels = load_qrels(QRELS)

    print("Loading BM25 results …")
    bm25_results = load_bm25(BM25_FILE)

    print("Loading Ollama results …")
    ollama_results = load_reranked(OLLAMA_FILE)

    # only evaluate on queries present in ollama output
    qids = [qid for qid in ollama_results if qrels.get(qid)]
    print(f"Evaluating {len(qids)} queries with gold labels …")

    bm25_scores   = evaluate(bm25_results,   qrels, qids)
    ollama_scores = evaluate(ollama_results, qrels, qids)

    # infer model name from raw_response field if available
    model_name = "llama3.1:8b"

    metrics = ["nDCG@1", "nDCG@5", "nDCG@10", "P@10"]
    print(f"\n  {'Metric':<12}  {'BM25':>8}  {'Ollama':>8}  {'Δ':>8}")
    print("  " + "─" * 42)
    for m in metrics:
        delta = ollama_scores[m] - bm25_scores[m]
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:<12}  {bm25_scores[m]:>7.2f}%  {ollama_scores[m]:>7.2f}%  "
              f"{sign}{delta:>6.2f}%")

    print("\nGenerating plot …")
    plot(bm25_scores, ollama_scores, len(qids), model_name)
    print("Done.")


if __name__ == "__main__":
    main()
