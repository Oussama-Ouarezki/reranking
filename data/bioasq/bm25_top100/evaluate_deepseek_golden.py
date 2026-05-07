"""
Evaluate BM25 (top-50) vs DeepSeek sliding-window reranked (top-50) on BioASQ.
Uses Task13BGoldenEnriched qrels and the output of deepseek_sliding_window_golden.py.

Metrics match the application exactly (ir_measures backend):
  nDCG@1/5/10/20, P@1/5/10/20, R@1/5/10/20, MRR@1/5/10/20, MAP@1/5/10/20

Reads:
  data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
  data/bioasq/bm25_top100/deepseek_sliding_reranked_golden.jsonl

Writes:
  data/bioasq/bm25_top100/images/evaluation_bm25_vs_deepseek_golden.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/evaluate_deepseek_golden.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ir_measures import nDCG, RR, Recall, P, AP, ScoredDoc, Qrel
import ir_measures

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE    = Path(__file__).resolve().parents[3]
QRELS   = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
DS_FILE = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_golden copy.jsonl"
IMG_DIR = BASE / "data/bioasq/bm25_top100/images"

KS = [1, 5, 10, 20]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_qrels(path):
    """Returns {qid: {docid: relevance_score}} — matches application deps.py exactly."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        header = f.readline()
        if not header.startswith("query-id"):
            f.seek(0)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load_results(path):
    """Returns (bm25_results, reranked_results) both as {qid: [docid, ...]}."""
    bm25 = {}
    reranked = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            bm25[r["qid"]]     = r["bm25_order"]
            reranked[r["qid"]] = r["permutation"]
    return bm25, reranked


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(results, qrels, qids):
    """Compute aggregate metrics using ir_measures — identical to application ranking.py.

    Assigns descending positional scores (n, n-1, …) so rank order is preserved,
    matching how generation.py recomputes metrics from saved doc-id lists.
    Returns ({metric_name: percent_value}, n_evaluated_queries).
    """
    run_objs: list[ScoredDoc] = []
    qrel_objs: list[Qrel] = []
    n_evaluated = 0

    for qid in qids:
        if qid not in qrels or not qrels[qid]:
            continue
        ranked = results.get(qid, [])
        if not ranked:
            continue
        n = len(ranked)
        for i, docid in enumerate(ranked):
            run_objs.append(ScoredDoc(qid, docid, score=float(n - i)))
        for docid, score in qrels[qid].items():
            qrel_objs.append(Qrel(qid, docid, int(score)))
        n_evaluated += 1

    measure_list = []
    for k in KS:
        measure_list.extend([nDCG @ k, RR @ k, P @ k, Recall @ k, AP @ k])

    agg = ir_measures.calc_aggregate(measure_list, qrel_objs, run_objs)

    out: dict[str, float] = {}
    for k in KS:
        out[f"nDCG@{k}"] = round(float(agg.get(nDCG @ k, 0.0)) * 100, 2)
        out[f"MRR@{k}"]  = round(float(agg.get(RR @ k,   0.0)) * 100, 2)
        out[f"P@{k}"]    = round(float(agg.get(P @ k,    0.0)) * 100, 2)
        out[f"R@{k}"]    = round(float(agg.get(Recall @ k, 0.0)) * 100, 2)
        out[f"MAP@{k}"]  = round(float(agg.get(AP @ k,   0.0)) * 100, 2)
    return out, n_evaluated


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(ax, metrics, bm25, ds, title, show_legend):
    x = np.arange(len(metrics))
    w = 0.35
    bm25_vals = [bm25[m] for m in metrics]
    ds_vals   = [ds[m]   for m in metrics]

    bars_b = ax.bar(x - w / 2, bm25_vals, w, label="BM25 top-50",
                    color="#4C72B0", alpha=0.88)
    bars_d = ax.bar(x + w / 2, ds_vals,   w,
                    label="DeepSeek-Chat sliding window (w=20/s=10)",
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
        f"BM25 vs DeepSeek-Chat Sliding Window (w=20/s=10) — BioASQ Golden ({n_queries} queries, top-50)",
        fontsize=13, y=1.01,
    )
    for ax, (metrics, title), show_legend in zip(
        axes, panels, [True, False, False, False, False]
    ):
        _draw_panel(ax, metrics, bm25, ds, title, show_legend)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "evaluation_bm25_vs_deepseek_golden.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading qrels …")
    qrels = load_qrels(QRELS)
    print(f"  {len(qrels)} queries with gold labels")

    print(f"Loading results from {DS_FILE.name} …")
    bm25_results, ds_results = load_results(DS_FILE)
    print(f"  {len(ds_results)} queries reranked")

    qids = list(ds_results.keys())

    bm25_scores, n_queries = evaluate(bm25_results, qrels, qids)
    ds_scores,   _         = evaluate(ds_results,   qrels, qids)
    print(f"Evaluated {n_queries} queries with gold labels")

    metric_order = (
        [f"nDCG@{k}" for k in KS]
        + [f"P@{k}"   for k in KS]
        + [f"R@{k}"   for k in KS]
        + [f"MRR@{k}" for k in KS]
        + [f"MAP@{k}" for k in KS]
    )
    print(f"\n  {'Metric':<12}  {'BM25':>8}  {'DeepSeek':>10}  {'Δ':>8}")
    print("  " + "─" * 46)
    for m in metric_order:
        delta = ds_scores[m] - bm25_scores[m]
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:<12}  {bm25_scores[m]:>7.2f}%  {ds_scores[m]:>9.2f}%  {sign}{delta:>6.2f}%")

    print("\nGenerating plot …")
    plot(bm25_scores, ds_scores, n_queries)
    print("Done.")


if __name__ == "__main__":
    main()
