"""Evaluate and plot — pure Qwen vs linear fusion (global α) vs linear fusion
dynamic (per-type α, all targeting nDCG@10).

Per-type α (all optimised for nDCG@10):
    summary = 0.800   factoid = 0.925   list = 0.875   yesno = 0.750
Global α: 0.825

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/cascade_eval_dyn10.tsv
        qwen4b_uncertainty/plots/cascade_eval_dyn10.png
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ir_measures
from ir_measures import nDCG, Qrel, ScoredDoc

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores.jsonl"
QRELS_F = BASE / "data/bioasq/processed/qrels.tsv"
OUT_TSV = BASE / "qwen4b_uncertainty/data/cascade_eval_dyn10.tsv"
PLOT = BASE / "qwen4b_uncertainty/plots/cascade_eval_dyn10.png"

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, nDCG @ 20]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20"]

ALPHA_GLOBAL = 0.825
ALPHA_DYN10 = {
    "summary": 0.800,
    "factoid": 0.925,
    "list":    0.875,
    "yesno":   0.750,
}


def minmax(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_qrels():
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def evaluate(run, qrels, qid_set):
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main():
    rows = [json.loads(l) for l in SCORES_F.open()]
    qrels = load_qrels()
    print(f"{len(rows)} queries / {len(qrels)} qrel rows")

    qwen_arr, qwen_norm, bm25_norm, docids, qtypes = {}, {}, {}, {}, {}
    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = sorted(r["scores"], key=lambda s: s["qwen_prob"], reverse=True)
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_arr[qid] = q
        qwen_norm[qid] = minmax(q)
        bm25_norm[qid] = minmax(b)
        docids[qid] = [s["docid"] for s in items]

    qids = list(qwen_arr.keys())
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    def build(alpha_fn):
        run = []
        for qid in qids:
            a = alpha_fn(qid)
            s = a * qwen_norm[qid] + (1 - a) * bm25_norm[qid] if a is not None \
                else qwen_arr[qid]
            for d, sv in zip(docids[qid], s):
                run.append(ScoredDoc(qid, d, float(sv)))
        return run

    runs = {
        "Qwen alone":           build(lambda q: None),
        "Linear fusion":        build(lambda q: ALPHA_GLOBAL),
        "Linear fusion dyn 10": build(lambda q: ALPHA_DYN10[qtypes[q]]),
    }

    rows_out = []
    for cfg, run in runs.items():
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            rows_out.append({"config": cfg, "scope": sc, **m})

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # ── Print table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'config':<22}  {'scope':<8}  " +
          "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    print("=" * 90)
    for cfg in runs:
        for sc in ["global"] + TYPES:
            row = df[(df["config"] == cfg) & (df["scope"] == sc)].iloc[0]
            print(f"{cfg:<22}  {sc:<8}  " +
                  "  ".join(f"{row[m]:>8.4f}" for m in METRIC_NAMES))
        print("-" * 90)

    # ── Plot — grouped bars per scope, 4 metrics, 3 configs ──────────────────
    fig, axes = plt.subplots(1, 5, figsize=(28, 5), sharey=False)
    bar_w = 0.27
    cfgs = list(runs.keys())
    palette = ["#9467bd", "#1f77b4", "#2ca02c"]
    x = np.arange(len(METRIC_NAMES))
    for ax, sc in zip(axes, ["global"] + TYPES):
        for i, (cfg, col) in enumerate(zip(cfgs, palette)):
            row = df[(df["config"] == cfg) & (df["scope"] == sc)].iloc[0]
            vals = [row[m] for m in METRIC_NAMES]
            ax.bar(x + (i - 1) * bar_w, vals, bar_w, color=col, label=cfg,
                   edgecolor="white", alpha=0.92)
            for xi, v in zip(x + (i - 1) * bar_w, vals):
                ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_NAMES, fontsize=9)
        title = sc if sc == "global" else f"{sc}  (α={ALPHA_DYN10[sc]:.3f})"
        ax.set_title(title, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("nDCG")
        ax.set_ylim(0.74, max(df[METRIC_NAMES].max()) + 0.04)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(
        "Qwen3-4B + BM25 linear fusion (no gate) — Qwen alone vs global α=0.825 "
        "vs per-type α (all optimised for nDCG@10)  —  500 BioASQ queries",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160); plt.close(fig)
    print(f"  → {PLOT}")


if __name__ == "__main__":
    main()
