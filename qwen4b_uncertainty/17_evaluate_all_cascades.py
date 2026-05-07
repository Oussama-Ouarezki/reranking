"""Evaluate all Qwen+BM25 fusion cascades end-to-end and plot.

Produces two plots:
  cascade_eval_all.png     — grouped bars for every variant
  cascade_eval_focus.png   — pure Qwen vs dynamic α vs gated dynamic α (focused)

Configs:
  pure_qwen
  qwen4b_linear_fusion              α=0.825 for every query
  qwen4b_linear_fusion_dynamic      mixed-target per-type α
  qwen4b_linear_fusion_dynamic_10   all-nDCG@10 per-type α
  qwen4b_linear_fusion_dynamic_gated  per-type (α, τ) on H@20 entropy

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/cascade_eval_all.tsv
        qwen4b_uncertainty/plots/cascade_eval_all.png
        qwen4b_uncertainty/plots/cascade_eval_focus.png
"""

import json
import math
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
OUT_TSV = BASE / "qwen4b_uncertainty/data/cascade_eval_all.tsv"
PLOT_ALL = BASE / "qwen4b_uncertainty/plots/cascade_eval_all.png"
PLOT_FOCUS = BASE / "qwen4b_uncertainty/plots/cascade_eval_focus.png"

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, nDCG @ 20]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20"]

ALPHA_GLOBAL = 0.825
ALPHA_DYN = {
    "list": 0.875, "summary": 0.800, "yesno": 0.750, "factoid": 0.875,
}
ALPHA_DYN10 = {
    "list": 0.875, "summary": 0.800, "yesno": 0.750, "factoid": 0.925,
}
GATED_PARAMS = {
    "list":    (0.875, 0.8242),
    "summary": (0.800, 0.4246),
    "yesno":   (0.750, 0.6494),
    "factoid": (0.875, 0.0000),
}


def minmax(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def norm_entropy_top_k(vals: np.ndarray, k: int) -> float:
    sub = vals[:k]
    s = sub.sum()
    if s <= 0 or len(sub) < 2:
        return 0.0
    p = sub / s
    p = np.clip(p, 1e-15, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(len(sub)))


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

    qwen_arr, qwen_norm, bm25_norm, docids, qtypes, H20 = {}, {}, {}, {}, {}, {}
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
        H20[qid] = norm_entropy_top_k(q, 20)

    qids = list(qwen_arr.keys())
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    def build_run(score_fn):
        run = []
        for qid in qids:
            s = score_fn(qid)
            for d, sv in zip(docids[qid], s):
                run.append(ScoredDoc(qid, d, float(sv)))
        return run

    def score_pure(qid):
        return qwen_arr[qid]

    def score_global(qid):
        return ALPHA_GLOBAL * qwen_norm[qid] + (1 - ALPHA_GLOBAL) * bm25_norm[qid]

    def score_dyn(qid):
        a = ALPHA_DYN[qtypes[qid]]
        return a * qwen_norm[qid] + (1 - a) * bm25_norm[qid]

    def score_dyn10(qid):
        a = ALPHA_DYN10[qtypes[qid]]
        return a * qwen_norm[qid] + (1 - a) * bm25_norm[qid]

    def score_gated(qid):
        a, tau = GATED_PARAMS[qtypes[qid]]
        if H20[qid] > tau:
            return a * qwen_norm[qid] + (1 - a) * bm25_norm[qid]
        return qwen_arr[qid]

    runs = {
        "pure_qwen":                          build_run(score_pure),
        "qwen4b_linear_fusion":               build_run(score_global),
        "qwen4b_linear_fusion_dynamic":       build_run(score_dyn),
        "qwen4b_linear_fusion_dynamic_10":    build_run(score_dyn10),
        "qwen4b_linear_fusion_dynamic_gated": build_run(score_gated),
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
    print("\n" + "=" * 110)
    print(f"{'config':<40}  {'scope':<8}  " +
          "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    print("=" * 110)
    for cfg in runs:
        for sc in ["global"] + TYPES:
            row = df[(df["config"] == cfg) & (df["scope"] == sc)].iloc[0]
            print(f"{cfg:<40}  {sc:<8}  " +
                  "  ".join(f"{row[m]:>8.4f}" for m in METRIC_NAMES))
        print("-" * 110)

    # ── Helper: grouped bars per scope ───────────────────────────────────────
    def _grouped_bars(cfgs, labels, palette, title, out_path):
        fig, axes = plt.subplots(1, 5, figsize=(28, 5.4), sharey=False)
        n_cfg = len(cfgs)
        bar_w = min(0.85 / n_cfg, 0.18)
        x = np.arange(len(METRIC_NAMES))
        offsets = (np.arange(n_cfg) - (n_cfg - 1) / 2) * bar_w
        for ax, sc in zip(axes, ["global"] + TYPES):
            for i, (cfg, lbl, col) in enumerate(zip(cfgs, labels, palette)):
                row = df[(df["config"] == cfg) & (df["scope"] == sc)].iloc[0]
                vals = [row[m] for m in METRIC_NAMES]
                ax.bar(x + offsets[i], vals, bar_w, color=col, label=lbl,
                       edgecolor="white", alpha=0.92)
                for xi, v in zip(x + offsets[i], vals):
                    ax.text(xi, v + 0.003, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=5.5)
            ax.set_xticks(x)
            ax.set_xticklabels(METRIC_NAMES, fontsize=9)
            ax.set_title(sc, fontweight="bold")
            if ax is axes[0]:
                ax.set_ylabel("nDCG")
            ymax = max(df[METRIC_NAMES].max())
            ax.set_ylim(0.74, ymax + 0.04)
            ax.legend(fontsize=6.5, loc="upper right")
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=160); plt.close(fig)
        print(f"  → {out_path}")

    # ── Plot 1: ALL configs ─────────────────────────────────────────────────
    cfgs_all = list(runs.keys())
    labels_all = [
        "Pure Qwen",
        "Linear (α=0.825)",
        "Linear dynamic (mixed)",
        "Linear dynamic (all @10)",
        "Linear dynamic gated H@20",
    ]
    palette_all = ["#9467bd", "#1f77b4", "#2ca02c", "#dd8452", "#d62728"]
    _grouped_bars(
        cfgs_all, labels_all, palette_all,
        "Qwen3-4B + BM25 fusion — all variants  (500 BioASQ queries)",
        PLOT_ALL,
    )

    # ── Plot 2: focus on Qwen vs dyn vs gated dyn ──────────────────────────
    cfgs_focus = ["pure_qwen", "qwen4b_linear_fusion_dynamic",
                  "qwen4b_linear_fusion_dynamic_gated"]
    labels_focus = ["Pure Qwen", "Linear dynamic (per-type α)",
                    "Linear dynamic gated (per-type α, τ on H@20)"]
    palette_focus = ["#9467bd", "#2ca02c", "#d62728"]
    _grouped_bars(
        cfgs_focus, labels_focus, palette_focus,
        "Pure Qwen vs Dynamic α vs Gated Dynamic α  (500 BioASQ queries)",
        PLOT_FOCUS,
    )


if __name__ == "__main__":
    main()
