"""Compare score-normalisation schemes inside the gated dynamic cascade.

Cascade (fixed): per-type (α, τ) on H@20 entropy.
    list (α=0.875, τ=0.8242)   summary (α=0.800, τ=0.4246)
    yesno (α=0.750, τ=0.6494)  factoid (α=0.875, τ=0.0)

When the gate triggers (H@20 > τ), fuse:
    score(d) = α · qwen_T(d) + (1 − α) · bm25_T(d)
where T is one of:
    raw         — no transform                                 (qwen ∈ [0,1], bm25 ∈ raw Lucene)
    sigmoid     — sigmoid of per-query z-score                 (centered around 0.5)
    softmax     — per-query softmax over 50 values             (sums to 1)
    minmax      — per-query (x − min) / (max − min)            (current default)
    standardize — per-query z-score (mean=0, std=1)

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/norm_comparison.tsv
        qwen4b_uncertainty/plots/norm_comparison.png
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
OUT_TSV = BASE / "qwen4b_uncertainty/data/norm_comparison.tsv"
PLOT = BASE / "qwen4b_uncertainty/plots/norm_comparison.png"

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, nDCG @ 20]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20"]

GATED_PARAMS = {
    "list":    (0.875, 0.8242),
    "summary": (0.800, 0.4246),
    "yesno":   (0.750, 0.6494),
    "factoid": (0.875, 0.0000),
}

NORMS = ["raw", "sigmoid", "softmax", "minmax", "standardize"]


def t_raw(x):       return x

def t_minmax(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)

def t_softmax(x):
    z = x - x.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0:
        return np.zeros_like(x)
    return e / s

def t_standardize(x):
    mu, sigma = x.mean(), x.std()
    if sigma < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sigma

def t_sigmoid(x):
    # sigmoid of per-query z-score → [0,1] with monotonic mapping centered at 0.5
    z = t_standardize(x)
    return 1.0 / (1.0 + np.exp(-z))

NORM_FN = {
    "raw":         t_raw,
    "sigmoid":     t_sigmoid,
    "softmax":     t_softmax,
    "minmax":      t_minmax,
    "standardize": t_standardize,
}


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

    qwen_arr, bm25_arr, docids, qtypes, H20 = {}, {}, {}, {}, {}
    qwen_T = {n: {} for n in NORMS}
    bm25_T = {n: {} for n in NORMS}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = sorted(r["scores"], key=lambda s: s["qwen_prob"], reverse=True)
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_arr[qid] = q
        bm25_arr[qid] = b
        docids[qid] = [s["docid"] for s in items]
        H20[qid] = norm_entropy_top_k(q, 20)
        for n in NORMS:
            qwen_T[n][qid] = NORM_FN[n](q)
            bm25_T[n][qid] = NORM_FN[n](b)

    qids = list(qwen_arr.keys())
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # Pure Qwen baseline (for reference)
    base_run = [ScoredDoc(qid, d, float(s)) for qid in qids
                for d, s in zip(docids[qid], qwen_arr[qid])]
    base = {sc: evaluate(base_run, qrels, qs) for sc, qs in scopes}

    rows_out = []
    for n in NORMS:
        run = []
        for qid in qids:
            a, tau = GATED_PARAMS[qtypes[qid]]
            if H20[qid] > tau:
                s = a * qwen_T[n][qid] + (1 - a) * bm25_T[n][qid]
            else:
                s = qwen_arr[qid]
            for d, sv in zip(docids[qid], s):
                run.append(ScoredDoc(qid, d, float(sv)))
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            rows_out.append({"norm": n, "scope": sc, **m})

    # Add pure-Qwen as a reference row
    for sc in ["global"] + TYPES:
        rows_out.append({"norm": "pure_qwen", "scope": sc, **base[sc]})

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # ── Print table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 95)
    print(f"{'norm':<13}  {'scope':<8}  " +
          "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    print("=" * 95)
    order = ["pure_qwen"] + NORMS
    for n in order:
        for sc in ["global"] + TYPES:
            row = df[(df["norm"] == n) & (df["scope"] == sc)].iloc[0]
            print(f"{n:<13}  {sc:<8}  " +
                  "  ".join(f"{row[m]:>8.4f}" for m in METRIC_NAMES))
        print("-" * 95)

    # ── Winners per (scope, metric) ──────────────────────────────────────────
    print("\n" + "=" * 95)
    print("WINNERS — best norm per (scope, metric); ties broken alphabetically")
    print("=" * 95)
    print(f"{'scope':<8}  " + "  ".join(f"{m:>22}" for m in METRIC_NAMES))
    win_count = defaultdict(int)
    for sc in ["global"] + TYPES:
        cells = []
        for m in METRIC_NAMES:
            sub = df[(df["scope"] == sc) & (df["norm"].isin(NORMS))]
            best_val = sub[m].max()
            best_norms = sorted(sub.loc[sub[m] == best_val, "norm"].tolist())
            for n in best_norms:
                win_count[n] += 1
            cells.append(f"{','.join(best_norms)} ({best_val:.4f})")
        print(f"{sc:<8}  " + "  ".join(f"{c:>22}" for c in cells))

    print("\nWin counts (across 5 scopes × 4 metrics = 20 cells, ties give multiple wins):")
    for n in sorted(win_count, key=lambda x: -win_count[x]):
        print(f"  {n:<13}  {win_count[n]} / 20")

    overall_winner = max(win_count, key=lambda n: win_count[n])
    print(f"\nOVERALL WINNER: {overall_winner}  ({win_count[overall_winner]}/20 cells)")

    # ── Plot — grouped bars per scope ───────────────────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.4), sharey=False)
    cfgs = ["pure_qwen"] + NORMS
    labels = ["Pure Qwen", "raw", "sigmoid", "softmax", "minmax", "standardize"]
    palette = ["#9467bd", "#7f7f7f", "#bcbd22", "#dd8452", "#1f77b4", "#2ca02c"]
    bar_w = 0.13
    x = np.arange(len(METRIC_NAMES))
    offsets = (np.arange(len(cfgs)) - (len(cfgs) - 1) / 2) * bar_w
    for ax, sc in zip(axes, ["global"] + TYPES):
        for i, (cfg, lbl, col) in enumerate(zip(cfgs, labels, palette)):
            row = df[(df["norm"] == cfg) & (df["scope"] == sc)].iloc[0]
            vals = [row[m] for m in METRIC_NAMES]
            ax.bar(x + offsets[i], vals, bar_w, color=col, label=lbl,
                   edgecolor="white", alpha=0.92)
            for xi, v in zip(x + offsets[i], vals):
                ax.text(xi, v + 0.003, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_NAMES, fontsize=9)
        ax.set_title(sc, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("nDCG")
        ymax = df[METRIC_NAMES].max().max()
        ax.set_ylim(0.74, ymax + 0.04)
        ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle(
        "Score-normalisation comparison — gated dynamic cascade (per-type α, τ on H@20)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160); plt.close(fig)
    print(f"\n  → {PLOT}")


if __name__ == "__main__":
    main()
