"""Evaluate the two Qwen3-4B + BM25 linear fusion cascades end-to-end.

Reports nDCG@1, @5, @10, @20 globally and per query type for:
  - Pure Qwen baseline (no fusion)
  - Config A: qwen4b_linear_fusion             — α=0.825 for every query
  - Config B: qwen4b_linear_fusion_dynamic     — per-type α
                list=0.875, summary=0.800, yesno=0.750, factoid=0.875

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/cascade_eval.tsv
        qwen4b_uncertainty/plots/cascade_eval.png
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
OUT_TSV = BASE / "qwen4b_uncertainty/data/cascade_eval.tsv"
PLOT = BASE / "qwen4b_uncertainty/plots/cascade_eval.png"

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, nDCG @ 20]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20"]

ALPHA_GLOBAL = 0.825
ALPHA_BY_TYPE = {
    "list":    0.875,
    "summary": 0.800,
    "yesno":   0.750,
    "factoid": 0.875,
}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def load_scores() -> list[dict]:
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate(run: list[ScoredDoc], qrels: list[Qrel],
             qid_set: set[str]) -> dict[str, float]:
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main() -> None:
    rows = load_scores()
    qrels = load_qrels()
    print(f"{len(rows)} queries / {len(qrels)} qrel rows")

    qwen_arr: dict[str, np.ndarray] = {}
    qwen_norm: dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    docids: dict[str, list[str]] = {}
    qtypes: dict[str, str] = {}

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
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    def build_run(alpha_fn) -> list[ScoredDoc]:
        run: list[ScoredDoc] = []
        for qid in qids:
            a = alpha_fn(qid)
            s = a * qwen_norm[qid] + (1 - a) * bm25_norm[qid] if a is not None \
                else qwen_arr[qid]
            for d, sv in zip(docids[qid], s):
                run.append(ScoredDoc(qid, d, float(sv)))
        return run

    runs = {
        "pure_qwen":                       build_run(lambda q: None),
        "qwen4b_linear_fusion":            build_run(lambda q: ALPHA_GLOBAL),
        "qwen4b_linear_fusion_dynamic":    build_run(lambda q: ALPHA_BY_TYPE[qtypes[q]]),
    }

    rows_out: list[dict] = []
    for cfg, run in runs.items():
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            rows_out.append({"config": cfg, "scope": sc, **m})

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # ── Print tables ───────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'config':<32}  {'scope':<8}  " +
          "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    print("=" * 100)
    for cfg in ["pure_qwen", "qwen4b_linear_fusion", "qwen4b_linear_fusion_dynamic"]:
        for sc in ["global"] + TYPES:
            row = df[(df["config"] == cfg) & (df["scope"] == sc)].iloc[0]
            print(f"{cfg:<32}  {sc:<8}  " +
                  "  ".join(f"{row[m]:>8.4f}" for m in METRIC_NAMES))
        print("-" * 100)

    # Δ tables vs pure Qwen
    print("\n" + "=" * 100)
    print("Δ vs pure Qwen baseline")
    print("=" * 100)
    print(f"{'config':<32}  {'scope':<8}  " +
          "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    base = {sc: df[(df["config"] == "pure_qwen") & (df["scope"] == sc)].iloc[0]
            for sc in ["global"] + TYPES}
    for cfg in ["qwen4b_linear_fusion", "qwen4b_linear_fusion_dynamic"]:
        for sc in ["global"] + TYPES:
            row = df[(df["config"] == cfg) & (df["scope"] == sc)].iloc[0]
            deltas = [row[m] - base[sc][m] for m in METRIC_NAMES]
            print(f"{cfg:<32}  {sc:<8}  " +
                  "  ".join(f"{d:>+8.4f}" for d in deltas))
        print("-" * 100)

    # ── Plot — grouped bars per scope, 4 metrics, 3 configs ─────────────────
    fig, axes = plt.subplots(1, 5, figsize=(28, 5), sharey=False)
    bar_w = 0.27
    cfgs = ["pure_qwen", "qwen4b_linear_fusion", "qwen4b_linear_fusion_dynamic"]
    cfg_labels = ["Pure Qwen", "Linear (α=0.825)", "Linear dynamic"]
    palette = ["#9467bd", "#1f77b4", "#2ca02c"]
    x = np.arange(len(METRIC_NAMES))
    for ax, sc in zip(axes, ["global"] + TYPES):
        for i, (cfg, lbl, col) in enumerate(zip(cfgs, cfg_labels, palette)):
            row = df[(df["config"] == cfg) & (df["scope"] == sc)].iloc[0]
            vals = [row[m] for m in METRIC_NAMES]
            ax.bar(x + (i - 1) * bar_w, vals, bar_w, color=col, label=lbl,
                   edgecolor="white", alpha=0.92)
            for xi, v in zip(x + (i - 1) * bar_w, vals):
                ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=0)
        ax.set_xticks(x)
        ax.set_xticklabels(METRIC_NAMES, fontsize=9)
        ax.set_title(sc, fontweight="bold")
        if ax is axes[0]:
            ax.set_ylabel("nDCG")
        ax.set_ylim(0.74, max(df[METRIC_NAMES].max()) + 0.04)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Qwen3-4B + BM25 linear fusion — overall metrics  (500 queries)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=160); plt.close(fig)
    print(f"\n  → {PLOT}")


if __name__ == "__main__":
    main()
