"""H@50-only optimisation for nDCG@1 and nDCG@5.

Two-stage analysis per scope (global + per query type):

Stage 1 — Best linear fusion α (always fuse, no gating)
    score = α · qwen_norm + (1-α) · bm25_norm   for every query
    Find α* that maximises nDCG@1 (and separately @5) per scope.

Stage 2 — Best (α, τ) with H@50 gate
    fuse only when H@50(query) > τ, otherwise keep pure Qwen.
    Find joint optimum (α, τ) per scope per metric.

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/h50_optim.tsv
        qwen4b_uncertainty/plots/h50_alpha_only.png
        qwen4b_uncertainty/plots/h50_alpha_tau_global.png
        qwen4b_uncertainty/plots/h50_alpha_tau_by_type.png
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import ir_measures
from ir_measures import nDCG, Qrel, ScoredDoc

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores.jsonl"
QRELS_F = BASE / "data/bioasq/processed/qrels.tsv"
OUT_TSV = BASE / "qwen4b_uncertainty/data/h50_optim.tsv"
PLOTS = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10"]

ALPHAS = np.round(np.linspace(0.0, 1.0, 21), 2)            # 0.00 .. 1.00, step 0.05
TAUS = np.round(np.linspace(0.0, 0.999, 41), 4)            # 0.0 .. 0.999, step ~0.025


def norm_entropy(vals: np.ndarray) -> float:
    s = vals.sum()
    if s <= 0 or len(vals) < 2:
        return 0.0
    p = vals / s
    p = np.clip(p, 1e-15, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(len(vals)))


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

    # Pre-compute per-query: sorted qwen, sorted bm25, docids, qtype, H@50
    qwen_arr: dict[str, np.ndarray] = {}
    bm25_arr: dict[str, np.ndarray] = {}
    qwen_norm: dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    docids: dict[str, list[str]] = {}
    qtypes: dict[str, str] = {}
    H50: dict[str, float] = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = sorted(r["scores"], key=lambda s: s["qwen_prob"], reverse=True)
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_arr[qid] = q
        bm25_arr[qid] = b
        qwen_norm[qid] = minmax(q)
        bm25_norm[qid] = minmax(b)
        docids[qid] = [s["docid"] for s in items]
        H50[qid] = norm_entropy(q[:50])

    qids = list(qwen_arr.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # ── Baseline: pure Qwen ─────────────────────────────────────────────────
    base_run = [ScoredDoc(qid, d, float(s))
                for qid in qids
                for d, s in zip(docids[qid], qwen_arr[qid])]
    base = {sc: evaluate(base_run, qrels, qs) for sc, qs in scopes}
    print("\nBaseline (pure Qwen):")
    print(f"  {'scope':<8}  {'ndcg@1':>8}  {'ndcg@5':>8}  {'ndcg@10':>8}")
    for sc in ["global"] + TYPES:
        print(f"  {sc:<8}  {base[sc]['ndcg@1']:>8.4f}  "
              f"{base[sc]['ndcg@5']:>8.4f}  {base[sc]['ndcg@10']:>8.4f}")

    # ── Stage 1 — sweep α only (always fuse, no gate) ──────────────────────
    print("\n=== Stage 1: linear α only (no gate) ===")
    rows_s1: list[dict] = []
    for alpha in tqdm(ALPHAS, desc="α sweep"):
        run = [ScoredDoc(qid, d, float(s))
               for qid in qids
               for d, s in zip(
                   docids[qid],
                   alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid],
               )]
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            rows_s1.append({"stage": 1, "scope": sc, "alpha": float(alpha),
                            "tau": np.nan, **m})

    # ── Stage 2 — joint sweep over (α, τ) with H@50 gate ───────────────────
    print("\n=== Stage 2: linear α + H@50 gate (fuse when H@50 > τ) ===")
    rows_s2: list[dict] = []
    for alpha in tqdm(ALPHAS, desc="α×τ sweep"):
        for tau in TAUS:
            run: list[ScoredDoc] = []
            for qid in qids:
                if H50[qid] > tau:
                    s = alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid]
                else:
                    s = qwen_arr[qid]
                for d, sv in zip(docids[qid], s):
                    run.append(ScoredDoc(qid, d, float(sv)))
            for sc, qs in scopes:
                m = evaluate(run, qrels, qs)
                rows_s2.append({"stage": 2, "scope": sc,
                                "alpha": float(alpha), "tau": float(tau), **m})

    df = pd.DataFrame(rows_s1 + rows_s2)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # ── Pick optima ─────────────────────────────────────────────────────────
    def best_stage1(scope: str, metric: str) -> dict:
        sub = df[(df["stage"] == 1) & (df["scope"] == scope)]
        return sub.loc[sub[metric].idxmax()].to_dict()

    def best_stage2(scope: str, metric: str) -> dict:
        sub = df[(df["stage"] == 2) & (df["scope"] == scope)]
        return sub.loc[sub[metric].idxmax()].to_dict()

    print("\n" + "=" * 90)
    print("STAGE 1 — Best α (always fuse, no gate)")
    print("=" * 90)
    print(f"{'scope':<8}  {'metric':<7}  {'α*':>6}  {'value':>8}  "
          f"{'baseline':>8}  {'Δ':>7}")
    for sc in ["global"] + TYPES:
        for m in METRIC_NAMES:
            b = best_stage1(sc, m)
            bv = base[sc][m]
            print(f"{sc:<8}  {m:<7}  {b['alpha']:>6.2f}  {b[m]:>8.4f}  "
                  f"{bv:>8.4f}  {b[m]-bv:>+7.4f}")

    print("\n" + "=" * 90)
    print("STAGE 2 — Best (α, τ) with H@50 gate (fuse when H@50 > τ)")
    print("=" * 90)
    print(f"{'scope':<8}  {'metric':<7}  {'α*':>6}  {'τ*':>7}  {'value':>8}  "
          f"{'baseline':>8}  {'Δ_vs_qwen':>10}  {'Δ_vs_stage1':>12}  {'%fused':>6}")
    for sc in ["global"] + TYPES:
        for m in METRIC_NAMES:
            b = best_stage2(sc, m)
            bv = base[sc][m]
            s1 = best_stage1(sc, m)[m]
            n_fused = sum(1 for q in (type_qids[sc] if sc in TYPES else qids)
                          if H50[q] > b["tau"])
            denom = len(type_qids[sc]) if sc in TYPES else len(qids)
            pct = 100.0 * n_fused / denom
            print(f"{sc:<8}  {m:<7}  {b['alpha']:>6.2f}  {b['tau']:>7.4f}  "
                  f"{b[m]:>8.4f}  {bv:>8.4f}  {b[m]-bv:>+10.4f}  "
                  f"{b[m]-s1:>+12.4f}  {pct:>5.1f}%")

    # ── Plot 1: Stage 1 — α curves per scope, two metrics ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    for ax, m in zip(axes, METRIC_NAMES):
        for sc in ["global"] + TYPES:
            sub = df[(df["stage"] == 1) & (df["scope"] == sc)].sort_values("alpha")
            ax.plot(sub["alpha"], sub[m], marker="o", ms=3, lw=1.8, label=sc)
            ax.axhline(base[sc][m], color="gray", ls=":", lw=0.7, alpha=0.5)
        ax.axvline(0.0, color="black", ls="--", lw=0.8, alpha=0.4, label="pure BM25")
        ax.axvline(1.0, color="black", ls="--", lw=0.8, alpha=0.4, label="pure Qwen")
        ax.set_xlabel("α (Qwen weight)")
        ax.set_ylabel(m)
        ax.set_title(f"Stage 1 — linear α only — {m}", fontweight="bold")
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    p = PLOTS / "h50_alpha_only.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── Plot 2: Stage 2 heatmaps — global, three metrics ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    for ax, m in zip(axes, METRIC_NAMES):
        sub = df[(df["stage"] == 2) & (df["scope"] == "global")]
        pivot = sub.pivot(index="alpha", columns="tau", values=m)
        sns.heatmap(pivot, ax=ax, cmap="viridis", cbar_kws={"label": m},
                    xticklabels=8, yticklabels=2)
        b = best_stage2("global", m)
        ax.set_title(f"Global — {m}\nbest α={b['alpha']:.2f} τ={b['tau']:.4f}  "
                     f"{m}={b[m]:.4f}", fontweight="bold")
        ax.set_xlabel("τ (H@50 threshold)")
        ax.set_ylabel("α (Qwen weight)")
    fig.tight_layout()
    p = PLOTS / "h50_alpha_tau_global.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── Plot 3: Stage 2 heatmaps per type, all 3 metrics ───────────────────
    fig, axes = plt.subplots(len(METRIC_NAMES), len(TYPES),
                              figsize=(5 * len(TYPES), 4.5 * len(METRIC_NAMES)))
    for col, t in enumerate(TYPES):
        for row, m in enumerate(METRIC_NAMES):
            ax = axes[row][col]
            sub = df[(df["stage"] == 2) & (df["scope"] == t)]
            pivot = sub.pivot(index="alpha", columns="tau", values=m)
            sns.heatmap(pivot, ax=ax, cmap="viridis",
                        cbar_kws={"label": m}, xticklabels=8, yticklabels=2)
            b = best_stage2(t, m)
            ax.set_title(f"{t} — {m}\nα={b['alpha']:.2f} τ={b['tau']:.4f}  "
                         f"{m}={b[m]:.4f}", fontsize=10, fontweight="bold")
            if row == len(METRIC_NAMES) - 1:
                ax.set_xlabel("τ")
            else:
                ax.set_xlabel("")
            if col == 0:
                ax.set_ylabel("α")
            else:
                ax.set_ylabel("")
    fig.suptitle("Stage 2 — α × τ heatmaps per query type", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = PLOTS / "h50_alpha_tau_by_type.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
