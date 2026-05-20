"""Multi-signal uncertainty sweep — gate Qwen→BM25 fusion on multiple signals.

Mirrors monoDuotgate/grid_search.py. For each query compute uncertainty signals from
Qwen P(yes) over the 50 BM25 candidates:

  GAP    = P(yes)[1] − P(yes)[2]                   uncertain when gap < τ
  MAU@5  = mean p*(1−p) over top-5 P(yes)          uncertain when MAU ≥ τ
  MAU@10 = mean p*(1−p) over top-10 P(yes)         uncertain when MAU ≥ τ
  H@20   = norm. rank-entropy over top-20 P(yes)   uncertain when H ≥ τ
  H@50   = norm. rank-entropy over top-50 P(yes)   uncertain when H ≥ τ

When uncertain → re-rank with linear fusion (α=0.7 Qwen + 0.3 BM25, per-query
min-max normalisation). Else → keep pure Qwen.

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/multi_signal_metrics.tsv
        qwen4b_uncertainty/plots/multi_signal_sweep.png
        qwen4b_uncertainty/plots/multi_signal_pareto.png
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
OUT_TSV = BASE / "qwen4b_uncertainty/data/multi_signal_metrics.tsv"
PLOTS = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ALPHA = 0.7   # winner from 04_entropy_sweep.py (linear fusion weight on Qwen)
TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10"]


# ── grids (mirroring monoDuotgate/grid_search.py) ────────────────────────────
def _grid_gap() -> list[float]:
    return sorted({
        0.0,
        *[round(v, 6) for v in np.logspace(-5, -0.3, 60)],
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0,
        math.inf,
    })


def _grid_mau() -> list[float]:
    return sorted({
        0.0,
        *[round(v, 6) for v in np.linspace(0.001, 0.249, 80)],
        0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25,
        math.inf,
    })


def _grid_ent() -> list[float]:
    return sorted({
        0.0,
        *[round(v, 5) for v in np.linspace(0.05, 0.999, 100)],
        0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99,
        math.inf,
    })


# ── helpers ───────────────────────────────────────────────────────────────────
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


# ── per-query signal computation ────────────────────────────────────────────
def compute_signals(rows: list[dict]) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list[str]],
    dict[str, str], dict[str, float], dict[str, float], dict[str, float],
    dict[str, float], dict[str, float],
]:
    qwen_arr: dict[str, np.ndarray] = {}
    bm25_arr: dict[str, np.ndarray] = {}
    docids: dict[str, list[str]] = {}
    qtypes: dict[str, str] = {}
    gap: dict[str, float] = {}
    mau5: dict[str, float] = {}
    mau10: dict[str, float] = {}
    h20: dict[str, float] = {}
    h50: dict[str, float] = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        # Sort docs by Qwen P(yes) descending.
        items = sorted(r["scores"], key=lambda s: s["qwen_prob"], reverse=True)
        q_sorted = np.array([s["qwen_prob"] for s in items], dtype=float)
        b_sorted = np.array([s["bm25_score"] for s in items], dtype=float)
        d_sorted = [s["docid"] for s in items]

        qwen_arr[qid] = q_sorted
        bm25_arr[qid] = b_sorted
        docids[qid] = d_sorted

        if len(q_sorted) >= 2:
            gap[qid] = float(q_sorted[0] - q_sorted[1])
        v5 = q_sorted[:5]
        v10 = q_sorted[:10]
        mau5[qid] = float(np.mean(v5 * (1 - v5)))
        mau10[qid] = float(np.mean(v10 * (1 - v10)))
        h20[qid] = norm_entropy(q_sorted[:20])
        h50[qid] = norm_entropy(q_sorted[:50])

    return qwen_arr, bm25_arr, docids, qtypes, gap, mau5, mau10, h20, h50


# ── ranking under a gate ────────────────────────────────────────────────────
def build_run(qids: list[str], qwen_arr, bm25_arr, docids,
              uncertain: dict[str, bool]) -> list[ScoredDoc]:
    run: list[ScoredDoc] = []
    for qid in qids:
        q = qwen_arr[qid]
        b = bm25_arr[qid]
        if uncertain[qid]:
            scores = ALPHA * minmax(q) + (1 - ALPHA) * minmax(b)
        else:
            scores = q
        for d, s in zip(docids[qid], scores):
            run.append(ScoredDoc(qid, d, float(s)))
    return run


def evaluate_scope(run: list[ScoredDoc], qrels: list[Qrel],
                   qid_set: set[str]) -> dict[str, float]:
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


# ── main sweep ────────────────────────────────────────────────────────────────
def main() -> None:
    rows = load_scores()
    qrels = load_qrels()
    print(f"{len(rows)} queries / {len(qrels)} qrel rows")

    qwen_arr, bm25_arr, docids, qtypes, gap, mau5, mau10, h20, h50 = compute_signals(rows)
    qids = list(qwen_arr.keys())

    # ── per-signal stats ────────────────────────────────────────────────────
    def pct(vals, label):
        v = sorted(vals)
        ps = [0, 10, 25, 50, 75, 90, 95, 99, 100]
        s = "  ".join(f"p{p}={v[min(int(p/100*len(v)), len(v)-1)]:.4f}" for p in ps)
        print(f"  {label:<8}  mean={np.mean(v):.4f}   {s}")

    print("\nSignal distributions:")
    pct(list(gap.values()), "GAP")
    pct(list(mau5.values()), "MAU@5")
    pct(list(mau10.values()), "MAU@10")
    pct(list(h20.values()), "H@20")
    pct(list(h50.values()), "H@50")

    type_qids: dict[str, set[str]] = defaultdict(set)
    all_qids: set[str] = set(qids)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)

    scopes = [("global", all_qids)] + [(t, type_qids[t]) for t in TYPES]

    SIGNALS = [
        ("GAP",    gap,   _grid_gap(), "lt"),    # uncertain when gap < tau
        ("MAU@5",  mau5,  _grid_mau(), "ge"),
        ("MAU@10", mau10, _grid_mau(), "ge"),
        ("H@20",   h20,   _grid_ent(), "ge"),
        ("H@50",   h50,   _grid_ent(), "ge"),
    ]

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []

    for sig_name, sig, grid, op in SIGNALS:
        for tau in tqdm(grid, desc=sig_name):
            if math.isinf(tau):
                if op == "lt":
                    uncertain = {q: True for q in qids}     # always fuse
                else:
                    uncertain = {q: False for q in qids}    # never fuse
            else:
                if op == "lt":
                    uncertain = {q: sig.get(q, math.inf) < tau for q in qids}
                else:
                    uncertain = {q: sig.get(q, -math.inf) >= tau for q in qids}
            n_unc = sum(uncertain.values())
            run = build_run(qids, qwen_arr, bm25_arr, docids, uncertain)
            for scope_name, qid_set in scopes:
                m = evaluate_scope(run, qrels, qid_set)
                rows_out.append({
                    "signal": sig_name,
                    "tau": tau,
                    "op": op,
                    "scope": scope_name,
                    "n_uncertain": n_unc,
                    "pct_fused": 100.0 * n_unc / len(qids),
                    **m,
                })

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # ── baseline (pure Qwen, never fuse) ────────────────────────────────────
    base_run = build_run(qids, qwen_arr, bm25_arr, docids,
                         {q: False for q in qids})
    base = {sc: evaluate_scope(base_run, qrels, qs) for sc, qs in scopes}
    print("\nBaseline (pure Qwen):")
    for sc in ["global"] + TYPES:
        print(f"  {sc:<8}  ndcg@10={base[sc]['ndcg@10']:.4f}")

    # ── plot 1: per-signal nDCG@10 vs tau, global only ──────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
    g = df[df["scope"] == "global"]
    g_finite = g[np.isfinite(g["tau"])]
    for ax, (sig_name, _, _, _) in zip(axes, SIGNALS):
        sub = g_finite[g_finite["signal"] == sig_name].sort_values("tau")
        ax.plot(sub["tau"], sub["ndcg@10"], marker="o", ms=3, lw=1.8,
                color="#1f77b4", label="ndcg@10")
        ax.plot(sub["tau"], sub["ndcg@5"], marker="s", ms=3, lw=1.4,
                color="#4c72b0", alpha=0.6, label="ndcg@5")
        ax.plot(sub["tau"], sub["ndcg@1"], marker="^", ms=3, lw=1.2,
                color="#e377c2", alpha=0.6, label="ndcg@1")
        ax.axhline(base["global"]["ndcg@10"], color="gray", ls=":",
                   lw=1.2, label=f"pure Qwen ({base['global']['ndcg@10']:.4f})")
        if sig_name == "GAP":
            ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_title(sig_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("τ")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("nDCG", fontsize=11)
    fig.suptitle("Qwen→BM25 fusion gated by uncertainty signals (global, α=0.7)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = PLOTS / "multi_signal_sweep.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── plot 2: Pareto — % fused (cost) vs nDCG@10 ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    colors = {"GAP": "#4c72b0", "MAU@5": "#2ca02c", "MAU@10": "#dd8452",
              "H@20": "#bcbd22", "H@50": "#8c564b"}
    for ax, metric in zip(axes, METRIC_NAMES):
        for sig_name, _, _, _ in SIGNALS:
            sub = g[g["signal"] == sig_name].sort_values("pct_fused")
            ax.plot(sub["pct_fused"], sub[metric], marker="o", ms=2.5, lw=2,
                    color=colors[sig_name], alpha=0.85, label=sig_name)
        ax.axhline(base["global"][metric], color="black", ls=":", lw=1.1,
                   label=f"pure Qwen ({base['global'][metric]:.4f})")
        ax.set_xlabel("% queries with fusion applied")
        ax.set_ylabel(metric)
        ax.set_title(f"Pareto — {metric}  (global)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Cost vs quality: cheap = fewer fused, expensive = always fused",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = PLOTS / "multi_signal_pareto.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── plot 3: per-type best-tau heatmap-style (5 signals × 4 types) ──────
    fig, axes = plt.subplots(len(TYPES), 5, figsize=(24, 4.0 * len(TYPES)),
                              sharey="row")
    for i, t in enumerate(TYPES):
        df_t = df[df["scope"] == t]
        df_t_f = df_t[np.isfinite(df_t["tau"])]
        for j, (sig_name, _, _, _) in enumerate(SIGNALS):
            ax = axes[i][j]
            sub = df_t_f[df_t_f["signal"] == sig_name].sort_values("tau")
            ax.plot(sub["tau"], sub["ndcg@10"], marker="o", ms=2.5, lw=1.8,
                    color=colors[sig_name])
            ax.axhline(base[t]["ndcg@10"], color="gray", ls=":", lw=1.2,
                       label=f"baseline {base[t]['ndcg@10']:.4f}")
            if sig_name == "GAP":
                ax.set_xscale("symlog", linthresh=1e-4)
            if i == 0:
                ax.set_title(sig_name, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{t}\nnDCG@10", fontsize=11)
            if i == len(TYPES) - 1:
                ax.set_xlabel("τ")
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Per-type sweep: nDCG@10 vs τ across signals", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    p = PLOTS / "multi_signal_by_type.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── best per signal × scope ─────────────────────────────────────────────
    print("\nBest τ per signal × scope (by ndcg@10, finite τ only):")
    print(f"{'signal':<8}  {'scope':<8}  {'τ*':<10}  {'ndcg@10':>8}  "
          f"{'baseline':>8}  {'Δ':>7}  {'%fused':>6}")
    finite = df[np.isfinite(df["tau"])]
    for sig_name, _, _, _ in SIGNALS:
        for scope in ["global"] + TYPES:
            sub = finite[(finite["signal"] == sig_name) & (finite["scope"] == scope)]
            if sub.empty:
                continue
            best = sub.loc[sub["ndcg@10"].idxmax()]
            b = base[scope]["ndcg@10"]
            print(f"{sig_name:<8}  {scope:<8}  τ={best['tau']:.5f}  "
                  f"{best['ndcg@10']:>8.4f}  {b:>8.4f}  "
                  f"{best['ndcg@10'] - b:+7.4f}  {best['pct_fused']:>5.1f}%")


if __name__ == "__main__":
    main()
