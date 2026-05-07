"""Uncertainty-aware (alpha, tau) sweep per query type — Task13BGoldenEnriched test set.

H@20 = normalised Shannon entropy of top-20 Qwen probabilities.
Gate: fuse with BM25 only when H@20 > tau, else keep pure Qwen.

    if H@20(query) > tau:
        score(d) = alpha * qwen_norm(d) + (1-alpha) * bm25_norm(d)
    else:
        score(d) = qwen_prob(d)

Stage 1 — best alpha only (always fuse, no gate)    [from script 23, reproduced]
Stage 2 — joint (alpha, tau) sweep with H@20 gate

Per-scope targets:
    global  -> nDCG@10
    summary -> nDCG@10
    factoid -> nDCG@5
    list    -> nDCG@3
    yesno   -> nDCG@1

Reads:  qwen4b_uncertainty/data/qwen_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/alpha_tau_test.tsv
        qwen4b_uncertainty/data/alpha_tau_best_test.json
        qwen4b_uncertainty/plots/alpha_tau_heatmap_test.png
        qwen4b_uncertainty/plots/alpha_tau_curves_test.png
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
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV  = BASE / "qwen4b_uncertainty/data/alpha_tau_test.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/alpha_tau_best_test.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES  = ["summary", "factoid", "list", "yesno"]
ALPHAS = np.round(np.linspace(0.0, 1.0, 41), 3)   # 0.000 .. 1.000, step 0.025
TAUS   = np.round(np.linspace(0.0, 1.0, 41), 3)   # normalised entropy, step 0.025

METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]

TYPE_TARGETS = {
    "global":  "ndcg@10",
    "summary": "ndcg@10",
    "factoid": "ndcg@5",
    "list":    "ndcg@3",
    "yesno":   "ndcg@1",
}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def norm_entropy_top20(vals: np.ndarray) -> float:
    top = np.sort(vals)[::-1][:20]
    s = top.sum()
    if s <= 0 or len(top) < 2:
        return 0.0
    p = top / s
    p = np.clip(p, 1e-15, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(len(top)))


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def evaluate(run: list[ScoredDoc], qrels: list[Qrel], qid_set: set[str]) -> dict[str, float]:
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q   = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main() -> None:
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"{len(rows)} queries loaded")

    qrels = load_qrels()
    print(f"{len(qrels)} qrel rows")

    qwen_raw:  dict[str, np.ndarray] = {}
    qwen_norm: dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    docids:    dict[str, list[str]]  = {}
    qtypes:    dict[str, str]        = {}
    H20:       dict[str, float]      = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_raw[qid]  = q
        qwen_norm[qid] = minmax(q)
        bm25_norm[qid] = minmax(b)
        docids[qid]    = [s["docid"] for s in items]
        H20[qid]       = norm_entropy_top20(q)

    qids = list(qwen_raw.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    h_vals = list(H20.values())
    print(f"\nH@20 range: {min(h_vals):.3f} .. {max(h_vals):.3f}  "
          f"(mean={np.mean(h_vals):.3f}, max_possible=1.0)")
    for t in TYPES:
        ht = [H20[q] for q in type_qids[t]]
        print(f"  {t:<8}  n={len(ht):3d}  H@20 mean={np.mean(ht):.3f}  "
              f"std={np.std(ht):.3f}  min={min(ht):.3f}  max={max(ht):.3f}")

    # Pure-Qwen baseline
    base_run = [ScoredDoc(qid, d, float(s))
                for qid in qids
                for d, s in zip(docids[qid], qwen_raw[qid])]
    base = {sc: evaluate(base_run, qrels, qs) for sc, qs in scopes}

    print("\nPure Qwen baseline:")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    for sc, _ in scopes:
        print(f"  {sc:<8}  " + "  ".join(f"{base[sc][m]:>8.4f}" for m in METRIC_NAMES))

    # Stage 1: alpha only (tau=0, always fuse)
    print(f"\n=== Stage 1: alpha sweep (no gate) — {len(ALPHAS)} values ===")
    rows_s1 = []
    for alpha in tqdm(ALPHAS, desc="alpha"):
        run = [ScoredDoc(qid, d, float(s))
               for qid in qids
               for d, s in zip(
                   docids[qid],
                   alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid],
               )]
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            rows_s1.append({"stage": 1, "scope": sc, "alpha": float(alpha),
                            "tau": float("nan"), **m})

    # Stage 2: joint (alpha, tau) with H@20 gate
    print(f"\n=== Stage 2: (alpha, tau) sweep with H@20 gate — "
          f"{len(ALPHAS)}x{len(TAUS)}={len(ALPHAS)*len(TAUS)} combos ===")
    rows_s2 = []
    for alpha in tqdm(ALPHAS, desc="alpha x tau"):
        for tau in TAUS:
            run = []
            for qid in qids:
                if H20[qid] > tau:
                    scores = alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid]
                else:
                    scores = qwen_raw[qid]
                for d, s in zip(docids[qid], scores):
                    run.append(ScoredDoc(qid, d, float(s)))
            for sc, qs in scopes:
                m = evaluate(run, qrels, qs)
                rows_s2.append({"stage": 2, "scope": sc,
                                "alpha": float(alpha), "tau": float(tau), **m})

    df = pd.DataFrame(rows_s1 + rows_s2)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    def best_s1(scope, metric):
        sub = df[(df["stage"] == 1) & (df["scope"] == scope)]
        return sub.loc[sub[metric].idxmax()].to_dict()

    def best_s2(scope, metric):
        sub = df[(df["stage"] == 2) & (df["scope"] == scope)]
        return sub.loc[sub[metric].idxmax()].to_dict()

    print("\n" + "=" * 100)
    print("STAGE 1 — Best alpha (always fuse, no gate)")
    print("=" * 100)
    print(f"  {'scope':<8}  {'target':<9}  {'alpha*':>7}  "
          + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δ'+m:>9}" for m in METRIC_NAMES))
    s1_results = {}
    for sc, _ in scopes:
        target = TYPE_TARGETS[sc]
        b = best_s1(sc, target)
        bv = base[sc]
        vals   = "  ".join(f"{b[m]:>8.4f}" for m in METRIC_NAMES)
        deltas = "  ".join(f"{b[m]-bv[m]:>+9.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {target:<9}  {b['alpha']:>7.3f}  {vals}  {deltas}")
        s1_results[sc] = b

    print("\n" + "=" * 100)
    print("STAGE 2 — Best (alpha, tau) with H@20 gate")
    print("=" * 100)
    print(f"  {'scope':<8}  {'target':<9}  {'alpha*':>7}  {'tau*':>6}  {'%fused':>7}  "
          + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δvs_qwen':>9}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δvs_s1':>8}" for m in METRIC_NAMES))

    s2_results = {}
    for sc, qs in scopes:
        target = TYPE_TARGETS[sc]
        b  = best_s2(sc, target)
        bv = base[sc]
        s1 = s1_results[sc]
        n_fused = sum(1 for q in qs if H20[q] > b["tau"])
        pct = 100.0 * n_fused / len(qs)
        vals    = "  ".join(f"{b[m]:>8.4f}" for m in METRIC_NAMES)
        dqwen   = "  ".join(f"{b[m]-bv[m]:>+9.4f}" for m in METRIC_NAMES)
        ds1     = "  ".join(f"{b[m]-s1[m]:>+8.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {target:<9}  {b['alpha']:>7.3f}  {b['tau']:>6.3f}  "
              f"{pct:>6.1f}%  {vals}  {dqwen}  {ds1}")
        s2_results[sc] = {
            "alpha_star":    float(b["alpha"]),
            "tau_star":      float(b["tau"]),
            "pct_fused":     round(pct, 1),
            "target_metric": target,
            "metrics":   {m: float(b[m]) for m in METRIC_NAMES},
            "baseline":  {m: float(bv[m]) for m in METRIC_NAMES},
            "delta_vs_qwen": {m: float(b[m] - bv[m]) for m in METRIC_NAMES},
            "delta_vs_s1":   {m: float(b[m] - s1[m]) for m in METRIC_NAMES},
            "stage1_alpha":  float(s1["alpha"]),
        }

    out = {"stage1_no_gate": {sc: {
               "alpha_star": float(s1_results[sc]["alpha"]),
               "target_metric": TYPE_TARGETS[sc],
               "metrics": {m: float(s1_results[sc][m]) for m in METRIC_NAMES},
               "baseline": {m: float(base[sc][m]) for m in METRIC_NAMES},
           } for sc, _ in scopes},
           "stage2_h20_gate": s2_results}
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # --- Plot 1: Stage 1 alpha curves per scope ---
    n = len(scopes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    pal = {"ndcg@1": "#e377c2", "ndcg@3": "#4c72b0",
           "ndcg@5": "#2ca02c", "ndcg@10": "#1f77b4"}
    for ax, (sc, _) in zip(axes, scopes):
        sub = df[(df["stage"] == 1) & (df["scope"] == sc)].sort_values("alpha")
        for m in METRIC_NAMES:
            ax.plot(sub["alpha"], sub[m], marker="o", ms=2.5, lw=1.6, color=pal[m], label=m)
            ax.axhline(base[sc][m], color=pal[m], ls=":", lw=0.8, alpha=0.5)
        target = TYPE_TARGETS[sc]
        a_star = s2_results[sc]["alpha_star"]
        t_star = s2_results[sc]["tau_star"]
        ax.axvline(s1_results[sc]["alpha"], color="orange", lw=1.4, ls="--",
                   label=f"S1 α*={s1_results[sc]['alpha']:.3f}")
        ax.axvline(a_star, color="red", lw=1.4, ls="-",
                   label=f"S2 α*={a_star:.3f}\nτ*={t_star:.3f}")
        ax.set_xlabel("α  (Qwen weight)")
        ax.set_ylabel("nDCG")
        ax.set_title(f"{sc}  (target: {target})", fontweight="bold")
        ax.legend(fontsize=7)
    fig.suptitle(
        "Stage 1 (no gate) α curves — Task13BGoldenEnriched test set\n"
        "Orange = best α (no gate)   Red = best α from Stage 2 (H@20 gate)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "alpha_tau_curves_test.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")

    # --- Plot 2: Stage 2 heatmaps (alpha x tau) per scope ---
    n_metrics = len(METRIC_NAMES)
    fig, axes = plt.subplots(n_metrics, n, figsize=(5 * n, 4.5 * n_metrics))
    for col, (sc, _) in enumerate(scopes):
        for row, m in enumerate(METRIC_NAMES):
            ax = axes[row][col]
            sub = df[(df["stage"] == 2) & (df["scope"] == sc)]
            pivot = sub.pivot(index="alpha", columns="tau", values=m)
            sns.heatmap(pivot, ax=ax, cmap="viridis", cbar_kws={"label": m},
                        xticklabels=8, yticklabels=4)
            b = best_s2(sc, m)
            ax.set_title(f"{sc} — {m}\nα*={b['alpha']:.3f}  τ*={b['tau']:.3f}  "
                         f"{m}={b[m]:.4f}", fontsize=9, fontweight="bold")
            ax.set_xlabel("τ  (H@20 threshold)" if row == n_metrics - 1 else "")
            ax.set_ylabel("α  (Qwen weight)" if col == 0 else "")
    fig.suptitle(
        "Stage 2 — (α × τ) heatmaps with H@20 uncertainty gate\n"
        "Task13BGoldenEnriched test set",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "alpha_tau_heatmap_test.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
