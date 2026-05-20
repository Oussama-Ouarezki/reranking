"""Adaptive per-query alpha based on monoT5 H@20 entropy bands.

For each query, alpha is chosen by how far its H@20 exceeds a start threshold tau:

    H@20 <= tau                       → pure monoT5  (no fusion)
    H@20 in (tau,           tau+w]    → alpha = 0.995
    H@20 in (tau+w,   tau+2w]         → alpha = 0.990
    H@20 in (tau+2w,  tau+3w]         → alpha = 0.985
    ...
    floor((H@20 - tau) / w) = k       → alpha = max(alpha_floor, 0.995 - k*0.005)

Parameters swept globally:
    tau        in [0.0 .. 0.9]  (10 values) — where fusion begins
    band_width in [0.05, 0.1, 0.2, 0.5]     — H@20 range per alpha step

Reads:  qwen4b_uncertainty/data/monot5_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/adaptive_alpha_sweep_test.tsv
        qwen4b_uncertainty/data/adaptive_alpha_best_test.json
        qwen4b_uncertainty/plots/adaptive_alpha_sweep_test.png
        qwen4b_uncertainty/plots/adaptive_alpha_heatmap_test.png
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
SCORES_F = BASE / "qwen4b_uncertainty/data/monot5_scores_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV  = BASE / "qwen4b_uncertainty/data/adaptive_alpha_sweep_test.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/adaptive_alpha_best_test.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES  = ["summary", "factoid", "list", "yesno"]
TAUS        = np.round(np.linspace(0.0, 0.9, 10), 2)   # 0.0, 0.1, ..., 0.9
BAND_WIDTHS = [0.05, 0.1, 0.2, 0.5]
ALPHA_START = 0.995
ALPHA_STEP  = 0.005
ALPHA_FLOOR = 0.5

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


def per_query_alpha(h20: float, tau: float, band_width: float) -> float:
    if h20 <= tau:
        return 1.0   # certain: pure monoT5
    k = int((h20 - tau) / band_width)
    return max(ALPHA_FLOOR, ALPHA_START - k * ALPHA_STEP)


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

    monot5_raw:  dict[str, np.ndarray] = {}
    monot5_norm: dict[str, np.ndarray] = {}
    bm25_norm:   dict[str, np.ndarray] = {}
    docids:      dict[str, list[str]]  = {}
    qtypes:      dict[str, str]        = {}
    H20:         dict[str, float]      = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = r["scores"]
        q = np.array([s["monot5_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"]  for s in items], dtype=float)
        monot5_raw[qid]  = q
        monot5_norm[qid] = minmax(q)
        bm25_norm[qid]   = minmax(b)
        docids[qid]      = [s["docid"] for s in items]
        H20[qid]         = norm_entropy_top20(q)

    qids = list(monot5_raw.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # Print H@20 distribution
    h_vals = list(H20.values())
    print(f"\nH@20: min={min(h_vals):.3f}  max={max(h_vals):.3f}  mean={np.mean(h_vals):.3f}")
    for t in TYPES:
        ht = [H20[q] for q in type_qids[t]]
        print(f"  {t:<8}  n={len(ht):3d}  mean={np.mean(ht):.3f}  std={np.std(ht):.3f}")

    # Baselines
    base_pure = {sc: evaluate(
        [ScoredDoc(qid, d, float(s)) for qid in qids
         for d, s in zip(docids[qid], monot5_raw[qid])], qrels, qs)
        for sc, qs in scopes}

    base_always_fuse_995 = {sc: evaluate(
        [ScoredDoc(qid, d, float(s)) for qid in qids
         for d, s in zip(docids[qid],
                         0.995 * monot5_norm[qid] + 0.005 * bm25_norm[qid])],
        qrels, qs) for sc, qs in scopes}

    print("\nBaselines (nDCG@10):")
    print(f"  {'scope':<8}  {'pure_monot5':>12}  {'always_fuse(0.995)':>19}")
    for sc, _ in scopes:
        print(f"  {sc:<8}  {base_pure[sc]['ndcg@10']:>12.4f}  "
              f"{base_always_fuse_995[sc]['ndcg@10']:>19.4f}")

    # Adaptive sweep
    print(f"\nSweeping {len(TAUS)} taus x {len(BAND_WIDTHS)} band_widths = "
          f"{len(TAUS)*len(BAND_WIDTHS)} combos …")
    sweep_rows = []
    for tau in tqdm(TAUS, desc="tau"):
        for bw in BAND_WIDTHS:
            # Show per-query alpha distribution for this config
            alphas_assigned = {qid: per_query_alpha(H20[qid], tau, bw) for qid in qids}
            n_fused = sum(1 for a in alphas_assigned.values() if a < 1.0)
            alpha_vals = [a for a in alphas_assigned.values() if a < 1.0]
            unique_alphas = sorted(set(alphas_assigned.values()), reverse=True)

            run = []
            for qid in qids:
                a = alphas_assigned[qid]
                if a < 1.0:
                    scores = a * monot5_norm[qid] + (1 - a) * bm25_norm[qid]
                else:
                    scores = monot5_raw[qid]
                for d, s in zip(docids[qid], scores):
                    run.append(ScoredDoc(qid, d, float(s)))

            for sc, qs in scopes:
                m = evaluate(run, qrels, qs)
                n_fused_sc = sum(1 for q in qs if alphas_assigned[q] < 1.0)
                mean_alpha_sc = np.mean([alphas_assigned[q] for q in qs])
                sweep_rows.append({
                    "tau": float(tau),
                    "band_width": float(bw),
                    "scope": sc,
                    "n_fused": n_fused_sc,
                    "pct_fused": round(100.0 * n_fused_sc / len(qs), 1),
                    "mean_alpha": round(float(mean_alpha_sc), 4),
                    "n_unique_alphas": len(unique_alphas),
                    **m,
                })

    df = pd.DataFrame(sweep_rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    def best(scope: str, metric: str) -> dict:
        sub = df[df["scope"] == scope]
        return sub.loc[sub[metric].idxmax()].to_dict()

    print("\n" + "=" * 110)
    print("BEST (tau, band_width) per scope — adaptive per-query alpha")
    print(f"alpha = max({ALPHA_FLOOR}, {ALPHA_START} - floor((H@20-tau)/band_width) * {ALPHA_STEP})")
    print("=" * 110)
    print(f"  {'scope':<8}  {'tau*':>5}  {'bw*':>5}  {'%fused':>7}  {'mean_α':>7}  "
          + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δpure_'+m:>11}" for m in METRIC_NAMES))

    results: dict[str, dict] = {}
    for sc, _ in scopes:
        target = TYPE_TARGETS[sc]
        b  = best(sc, target)
        bv = base_pure[sc]
        vals  = "  ".join(f"{b[m]:>8.4f}" for m in METRIC_NAMES)
        dpure = "  ".join(f"{b[m]-bv[m]:>+11.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {b['tau']:>5.2f}  {b['band_width']:>5.2f}  "
              f"{b['pct_fused']:>6.1f}%  {b['mean_alpha']:>7.4f}  {vals}  {dpure}")
        results[sc] = {
            "tau_star":       float(b["tau"]),
            "band_width_star": float(b["band_width"]),
            "pct_fused":      float(b["pct_fused"]),
            "mean_alpha":     float(b["mean_alpha"]),
            "target_metric":  target,
            "metrics":        {m: float(b[m]) for m in METRIC_NAMES},
            "baseline_pure":  {m: float(bv[m]) for m in METRIC_NAMES},
            "delta_vs_pure":  {m: float(b[m] - bv[m]) for m in METRIC_NAMES},
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Also print full grid for global scope (nDCG@10)
    print("\n--- Global nDCG@10 grid (tau × band_width) ---")
    grid = df[df["scope"] == "global"].pivot_table(
        index="tau", columns="band_width", values="ndcg@10"
    )
    print(grid.to_string(float_format="{:.4f}".format))

    # Also print per-query alpha breakdown for best global config
    best_g = results["global"]
    tau_g, bw_g = best_g["tau_star"], best_g["band_width_star"]
    alpha_counts: dict[float, int] = defaultdict(int)
    for qid in qids:
        a = per_query_alpha(H20[qid], tau_g, bw_g)
        alpha_counts[a] += 1
    print(f"\nPer-query alpha distribution (tau={tau_g}, bw={bw_g}):")
    for a in sorted(alpha_counts.keys(), reverse=True):
        label = "no fusion" if a == 1.0 else f"α={a:.3f}"
        bar = "█" * alpha_counts[a]
        print(f"  {label:>12}  {alpha_counts[a]:>3} queries  {bar}")

    # --- Plot 1: nDCG@10 vs tau, one line per band_width, global scope ---
    fig, axes = plt.subplots(1, len(scopes), figsize=(5 * len(scopes), 5))
    bw_colors = {0.05: "#1f77b4", 0.1: "#ff7f0e", 0.2: "#2ca02c", 0.5: "#d62728"}
    for ax, (sc, _) in zip(axes, scopes):
        target = TYPE_TARGETS[sc]
        sub = df[df["scope"] == sc]
        for bw in BAND_WIDTHS:
            s = sub[sub["band_width"] == bw].sort_values("tau")
            ax.plot(s["tau"], s[target], marker="o", ms=4, lw=1.8,
                    color=bw_colors[bw], label=f"bw={bw}")
        ax.axhline(base_pure[sc][target], color="gray", ls=":", lw=1.0, label="pure monoT5")
        ax.axhline(base_always_fuse_995[sc][target], color="black", ls="--",
                   lw=1.0, label="always α=0.995")
        b = results[sc]
        ax.axvline(b["tau_star"], color="red", lw=1.4, ls="-",
                   label=f"τ*={b['tau_star']:.2f} bw={b['band_width_star']:.2f}")
        ax.set_xlabel("τ (fusion start threshold)")
        ax.set_ylabel(target)
        ax.set_title(f"{sc}  (target: {target})", fontweight="bold")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Adaptive alpha: α={ALPHA_START}→{ALPHA_START-ALPHA_STEP}→{ALPHA_START-2*ALPHA_STEP}… per query\n"
        "monoT5 + BM25 — Task13BGoldenEnriched test set",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "adaptive_alpha_sweep_test.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"\n  → {p}")

    # --- Plot 2: heatmap tau × band_width for each metric, global scope ---
    fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(5 * len(METRIC_NAMES), 4))
    for ax, m in zip(axes, METRIC_NAMES):
        pivot = df[df["scope"] == "global"].pivot_table(
            index="tau", columns="band_width", values=m
        )
        sns.heatmap(pivot, ax=ax, cmap="viridis", annot=True, fmt=".4f",
                    cbar_kws={"label": m}, annot_kws={"size": 7})
        b_m = df[(df["scope"] == "global")].loc[
            df[df["scope"] == "global"][m].idxmax()
        ]
        ax.set_title(f"Global — {m}\nbest τ={b_m['tau']:.2f} bw={b_m['band_width']:.2f}  "
                     f"{m}={b_m[m]:.4f}", fontweight="bold", fontsize=9)
        ax.set_xlabel("band_width")
        ax.set_ylabel("τ")
    fig.suptitle(
        "Adaptive alpha heatmap — global scope\n"
        "monoT5 + BM25, Task13BGoldenEnriched test set",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "adaptive_alpha_heatmap_test.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
