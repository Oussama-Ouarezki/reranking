"""Gated dynamic linear fusion — fix per-type α, sweep entropy threshold τ.

Per-type α is the mixed-target dynamic config (each type's natural metric):
    list    α=0.875  (target nDCG@3)
    summary α=0.800  (target nDCG@10)
    yesno   α=0.750  (target nDCG@1)
    factoid α=0.875  (target nDCG@5)

Gate: when H@K(query) > τ → fuse with linear (per-type α);
      when H@K(query) ≤ τ → keep pure Qwen.

Stage 1 — Global τ comparison: H@20 vs H@50
    For each K ∈ {20, 50}, sweep τ globally and pick the τ that maximises
    GLOBAL nDCG@10. Compare H@20 vs H@50 to choose K.

Stage 2 — Per-type τ on the winning K
    Optimise τ per query type, each against that type's target metric:
        list→nDCG@3, summary→nDCG@10, yesno→nDCG@1, factoid→nDCG@5

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/gated_dynamic_grid.tsv
        qwen4b_uncertainty/data/gated_dynamic_params.json
        qwen4b_uncertainty/plots/gated_dynamic_global.png      (Stage 1)
        qwen4b_uncertainty/plots/gated_dynamic_per_type.png    (Stage 2)
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
OUT_TSV = BASE / "qwen4b_uncertainty/data/gated_dynamic_grid.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/gated_dynamic_params.json"
PLOTS = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]

# Mixed-target per-type α (each type's natural metric)
ALPHA_BY_TYPE = {
    "list":    0.875,
    "summary": 0.800,
    "yesno":   0.750,
    "factoid": 0.875,
}
TYPE_TARGETS = {
    "list":    "ndcg@3",
    "summary": "ndcg@10",
    "yesno":   "ndcg@1",
    "factoid": "ndcg@5",
}
GLOBAL_TARGET = "ndcg@10"

ENT_K_OPTIONS = [20, 50]
TAUS = np.round(np.linspace(0.0, 0.999, 41), 4)


def norm_entropy(vals: np.ndarray) -> float:
    s = vals.sum()
    if s <= 0 or len(vals) < 2:
        return 0.0
    p = vals / s
    p = np.clip(p, 1e-15, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(len(vals)))


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
    H20, H50 = {}, {}

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
        H20[qid] = norm_entropy(q[:20])
        H50[qid] = norm_entropy(q[:50])

    qids = list(qwen_arr.keys())
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]
    H_TABLE = {20: H20, 50: H50}

    # Pure Qwen baseline
    base_run = [ScoredDoc(qid, d, float(s)) for qid in qids
                for d, s in zip(docids[qid], qwen_arr[qid])]
    base = {sc: evaluate(base_run, qrels, qs) for sc, qs in scopes}

    # Always-fuse-dyn baseline (no gate, mixed-target α)
    fuse_always_run = []
    for qid in qids:
        a = ALPHA_BY_TYPE[qtypes[qid]]
        s = a * qwen_norm[qid] + (1 - a) * bm25_norm[qid]
        for d, sv in zip(docids[qid], s):
            fuse_always_run.append(ScoredDoc(qid, d, float(sv)))
    fuse_always = {sc: evaluate(fuse_always_run, qrels, qs) for sc, qs in scopes}

    print("\nBaselines:")
    print(f"  {'scope':<8}  {'metric':<7}  {'pure_qwen':>10}  {'fuse_always':>12}")
    for sc in ["global"] + TYPES:
        for m in METRIC_NAMES:
            print(f"  {sc:<8}  {m:<7}  {base[sc][m]:>10.4f}  {fuse_always[sc][m]:>12.4f}")
        print("  " + "-" * 50)

    # ── Sweep both K and τ ─────────────────────────────────────────────────
    rows_out = []
    print(f"\nSweeping K={ENT_K_OPTIONS} × τ({len(TAUS)}) = "
          f"{len(ENT_K_OPTIONS)*len(TAUS)} configs …")
    for K in ENT_K_OPTIONS:
        H = H_TABLE[K]
        for tau in tqdm(TAUS, desc=f"H@{K}"):
            run = []
            n_fused_by_type = defaultdict(int)
            n_fused_global = 0
            for qid in qids:
                a = ALPHA_BY_TYPE[qtypes[qid]]
                if H[qid] > tau:
                    s = a * qwen_norm[qid] + (1 - a) * bm25_norm[qid]
                    n_fused_by_type[qtypes[qid]] += 1
                    n_fused_global += 1
                else:
                    s = qwen_arr[qid]
                for d, sv in zip(docids[qid], s):
                    run.append(ScoredDoc(qid, d, float(sv)))
            for sc, qs in scopes:
                m = evaluate(run, qrels, qs)
                if sc == "global":
                    n_unc = n_fused_global
                    denom = len(qids)
                else:
                    n_unc = n_fused_by_type[sc]
                    denom = len(type_qids[sc])
                rows_out.append({
                    "K": K, "tau": float(tau), "scope": sc,
                    "n_fused": n_unc,
                    "pct_fused": 100.0 * n_unc / denom,
                    **m,
                })
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    # ── Stage 1: global τ comparison, K ∈ {20, 50}, target = global nDCG@10 ─
    print("\n" + "=" * 90)
    print(f"STAGE 1 — Global τ comparison (target = global {GLOBAL_TARGET})")
    print("=" * 90)
    stage1: dict[int, dict] = {}
    for K in ENT_K_OPTIONS:
        sub = df[(df["K"] == K) & (df["scope"] == "global")]
        best = sub.loc[sub[GLOBAL_TARGET].idxmax()]
        stage1[K] = best.to_dict()
        print(f"  H@{K}:  τ* = {best['tau']:.4f}  "
              f"{GLOBAL_TARGET} = {best[GLOBAL_TARGET]:.4f}  "
              f"(pure Qwen = {base['global'][GLOBAL_TARGET]:.4f}, "
              f"fuse_always = {fuse_always['global'][GLOBAL_TARGET]:.4f})  "
              f"%fused = {best['pct_fused']:.1f}%")
    K_winner = max(ENT_K_OPTIONS, key=lambda K: stage1[K][GLOBAL_TARGET])
    print(f"\n  WINNER: H@{K_winner}  "
          f"({GLOBAL_TARGET} = {stage1[K_winner][GLOBAL_TARGET]:.4f})")

    # ── Stage 2: per-type τ on winner K, target = per-type metric ──────────
    print("\n" + "=" * 90)
    print(f"STAGE 2 — Per-type τ on H@{K_winner}, each at its target metric")
    print("=" * 90)
    stage2: dict[str, dict] = {}
    for t in TYPES:
        target = TYPE_TARGETS[t]
        sub = df[(df["K"] == K_winner) & (df["scope"] == t)]
        best = sub.loc[sub[target].idxmax()].to_dict()
        stage2[t] = best
        bv = base[t][target]
        fv = fuse_always[t][target]
        print(f"  {t:<8}  target={target:<7}  τ*={best['tau']:.4f}  "
              f"value={best[target]:.4f}  "
              f"(pure Qwen={bv:.4f}, fuse_always={fv:.4f})  "
              f"%fused={best['pct_fused']:.1f}%")

    # ── Save params JSON ────────────────────────────────────────────────────
    params = {
        "alpha_by_type": ALPHA_BY_TYPE,
        "type_targets":  TYPE_TARGETS,
        "stage1_global": {
            f"H@{K}": {"tau": float(stage1[K]["tau"]),
                       "ndcg@10": float(stage1[K][GLOBAL_TARGET]),
                       "pct_fused": float(stage1[K]["pct_fused"])}
            for K in ENT_K_OPTIONS
        },
        "K_winner": K_winner,
        "stage2_per_type": {
            t: {
                "K": K_winner,
                "tau": float(stage2[t]["tau"]),
                "alpha": ALPHA_BY_TYPE[t],
                "target_metric": TYPE_TARGETS[t],
                "metric_value": float(stage2[t][TYPE_TARGETS[t]]),
                "pct_fused": float(stage2[t]["pct_fused"]),
            }
            for t in TYPES
        },
    }
    OUT_JSON.write_text(json.dumps(params, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # ── Plot 1: Stage 1 — global τ curves for H@20 and H@50, 4 metrics ─────
    fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(6 * len(METRIC_NAMES), 5),
                              sharey=False)
    palette = {20: "#bcbd22", 50: "#8c564b"}
    for ax, m in zip(axes, METRIC_NAMES):
        for K in ENT_K_OPTIONS:
            sub = df[(df["K"] == K) & (df["scope"] == "global")].sort_values("tau")
            ax.plot(sub["tau"], sub[m], marker="o", ms=3, lw=1.8,
                    color=palette[K], label=f"H@{K}")
            best = stage1[K]
            ax.axvline(best["tau"], color=palette[K], ls="--", lw=1.2, alpha=0.7,
                       label=f"τ*(H@{K})={best['tau']:.3f}")
        ax.axhline(base["global"][m], color="gray", ls=":", lw=1.0,
                   label=f"pure Qwen ({base['global'][m]:.4f})")
        ax.axhline(fuse_always["global"][m], color="black", ls=":", lw=1.0,
                   label=f"fuse_always ({fuse_always['global'][m]:.4f})")
        ax.set_xlabel("τ")
        ax.set_ylabel(m)
        title_marker = "  ★ Stage 1 target" if m == GLOBAL_TARGET else ""
        ax.set_title(f"Global — {m}{title_marker}", fontweight="bold")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"Stage 1 — Global τ comparison H@20 vs H@50  (winner = H@{K_winner})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "gated_dynamic_global.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")

    # ── Plot 2: Stage 2 — per-type τ curves (winner K only) ────────────────
    fig, axes = plt.subplots(1, len(TYPES), figsize=(6 * len(TYPES), 5))
    for ax, t in zip(axes, TYPES):
        target = TYPE_TARGETS[t]
        sub = df[(df["K"] == K_winner) & (df["scope"] == t)].sort_values("tau")
        for m in METRIC_NAMES:
            lw = 2.4 if m == target else 1.0
            alpha = 1.0 if m == target else 0.45
            ax.plot(sub["tau"], sub[m], marker="o", ms=2.5, lw=lw,
                    alpha=alpha, label=f"{m}{' (target)' if m == target else ''}")
        # mark chosen τ*
        best = stage2[t]
        ax.axvline(best["tau"], color="red", ls="-", lw=1.5,
                   label=f"τ*={best['tau']:.3f}")
        ax.axhline(base[t][target], color="gray", ls=":", lw=1.0,
                   label=f"pure Qwen ({base[t][target]:.4f})")
        ax.axhline(fuse_always[t][target], color="black", ls=":", lw=1.0,
                   label=f"fuse_always ({fuse_always[t][target]:.4f})")
        ax.set_xlabel("τ")
        ax.set_ylabel(target)
        ax.set_title(f"{t}  (target {target}, α={ALPHA_BY_TYPE[t]:.3f})",
                     fontweight="bold")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"Stage 2 — Per-type τ on H@{K_winner}  (each type optimised for its target)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "gated_dynamic_per_type.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
