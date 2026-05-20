"""H@20 vs H@50 threshold search — side-by-side per-type comparison.

Per-type α fixed (mixed-target dynamic):
    list α=0.875   summary α=0.800   yesno α=0.750   factoid α=0.875

Gate: when H@K > τ → fuse (linear, per-type α); else keep pure Qwen.

For each (K ∈ {20, 50}, query type), find τ that maximises the type's target:
    list→nDCG@3, summary→nDCG@10, yesno→nDCG@1, factoid→nDCG@5
Also report the global optimum at nDCG@10.

Reads:  qwen4b_uncertainty/data/gated_dynamic_grid.tsv  (precomputed sweep)
Writes: qwen4b_uncertainty/data/h20_vs_h50_best.tsv
        qwen4b_uncertainty/plots/h20_vs_h50_per_type.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE = Path(__file__).resolve().parents[1]
GRID_F = BASE / "qwen4b_uncertainty/data/gated_dynamic_grid.tsv"
OUT_TSV = BASE / "qwen4b_uncertainty/data/h20_vs_h50_best.tsv"
PLOT = BASE / "qwen4b_uncertainty/plots/h20_vs_h50_per_type.png"

TYPES = ["summary", "factoid", "list", "yesno"]
TYPE_TARGETS = {
    "list":    "ndcg@3",
    "summary": "ndcg@10",
    "yesno":   "ndcg@1",
    "factoid": "ndcg@5",
}
GLOBAL_TARGET = "ndcg@10"
ENT_K_OPTIONS = [20, 50]


def main():
    df = pd.read_csv(GRID_F, sep="\t")
    print(f"loaded {len(df)} rows from {GRID_F}")

    # ── For each (K, scope), find best τ at scope's target ────────────────
    rows_out = []
    print("\n" + "=" * 110)
    print(f"{'scope':<8}  {'target':<8}  "
          f"{'K':>3}  {'τ*':>7}  {'value':>8}  {'pct_fused':>10}  "
          f"{'ndcg@1':>8}  {'ndcg@3':>8}  {'ndcg@5':>8}  {'ndcg@10':>8}")
    print("=" * 110)
    for sc in ["global"] + TYPES:
        target = GLOBAL_TARGET if sc == "global" else TYPE_TARGETS[sc]
        for K in ENT_K_OPTIONS:
            sub = df[(df["scope"] == sc) & (df["K"] == K)]
            best = sub.loc[sub[target].idxmax()]
            print(f"{sc:<8}  {target:<8}  {K:>3}  "
                  f"{best['tau']:>7.4f}  {best[target]:>8.4f}  "
                  f"{best['pct_fused']:>9.1f}%  "
                  f"{best['ndcg@1']:>8.4f}  {best['ndcg@3']:>8.4f}  "
                  f"{best['ndcg@5']:>8.4f}  {best['ndcg@10']:>8.4f}")
            rows_out.append({
                "scope": sc, "target": target, "K": int(K),
                "tau_star": float(best["tau"]),
                "target_value": float(best[target]),
                "pct_fused": float(best["pct_fused"]),
                "ndcg@1":  float(best["ndcg@1"]),
                "ndcg@3":  float(best["ndcg@3"]),
                "ndcg@5":  float(best["ndcg@5"]),
                "ndcg@10": float(best["ndcg@10"]),
            })
        print("-" * 110)

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # ── Side-by-side: H@20 vs H@50 winner per scope ──────────────────────
    print("\n" + "=" * 80)
    print("H@20 vs H@50 — winner per scope (at scope's target metric)")
    print("=" * 80)
    for sc in ["global"] + TYPES:
        target = GLOBAL_TARGET if sc == "global" else TYPE_TARGETS[sc]
        v20 = out_df[(out_df["scope"] == sc) & (out_df["K"] == 20)].iloc[0]
        v50 = out_df[(out_df["scope"] == sc) & (out_df["K"] == 50)].iloc[0]
        if v20["target_value"] > v50["target_value"]:
            winner, gap = f"H@20", v20["target_value"] - v50["target_value"]
        elif v50["target_value"] > v20["target_value"]:
            winner, gap = f"H@50", v50["target_value"] - v20["target_value"]
        else:
            winner, gap = "tie", 0.0
        print(f"  {sc:<8}  target={target:<7}  "
              f"H@20={v20['target_value']:.4f} (τ={v20['tau_star']:.4f}, "
              f"{v20['pct_fused']:.0f}% fused)  "
              f"H@50={v50['target_value']:.4f} (τ={v50['tau_star']:.4f}, "
              f"{v50['pct_fused']:.0f}% fused)  "
              f"→ {winner}{f' (Δ=+{gap:.4f})' if gap > 0 else ''}")

    # ── Plot — per-type τ curves with both K, target metric only ──────────
    fig, axes = plt.subplots(1, len(TYPES) + 1, figsize=(6 * (len(TYPES) + 1), 5))
    palette = {20: "#1f77b4", 50: "#dd8452"}
    for ax, sc in zip(axes, ["global"] + TYPES):
        target = GLOBAL_TARGET if sc == "global" else TYPE_TARGETS[sc]
        for K in ENT_K_OPTIONS:
            sub = df[(df["scope"] == sc) & (df["K"] == K)].sort_values("tau")
            ax.plot(sub["tau"], sub[target], marker="o", ms=3, lw=1.8,
                    color=palette[K], label=f"H@{K}")
            best = out_df[(out_df["scope"] == sc) & (out_df["K"] == K)].iloc[0]
            ax.axvline(best["tau_star"], color=palette[K], ls="--", lw=1.2,
                       alpha=0.7,
                       label=f"τ*(H@{K})={best['tau_star']:.3f}  "
                             f"({target}={best['target_value']:.4f})")
        ax.set_xlabel("τ")
        ax.set_ylabel(target)
        title = sc if sc == "global" else f"{sc}  (α={'0.875' if sc in ('list','factoid') else ('0.800' if sc=='summary' else '0.750')})"
        ax.set_title(f"{title}  (target {target})", fontweight="bold")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "H@20 vs H@50 threshold search — gate fires when H@K > τ "
        "(per-type α fixed, target metric per type)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150); plt.close(fig)
    print(f"\n  → {PLOT}")


if __name__ == "__main__":
    main()
