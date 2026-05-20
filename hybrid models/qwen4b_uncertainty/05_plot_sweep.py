"""Plot nDCG@1/@5/@10 vs tau for global and per-type scopes.

Reads:  qwen4b_uncertainty/data/sweep_metrics.tsv
Writes: qwen4b_uncertainty/plots/sweep_global.png
        qwen4b_uncertainty/plots/sweep_by_type.png
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE = Path(__file__).resolve().parents[1]
TSV = BASE / "qwen4b_uncertainty/data/sweep_metrics.tsv"
PLOTS = BASE / "qwen4b_uncertainty/plots"

METRICS = ["ndcg@1", "ndcg@5", "ndcg@10"]
TYPES = ["summary", "factoid", "list", "yesno"]


def variant_label(row: pd.Series) -> str:
    if row["fusion"] == "linear":
        return f"linear α={row['alpha']:.1f}"
    return "RRF"


def variant_style(label: str) -> dict:
    if label.startswith("linear"):
        return {"linewidth": 2.2, "linestyle": "-"}
    return {"linewidth": 1.4, "linestyle": "--", "color": "black"}


def plot_panel(ax, df_scope: pd.DataFrame, metric: str, baseline: float, title: str) -> None:
    finite = df_scope[np.isfinite(df_scope["tau"])]
    for label, sub in finite.groupby("variant"):
        sub = sub.sort_values("tau")
        ax.plot(sub["tau"], sub[metric], label=label, **variant_style(label))
    ax.axhline(baseline, color="gray", linestyle=":", linewidth=1.2, label="pure Qwen")
    ax.set_xlabel("τ (entropy threshold)")
    ax.set_ylabel(metric)
    ax.set_title(title)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TSV, sep="\t")
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce")
    df["variant"] = df.apply(variant_label, axis=1)
    df["tau"] = pd.to_numeric(df["tau"], errors="coerce")

    # Pure-Qwen baseline = tau = +inf rows (one per scope/variant; metric is identical)
    base_rows = df[~np.isfinite(df["tau"])]
    baselines = (
        base_rows.groupby("scope")[METRICS].mean().to_dict(orient="index")
    )

    # ── Global figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    df_g = df[df["scope"] == "global"]
    for ax, metric in zip(axes, METRICS):
        plot_panel(ax, df_g, metric, baselines["global"][metric], f"Global — {metric}")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Uncertainty-gated BM25 fusion — global", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "sweep_global.png", dpi=140)
    plt.close(fig)
    print(f"wrote {PLOTS / 'sweep_global.png'}")

    # ── Per-type figure (4 rows x 3 cols) ─────────────────────────────────────
    fig, axes = plt.subplots(len(TYPES), 3, figsize=(18, 4.2 * len(TYPES)))
    for i, t in enumerate(TYPES):
        df_t = df[df["scope"] == t]
        base = baselines.get(t, {m: float("nan") for m in METRICS})
        for j, metric in enumerate(METRICS):
            ax = axes[i][j]
            plot_panel(ax, df_t, metric, base[metric], f"{t} — {metric}")
            if i == 0 and j == 2:
                ax.legend(loc="best", fontsize=8)
    fig.suptitle("Uncertainty-gated BM25 fusion — per query type", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOTS / "sweep_by_type.png", dpi=140)
    plt.close(fig)
    print(f"wrote {PLOTS / 'sweep_by_type.png'}")

    # ── Best-tau summary ──────────────────────────────────────────────────────
    print("\nBest τ per scope/variant (by ndcg@10):")
    finite = df[np.isfinite(df["tau"])]
    for scope in ["global"] + TYPES:
        sub = finite[finite["scope"] == scope]
        for variant, sg in sub.groupby("variant"):
            best = sg.loc[sg["ndcg@10"].idxmax()]
            print(
                f"  {scope:<8}  {variant:<14}  τ*={best['tau']:.3f}  "
                f"ndcg@10={best['ndcg@10']:.4f}  (baseline={baselines[scope]['ndcg@10']:.4f})"
            )


if __name__ == "__main__":
    main()
