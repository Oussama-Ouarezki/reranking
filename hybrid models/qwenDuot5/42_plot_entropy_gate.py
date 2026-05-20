"""Plot entropy gate sweep: metrics vs % routed to duoT5."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

ROOT     = Path(__file__).resolve().parent
SWEEP_F  = ROOT / "entropy_gate20_sweep.tsv"
BEST_F   = ROOT / "entropy_gate20_best.json"
OUT      = ROOT / "entropy_gate20_plot.png"

QWEN_BASELINES = {"ndcg@1": 0.9435, "ndcg@5": 0.9229, "ndcg@10": 0.9007, "mrr@10": 0.9652}
DUO_BASELINES  = {"ndcg@1": 0.9375, "ndcg@5": 0.9169, "ndcg@10": 0.8979, "mrr@10": 0.9628}

COLORS = {"ndcg@1": "#e15759", "ndcg@5": "#f28e2b", "ndcg@10": "#4e79a7", "mrr@10": "#59a14f"}
LABELS = {"ndcg@1": "nDCG@1", "ndcg@5": "nDCG@5", "ndcg@10": "nDCG@10", "mrr@10": "MRR@10"}

df   = pd.read_csv(SWEEP_F, sep="\t")
best = json.loads(BEST_F.read_text())

fig, (ax_metric, ax_pct) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                          gridspec_kw={"height_ratios": [3, 1]})

metrics = ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@10"]

for m in metrics:
    ax_metric.plot(df["pct_routed"], df[m], color=COLORS[m], label=LABELS[m], linewidth=2)
    ax_metric.axhline(QWEN_BASELINES[m], color=COLORS[m], linestyle="--", linewidth=1, alpha=0.5)
    ax_metric.axhline(DUO_BASELINES[m],  color=COLORS[m], linestyle=":",  linewidth=1, alpha=0.5)

best_pct = best["pct_routed"]
ax_metric.axvline(best_pct, color="black", linestyle="-.", linewidth=1.5, alpha=0.8,
                  label=f"Best τ ({best_pct:.1f}% routed)")

ax_metric.set_ylabel("Score", fontsize=12)
ax_metric.set_title("Entropy Gate H@20: Qwen → duoT5\n"
                    "(-- Qwen baseline, ··· duoT5 baseline)", fontsize=13)
ax_metric.legend(fontsize=10, loc="lower left")
ax_metric.set_ylim(0.88, 0.98)

# Lower panel: % routed
ax_pct.plot(df["pct_routed"], df["pct_routed"], color="gray", linewidth=1)
ax_pct.fill_between(df["pct_routed"], df["pct_routed"], alpha=0.15, color="gray")
ax_pct.axvline(best_pct, color="black", linestyle="-.", linewidth=1.5, alpha=0.8)
ax_pct.set_xlabel("% queries routed to duoT5  (τ decreases →)", fontsize=12)
ax_pct.set_ylabel("% routed", fontsize=11)
ax_pct.set_xlim(df["pct_routed"].min(), df["pct_routed"].max())

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT}")
