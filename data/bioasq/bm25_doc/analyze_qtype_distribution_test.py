"""
Distribution of question types per batch — BioASQ test set (13B1–13B4).

Reads from  : data/bioasq/raw/Task13BGoldenEnriched/13B{1,2,3,4}_golden.json
Writes to   : data/bioasq/bm25_doc/images/

Outputs:
  qtype_dist_test_bar.png        — count + % per type (combined + per batch)
  qtype_dist_test_stacked.png    — stacked bar showing type mix per batch
  qtype_dist_test_pie.png        — pie chart of overall type proportions
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE     = Path(__file__).resolve().parents[3]
TEST_DIR = BASE / "data/bioasq/raw/Task13BGoldenEnriched"
OUT_DIR  = BASE / "data/bioasq/bm25_doc/images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCHES = ["13B1", "13B2", "13B3", "13B4"]
TYPES   = ["factoid", "list", "yesno", "summary"]

COLORS = {
    "factoid": "#4c72b0",
    "list":    "#dd8452",
    "yesno":   "#2ca02c",
    "summary": "#d62728",
}

# ── load data ─────────────────────────────────────────────────────────────────
batch_counters: dict[str, Counter] = {}
all_types: list[str] = []

for batch in BATCHES:
    golden = json.load(open(TEST_DIR / f"{batch}_golden.json"))
    types  = [q["type"] for q in golden["questions"]]
    batch_counters[batch] = Counter(types)
    all_types.extend(types)
    print(f"  {batch}: {len(types)} questions — {dict(Counter(types))}")

total = len(all_types)
overall = Counter(all_types)
print(f"\nTotal: {total} questions")
for t in TYPES:
    print(f"  {t:<10} {overall[t]:>4}  ({100*overall[t]/total:.1f}%)")


# ── Plot 1: grouped bar — combined + per batch side by side ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combined bar chart
ax = axes[0]
counts = [overall[t] for t in TYPES]
bars   = ax.bar(TYPES, counts, color=[COLORS[t] for t in TYPES],
                edgecolor="white", alpha=0.85, width=0.55)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{cnt}\n({100*cnt/total:.1f}%)", ha="center", va="bottom",
            fontsize=10, fontweight="bold")
ax.set_title("All Batches Combined", fontsize=12, fontweight="bold")
ax.set_xlabel("Question type")
ax.set_ylabel("Number of questions")
ax.set_ylim(0, max(counts) * 1.22)

# Per-batch grouped bar
ax = axes[1]
x      = np.arange(len(TYPES))
n_bat  = len(BATCHES)
width  = 0.18
bat_colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
for i, (batch, col) in enumerate(zip(BATCHES, bat_colors)):
    vals = [batch_counters[batch].get(t, 0) for t in TYPES]
    offset = (i - n_bat / 2 + 0.5) * width
    rects = ax.bar(x + offset, vals, width, label=batch,
                   color=col, edgecolor="white", alpha=0.82)
    for rect, v in zip(rects, vals):
        if v > 0:
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + 0.4, str(v),
                    ha="center", va="bottom", fontsize=7)

ax.set_xticks(x)
ax.set_xticklabels(TYPES, fontsize=11)
ax.set_title("Per Batch Breakdown", fontsize=12, fontweight="bold")
ax.set_xlabel("Question type")
ax.set_ylabel("Number of questions")
ax.legend(fontsize=10)

fig.suptitle(
    f"Question Type Distribution — BioASQ Test Set (n={total})",
    fontsize=13, fontweight="bold", y=1.02
)
fig.tight_layout()
out = OUT_DIR / "qtype_dist_test_bar.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")


# ── Plot 2: 100% stacked bar per batch ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

bottoms = np.zeros(len(BATCHES))
x_pos   = np.arange(len(BATCHES))
for t in TYPES:
    vals = np.array([batch_counters[b].get(t, 0) for b in BATCHES], dtype=float)
    totals = np.array([sum(batch_counters[b].values()) for b in BATCHES], dtype=float)
    pcts   = vals / totals * 100
    bars   = ax.bar(x_pos, pcts, bottom=bottoms, label=t,
                    color=COLORS[t], edgecolor="white", alpha=0.85, width=0.5)
    for bar, pct, v in zip(bars, pcts, vals):
        if pct > 4:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + pct / 2,
                    f"{t[0].upper()}\n{int(v)} ({pct:.0f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold")
    bottoms += pcts

batch_totals = [sum(batch_counters[b].values()) for b in BATCHES]
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(BATCHES, batch_totals)], fontsize=11)
ax.set_ylabel("% of questions", fontsize=12)
ax.set_ylim(0, 105)
ax.set_title(
    f"Question Type Mix per Batch — BioASQ Test Set (n={total})",
    fontsize=13, fontweight="bold"
)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
fig.tight_layout()
out = OUT_DIR / "qtype_dist_test_stacked.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved → {out}")


# ── Plot 3: overall pie chart ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
sizes  = [overall[t] for t in TYPES]
explode = [0.03] * len(TYPES)
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=[f"{t}\n({overall[t]})" for t in TYPES],
    colors=[COLORS[t] for t in TYPES],
    autopct="%1.1f%%",
    startangle=140,
    explode=explode,
    pctdistance=0.75,
    textprops={"fontsize": 11},
)
for at in autotexts:
    at.set_fontweight("bold")
ax.set_title(
    f"Overall Question Type Distribution\nBioASQ Test Set (n={total})",
    fontsize=13, fontweight="bold"
)
fig.tight_layout()
out = OUT_DIR / "qtype_dist_test_pie.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved → {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'─'*55}")
print(f"  {'Type':<10} {'Total':>6}  {'%':>6}  " +
      "  ".join(f"{b:>5}" for b in BATCHES))
print(f"  {'─'*53}")
for t in TYPES:
    per_batch = "  ".join(f"{batch_counters[b].get(t,0):>5}" for b in BATCHES)
    print(f"  {t:<10} {overall[t]:>6}  {100*overall[t]/total:>5.1f}%  {per_batch}")
print(f"  {'─'*53}")
print(f"  {'TOTAL':<10} {total:>6}  {'100.0%':>6}  " +
      "  ".join(f"{sum(batch_counters[b].values()):>5}" for b in BATCHES))
print(f"{'─'*55}")
