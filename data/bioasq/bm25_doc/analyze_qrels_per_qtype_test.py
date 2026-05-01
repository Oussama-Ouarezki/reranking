"""
Distribution of relevant documents per query, broken down by question type
— BioASQ test set (13B1–13B4).

Reads from  : data/bioasq/raw/Task13BGoldenEnriched/13B{1,2,3,4}_golden.json
Writes to   : data/bioasq/bm25_doc/images/

Outputs:
  qrels_per_qtype_histogram.png  — histogram per type (2×2 grid) + combined overlay
  qrels_per_qtype_cumulative.png — cumulative % per type on one axis
"""

import json
from collections import defaultdict
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
COLORS  = {
    "factoid": "#4c72b0",
    "list":    "#dd8452",
    "yesno":   "#2ca02c",
    "summary": "#d62728",
}

# ── load ──────────────────────────────────────────────────────────────────────
qrels:  dict[str, set]  = defaultdict(set)
qtypes: dict[str, str]  = {}

for batch in BATCHES:
    golden = json.load(open(TEST_DIR / f"{batch}_golden.json"))
    for q in golden["questions"]:
        qid = q["id"]
        qtypes[qid] = q["type"]
        for doc_url in q.get("documents", []):
            qrels[qid].add(doc_url.split("/")[-1])

# group counts by type
type_counts: dict[str, list[int]] = {t: [] for t in TYPES}
for qid, docs in qrels.items():
    t = qtypes.get(qid, "unknown")
    if t in type_counts:
        type_counts[t].append(len(docs))

all_counts = [c for vals in type_counts.values() for c in vals]

print(f"Total queries: {len(all_counts)}")
print(f"\n{'Type':<10} {'n':>5}  {'mean':>6}  {'median':>7}  {'min':>4}  {'max':>4}")
print("─" * 45)
for t in TYPES:
    v = type_counts[t]
    print(f"{t:<10} {len(v):>5}  {np.mean(v):>6.1f}  {np.median(v):>7.1f}"
          f"  {min(v):>4}  {max(v):>4}")


# ── Plot 1: 2×2 histogram grid per type + KDE overlay ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 2×2 individual panels (top-left 4 cells)
panel_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
max_val = max(all_counts)
bins = list(range(0, max_val + 5, 2))

for ax, t in zip(panel_axes, TYPES):
    vals = type_counts[t]
    col  = COLORS[t]
    sns.histplot(vals, bins=bins, discrete=False, color=col, alpha=0.75,
                 edgecolor="white", ax=ax, label=f"n={len(vals)}")
    ax.axvline(np.mean(vals),   color="black",  lw=2,   ls="--",
               label=f"Mean = {np.mean(vals):.1f}")
    ax.axvline(np.median(vals), color="#f39c12", lw=2,   ls=":",
               label=f"Median = {np.median(vals):.0f}")
    ax.set_title(f"{t.capitalize()}", fontsize=13, fontweight="bold", color=col)
    ax.set_xlabel("Relevant documents per query")
    ax.set_ylabel("Number of queries")
    ax.legend(fontsize=9)

# Top-right: all-types KDE overlay
ax_kde = axes[0, 2]
for t in TYPES:
    sns.kdeplot(type_counts[t], ax=ax_kde, color=COLORS[t],
                lw=2.2, fill=True, alpha=0.15, label=t.capitalize())
ax_kde.axvline(np.mean(all_counts), color="black", lw=1.5, ls="--",
               label=f"Overall mean={np.mean(all_counts):.1f}")
ax_kde.set_title("All Types — KDE Overlay", fontsize=13, fontweight="bold")
ax_kde.set_xlabel("Relevant documents per query")
ax_kde.set_ylabel("Density")
ax_kde.legend(fontsize=9)

# Bottom-right: combined histogram of all queries
ax_all = axes[1, 2]
sns.histplot(all_counts, bins=bins, color="steelblue", alpha=0.8,
             edgecolor="white", ax=ax_all)
ax_all.axvline(np.mean(all_counts),   color="#e74c3c", lw=2, ls="--",
               label=f"Mean = {np.mean(all_counts):.1f}")
ax_all.axvline(np.median(all_counts), color="#f39c12", lw=2, ls=":",
               label=f"Median = {np.median(all_counts):.0f}")
ax_all.set_title("All Types Combined", fontsize=13, fontweight="bold")
ax_all.set_xlabel("Relevant documents per query")
ax_all.set_ylabel("Number of queries")
ax_all.legend(fontsize=9)

fig.suptitle(
    "Relevant Documents per Query by Question Type — BioASQ Test Set",
    fontsize=14, fontweight="bold", y=1.01
)
fig.tight_layout()
out = OUT_DIR / "qrels_per_qtype_histogram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")


# ── Plot 2: cumulative distribution per type ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for t in TYPES:
    vals   = np.sort(type_counts[t])
    cumul  = np.arange(1, len(vals) + 1) / len(vals) * 100
    ax.plot(vals, cumul, color=COLORS[t], lw=2.5,
            label=f"{t.capitalize()} (n={len(vals)}, med={np.median(vals):.0f})")

# overall
sorted_all = np.sort(all_counts)
cumul_all  = np.arange(1, len(sorted_all) + 1) / len(sorted_all) * 100
ax.plot(sorted_all, cumul_all, color="black", lw=2, ls="--", alpha=0.6,
        label=f"Overall (n={len(all_counts)}, med={np.median(all_counts):.0f})")

for pct, ls in [(50, ":"), (75, "--"), (90, "-.")]:
    ax.axhline(pct, color="gray", lw=0.9, ls=ls, alpha=0.5)
    ax.text(max_val * 0.98, pct + 0.8, f"P{pct}", ha="right",
            fontsize=8, color="gray")

ax.set_xlabel("Number of relevant documents", fontsize=12)
ax.set_ylabel("% of queries", fontsize=12)
ax.set_title(
    "Cumulative Distribution of Relevant Docs per Query — by Question Type",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=10)
fig.tight_layout()
out = OUT_DIR / "qrels_per_qtype_cumulative.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved → {out}")
