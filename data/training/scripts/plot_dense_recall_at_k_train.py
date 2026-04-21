"""
Plot MedCPT Dense Retrieval Recall@K curve on the truncated BioASQ training set.
Reads pre-computed scores from data/training/dense_retrival/recall_scores.tsv.
Saves: data/training/images/dense_recall_at_k_train.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────
RECALL_TSV = Path("data/training/dense_retrival/recall_scores.tsv")
OUT_PNG    = Path("data/training/images/dense_recall_at_k_train.png")
K_VALUES   = [1, 5, 10, 20]

# ── Load mean recall scores ───────────────────────────────────────────────────
print(f"Reading {RECALL_TSV} …")
per_query: dict[str, list[float]] = {}
mean_row: list[float] | None = None

with RECALL_TSV.open(encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    headers = reader.fieldnames or []
    recall_cols = [h for h in headers if h.startswith("recall@")]
    for row in reader:
        scores = [float(row[col]) for col in recall_cols]
        if row["qid"] == "mean":
            mean_row = scores
        else:
            per_query[row["qid"]] = scores

if mean_row is None:
    # compute from per-query if mean row absent
    n = len(per_query)
    mean_row = [
        sum(v[i] for v in per_query.values()) / n
        for i in range(len(K_VALUES))
    ]

print(f"  {len(per_query):,} queries loaded")

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n  {'K':<6}  {'Mean Recall@K':>14}")
print("  " + "─" * 22)
for k, m in zip(K_VALUES, mean_row):
    print(f"  {k:<6}  {m:>14.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

x    = range(len(K_VALUES))
bars = ax.bar(x, mean_row, color="#E91E63", alpha=0.85, width=0.5,
              label="MedCPT Dense (training set)")

for bar, val in zip(bars, mean_row):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(list(x))
ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=11)
ax.set_xlabel("K  (number of retrieved documents)", fontsize=12)
ax.set_ylabel("Mean Recall@K", fontsize=12)
ax.set_title(
    "MedCPT Dense Retrieval — Recall@K\nBioASQ Truncated Training Set",
    fontsize=13,
)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nPlot saved → {OUT_PNG}")
