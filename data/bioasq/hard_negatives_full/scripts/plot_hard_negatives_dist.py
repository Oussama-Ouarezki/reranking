"""
Distribution plots for hard negatives:
  1. CE score distribution (histogram + KDE)
  2. BM25 rank distribution (histogram)
  3. CE score vs BM25 rank (2D hexbin)

Output: data/bioasq/hard_negatives_full/images/
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE    = Path(__file__).resolve().parents[4]
IN_FILE = BASE / "data/bioasq/hard_negatives_full/split/hard_negatives.jsonl"
IMG_DIR = BASE / "data/bioasq/hard_negatives_full/images"
IMG_DIR.mkdir(exist_ok=True)

print("Loading hard negatives...")
ce_scores  = []
bm25_ranks = []

with IN_FILE.open() as f:
    for line in f:
        d = json.loads(line)
        ce_scores.append(d["ce_score"])
        bm25_ranks.append(d["bm25_rank"])

ce_scores  = np.array(ce_scores)
bm25_ranks = np.array(bm25_ranks)
print(f"Loaded {len(ce_scores):,} hard negatives")

# ── 1. CE score distribution ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(ce_scores, bins=80, density=True, alpha=0.55, color="steelblue", label="Histogram")
sns.kdeplot(ce_scores, ax=ax, color="tomato", linewidth=2, label="KDE")
ax.set_xlabel("MedCPT Cross-Encoder Score", fontsize=12)
ax.set_ylabel("Density", fontsize=12)
ax.set_title("CE Score Distribution — Hard Negatives", fontsize=13)
ax.legend()
fig.tight_layout()
out = IMG_DIR / "ce_score_distribution.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")

# ── 2. BM25 rank distribution ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(bm25_ranks, bins=80, color="mediumseagreen", alpha=0.75)
ax.set_xlabel("BM25 Rank", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("BM25 Rank Distribution — Hard Negatives", fontsize=13)
fig.tight_layout()
out = IMG_DIR / "bm25_rank_distribution.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")

# ── 3. CE score vs BM25 rank (hexbin) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
hb = ax.hexbin(bm25_ranks, ce_scores, gridsize=50, cmap="YlOrRd", mincnt=1)
fig.colorbar(hb, ax=ax, label="Count")
ax.set_xlabel("BM25 Rank", fontsize=12)
ax.set_ylabel("CE Score", fontsize=12)
ax.set_title("CE Score vs BM25 Rank — Hard Negatives", fontsize=13)
fig.tight_layout()
out = IMG_DIR / "ce_score_vs_bm25_rank.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved {out}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\nCE score  — mean={ce_scores.mean():.4f}  std={ce_scores.std():.4f}  "
      f"min={ce_scores.min():.4f}  max={ce_scores.max():.4f}")
print(f"BM25 rank — mean={bm25_ranks.mean():.1f}  std={bm25_ranks.std():.1f}  "
      f"min={bm25_ranks.min()}  max={bm25_ranks.max()}")
