"""
Distribution of relevant documents per query — BioASQ test set (13B1–13B4).

Reads from  : data/bioasq/raw/Task13BGoldenEnriched/13B{1,2,3,4}/
Writes to   : data/bioasq/bm25_doc/images/

Outputs:
  qrels_dist_test_histogram.png   — relevant docs per query (combined + per batch)
  qrels_dist_test_cumulative.png  — cumulative % of queries
"""

import json
from collections import Counter, defaultdict
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
BATCHES  = ["13B1", "13B2", "13B3", "13B4"]


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path) as f:
        next(f)
        for line in f:
            qid, did, _ = line.strip().split("\t")
            qrels[qid].add(did)
    return qrels


def load_queries(path):
    with open(path) as f:
        return [json.loads(l)["_id"] for l in f]


# ── Load all batches ──────────────────────────────────────────────────────────
batch_counts: dict[str, list[int]] = {}
all_counts: list[int] = []

for batch in BATCHES:
    qids  = load_queries(TEST_DIR / batch / "queries.jsonl")
    qrels = load_qrels(TEST_DIR / batch / "qrels.tsv")
    counts = [len(qrels.get(qid, set())) for qid in qids]
    counts_nz = [c for c in counts if c > 0]
    batch_counts[batch] = counts_nz
    all_counts.extend(counts_nz)

print(f"Total test queries with ≥1 relevant doc: {len(all_counts)}")
for batch, counts in batch_counts.items():
    print(f"  {batch}: {len(counts)} queries  mean={np.mean(counts):.1f}  max={max(counts)}")

# ── Plot 1: Histogram — combined + per-batch overlay ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Combined
ax = axes[0]
max_val = max(all_counts)
bins = list(range(0, max_val + 2))
sns.histplot(all_counts, bins=bins, discrete=True, color="steelblue", alpha=0.8, ax=ax)
ax.axvline(np.mean(all_counts),   color="#e74c3c", linewidth=2, linestyle="--", label=f"Mean = {np.mean(all_counts):.1f}")
ax.axvline(np.median(all_counts), color="#f39c12", linewidth=2, linestyle=":",  label=f"Median = {np.median(all_counts):.0f}")
ax.set_title("All Test Batches Combined", fontsize=12)
ax.set_xlabel("Relevant documents per query")
ax.set_ylabel("Number of queries")
ax.legend(fontsize=10)

# Per-batch
ax = axes[1]
colors = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
for (batch, counts), color in zip(batch_counts.items(), colors):
    sns.kdeplot(counts, ax=ax, label=batch, color=color, linewidth=2, fill=True, alpha=0.15)
ax.set_title("Per-Batch KDE", fontsize=12)
ax.set_xlabel("Relevant documents per query")
ax.set_ylabel("Density")
ax.legend(fontsize=10)

fig.suptitle("Distribution of Relevant Documents per Query — BioASQ Test Set", fontsize=13, y=1.01)
fig.tight_layout()
out = OUT_DIR / "qrels_dist_test_histogram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")

# ── Plot 2: Cumulative distribution ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

sorted_all = np.sort(all_counts)
cumulative = np.arange(1, len(sorted_all) + 1) / len(sorted_all) * 100
ax.plot(sorted_all, cumulative, color="steelblue", linewidth=2.5, label="All batches")

for pct, color, ls in [(50, "#f39c12", ":"), (75, "#27ae60", "--"), (90, "#e74c3c", "-.")]:
    val = np.percentile(all_counts, pct)
    ax.axvline(val, color=color, linewidth=1.5, linestyle=ls, label=f"P{pct} = {val:.0f} docs")
    ax.axhline(pct, color=color, linewidth=0.8, linestyle=ls, alpha=0.4)

ax.set_title("Cumulative Distribution — Relevant Docs per Query (Test Set)", fontsize=13)
ax.set_xlabel("Number of relevant documents", fontsize=12)
ax.set_ylabel("% of queries", fontsize=12)
ax.legend(fontsize=11)
fig.tight_layout()
out = OUT_DIR / "qrels_dist_test_cumulative.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"Saved → {out}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\n{'─'*45}")
freq = Counter(all_counts)
print(f"  {'# docs':<10} {'# queries':<12} {'% queries'}")
print(f"  {'─'*35}")
for n in sorted(freq):
    print(f"  {n:<10} {freq[n]:<12,} {100*freq[n]/len(all_counts):.1f}%")
print(f"{'─'*45}")
print(f"  Mean   : {np.mean(all_counts):.2f}")
print(f"  Median : {np.median(all_counts):.1f}")
print(f"  Max    : {max(all_counts)}")
