import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

SCORES_PATH = Path("qwen3_0.6b/data/qwen06b_scores_test.jsonl")
OUT_PATH = Path("qwen3_0.6b/qwen06b_score_distribution_detailed.png")

probs = []
with SCORES_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        for s in rec["scores"]:
            probs.append(s["qwen_prob"])
probs = np.asarray(probs, dtype=np.float64)
n = len(probs)

# stats
mean = probs.mean()
median = np.median(probs)
std = probs.std()
pcts = {p: np.percentile(probs, p) for p in (1, 5, 25, 50, 75, 95, 99)}
frac_lt_001 = (probs < 0.01).mean()
frac_gt_099 = (probs > 0.99).mean()
frac_gt_0999 = (probs > 0.999).mean()
frac_mid = ((probs >= 0.1) & (probs <= 0.9)).mean()

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.22)

# 1. Linear-scale histogram with fine bins
ax1 = fig.add_subplot(gs[0, 0])
bins = np.linspace(0, 1, 101)
ax1.hist(probs, bins=bins, color="#3b7dd8", edgecolor="white", linewidth=0.3)
ax1.axvline(mean, color="crimson", linestyle="--", linewidth=1.2, label=f"mean={mean:.3f}")
ax1.axvline(median, color="black", linestyle=":", linewidth=1.2, label=f"median={median:.3g}")
ax1.set_xlabel("Qwen relevance probability")
ax1.set_ylabel("Count")
ax1.set_title("Linear scale (100 bins)")
ax1.legend(loc="upper center")

# 2. Log y-axis: shows the bulk near 0 vs spike near 1
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(probs, bins=bins, color="#6a3bd8", edgecolor="white", linewidth=0.3)
ax2.set_yscale("log")
ax2.set_xlabel("Qwen relevance probability")
ax2.set_ylabel("Count (log)")
ax2.set_title("Linear x, log y — exposes long-tail bins")

# 3. Log10 of probability (shows the lower-tail structure)
ax3 = fig.add_subplot(gs[1, 0])
log_probs = np.log10(np.clip(probs, 1e-12, 1.0))
ax3.hist(log_probs, bins=80, color="#d8743b", edgecolor="white", linewidth=0.3)
ax3.set_xlabel("log10(Qwen probability)")
ax3.set_ylabel("Count")
ax3.set_title("Log-probability — reveals low-confidence spread")

# 4. CDF
ax4 = fig.add_subplot(gs[1, 1])
sorted_p = np.sort(probs)
cdf = np.arange(1, n + 1) / n
ax4.plot(sorted_p, cdf, color="#2f9e6b", linewidth=1.4)
for p in (25, 50, 75, 95):
    v = pcts[p]
    ax4.axvline(v, color="grey", linestyle=":", linewidth=0.8)
    ax4.annotate(f"P{p}={v:.3g}", xy=(v, p / 100), xytext=(5, -10),
                 textcoords="offset points", fontsize=8)
ax4.set_xlabel("Qwen probability")
ax4.set_ylabel("Cumulative fraction")
ax4.set_title("Empirical CDF")

# 5. Zoom near 1.0 with fine bins
ax5 = fig.add_subplot(gs[2, 0])
zoom = probs[probs >= 0.9]
ax5.hist(zoom, bins=np.linspace(0.9, 1.0, 80), color="#3b7dd8", edgecolor="white", linewidth=0.3)
ax5.set_xlim(0.9, 1.0)
ax5.set_xlabel("Qwen probability")
ax5.set_ylabel("Count")
ax5.set_title(f"Zoom [0.9, 1.0]  (n={len(zoom):,}, {100 * len(zoom) / n:.1f}%)")

# 6. Stats / region breakdown panel
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis("off")
lines = [
    f"Total scores: {n:,}  (340 queries × 50 docs)",
    "",
    f"mean   = {mean:.4f}",
    f"median = {median:.4g}",
    f"std    = {std:.4f}",
    f"min    = {probs.min():.3g}",
    f"max    = {probs.max():.6f}",
    "",
    "Percentiles:",
    *(f"  P{p:>2} = {v:.4g}" for p, v in pcts.items()),
    "",
    "Region fractions:",
    f"  prob < 0.01   : {frac_lt_001 * 100:5.2f}%",
    f"  0.1 ≤ p ≤ 0.9 : {frac_mid * 100:5.2f}%   (uncertain band)",
    f"  prob > 0.99   : {frac_gt_099 * 100:5.2f}%",
    f"  prob > 0.999  : {frac_gt_0999 * 100:5.2f}%",
]
ax6.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
         family="monospace", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="grey", alpha=0.9))
ax6.set_title("Summary statistics")

fig.suptitle(
    "Qwen3-0.6B reranker score distribution — BioASQ test (340 queries, top-50 BM25)",
    fontsize=14, y=0.995,
)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
