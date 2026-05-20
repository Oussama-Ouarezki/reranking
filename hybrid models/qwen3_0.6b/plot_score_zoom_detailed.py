import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

SCORES_PATH = Path("qwen3_0.6b/data/qwen06b_scores_test.jsonl")
OUT_PATH = Path("qwen3_0.6b/qwen06b_score_zoom_0999_detailed.png")

LO, HI = 0.999, 1.0

probs = []
with SCORES_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        for s in rec["scores"]:
            probs.append(s["qwen_prob"])
probs = np.asarray(probs, dtype=np.float64)
n_total = len(probs)

zoom = probs[(probs >= LO) & (probs <= HI)]
n_zoom = len(zoom)

# distinct float values in this window — qwen_prob comes from bf16/fp16 logits
# so values land on a small grid; counting unique values reveals quantization
uniq, counts = np.unique(zoom, return_counts=True)

mean_z = zoom.mean()
median_z = np.median(zoom)
std_z = zoom.std()

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.22)

# 1. Fine-binned histogram
ax1 = fig.add_subplot(gs[0, 0])
bins = np.linspace(LO, HI, 200)
ax1.hist(zoom, bins=bins, color="#3b7dd8", edgecolor="white", linewidth=0.2)
ax1.axvline(mean_z, color="crimson", linestyle="--", linewidth=1.2, label=f"mean={mean_z:.6f}")
ax1.axvline(median_z, color="black", linestyle=":", linewidth=1.2, label=f"median={median_z:.6f}")
ax1.set_xlim(LO, HI)
ax1.set_xlabel("Qwen relevance probability")
ax1.set_ylabel("Count")
ax1.set_title(f"Histogram, 200 bins  (n={n_zoom:,})")
ax1.legend(loc="upper left", fontsize=9)

# 2. Log-y to expose rare values vs the dominant spike at ~1.0
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(zoom, bins=bins, color="#6a3bd8", edgecolor="white", linewidth=0.2)
ax2.set_yscale("log")
ax2.set_xlim(LO, HI)
ax2.set_xlabel("Qwen relevance probability")
ax2.set_ylabel("Count (log)")
ax2.set_title("Same range, log-y — exposes minor bins")

# 3. Stem plot at the actual unique float values (shows fp quantization grid)
ax3 = fig.add_subplot(gs[1, 0])
markerline, stemlines, baseline = ax3.stem(uniq, counts, basefmt=" ")
plt.setp(stemlines, color="#2f9e6b", linewidth=1.0)
plt.setp(markerline, color="#2f9e6b", markersize=3)
ax3.set_xlim(LO, HI)
ax3.set_xlabel("Qwen probability (unique values only)")
ax3.set_ylabel("Count at that exact value")
ax3.set_title(f"Per-value counts  ({len(uniq)} distinct floats in [{LO}, {HI}])")

# 4. Top-N exact values + summary stats
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis("off")
order = np.argsort(counts)[::-1]
top_k = min(15, len(uniq))
top_lines = ["value           count   share-of-zoom"]
for idx in order[:top_k]:
    v = uniq[idx]
    c = counts[idx]
    top_lines.append(f"{v:.10f}   {c:5d}   {100 * c / n_zoom:5.2f}%")

text = "\n".join([
    f"Window: [{LO}, {HI}]",
    f"In window: {n_zoom:,} of {n_total:,}  ({100 * n_zoom / n_total:.2f}%)",
    f"Distinct float values: {len(uniq)}",
    "",
    f"mean   = {mean_z:.8f}",
    f"median = {median_z:.8f}",
    f"std    = {std_z:.2e}",
    f"min    = {zoom.min():.8f}",
    f"max    = {zoom.max():.8f}",
    "",
    f"Top {top_k} most frequent exact values:",
    *top_lines,
])
ax4.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=9,
         bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                   edgecolor="grey", alpha=0.9))
ax4.set_title("Summary + top exact values")

fig.suptitle(
    f"Qwen3-0.6B score distribution — zoom [{LO}, {HI}]   "
    f"({n_zoom:,} / {n_total:,} scores, {100 * n_zoom / n_total:.2f}%)",
    fontsize=14, y=0.995,
)
fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
print(f"Distinct values in [{LO}, {HI}]: {len(uniq)}")
