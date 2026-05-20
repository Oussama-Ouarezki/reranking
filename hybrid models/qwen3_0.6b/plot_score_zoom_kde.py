import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

SCORES_PATH = Path("qwen3_0.6b/data/qwen06b_scores_test.jsonl")
OUT_PATH = Path("qwen3_0.6b/qwen06b_score_zoom_0999_kde.png")

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

# qwen_prob lives on a coarse fp16/bf16 grid — only 3 distinct values in [0.999, 1].
# To get a "continuous" view, smooth with a Gaussian KDE (with a narrow bandwidth
# so the smoothing is visible but doesn't wash out structure).
xs = np.linspace(LO, HI, 1000)

bandwidths = [3e-5, 1e-4, 3e-4]
colors = ["#3b7dd8", "#2f9e6b", "#d8743b"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# Left: KDEs at multiple bandwidths overlaid
axL = axes[0]
for bw, c in zip(bandwidths, colors):
    kde = gaussian_kde(zoom, bw_method=bw / zoom.std())
    ys = kde(xs)
    axL.plot(xs, ys, color=c, linewidth=1.8, label=f"bandwidth = {bw:.0e}")
    axL.fill_between(xs, ys, alpha=0.12, color=c)
axL.set_xlim(LO, HI)
axL.set_xlabel("Qwen relevance probability")
axL.set_ylabel("Density (KDE)")
axL.set_title(f"Smoothed density in [{LO}, {HI}]  (n={n_zoom:,})")
axL.legend(loc="upper left", fontsize=9)
axL.ticklabel_format(useOffset=False, style="plain", axis="x")

# Right: KDE + rug + actual data spikes for context
axR = axes[1]
kde = gaussian_kde(zoom, bw_method=1e-4 / zoom.std())
ys = kde(xs)
axR.plot(xs, ys, color="#3b7dd8", linewidth=1.8, label="KDE (bw=1e-4)")
axR.fill_between(xs, ys, alpha=0.18, color="#3b7dd8")

uniq, counts = np.unique(zoom, return_counts=True)
axR2 = axR.twinx()
axR2.vlines(uniq, 0, counts, color="crimson", linewidth=1.5, alpha=0.7,
            label="actual fp grid")
axR2.set_ylabel("Count at exact float value", color="crimson")
axR2.tick_params(axis="y", labelcolor="crimson")
axR2.grid(False)

axR.set_xlim(LO, HI)
axR.set_xlabel("Qwen relevance probability")
axR.set_ylabel("Density (KDE)", color="#3b7dd8")
axR.tick_params(axis="y", labelcolor="#3b7dd8")
axR.set_title(f"KDE overlaid on the {len(uniq)} actual fp16 values")
axR.ticklabel_format(useOffset=False, style="plain", axis="x")

mean_z = zoom.mean()
median_z = np.median(zoom)
fig.suptitle(
    f"Qwen3-0.6B — continuous (KDE) view of [{LO}, {HI}]   "
    f"n={n_zoom:,} of {n_total:,}  ({100 * n_zoom / n_total:.2f}%)   "
    f"mean={mean_z:.6f}  median={median_z:.6f}",
    fontsize=12, y=1.02,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
print(f"Distinct fp values in window: {len(uniq)} -> {uniq.tolist()}")
print(f"Counts: {counts.tolist()}")
