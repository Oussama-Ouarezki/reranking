import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

SCORES_PATH = Path("qwen3_0.6b/data/qwen06b_scores_test.jsonl")
OUT_PATH = Path("qwen3_0.6b/qwen06b_score_distribution_zoom.png")

probs = []
with SCORES_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        for s in rec["scores"]:
            probs.append(s["qwen_prob"])
probs = np.asarray(probs, dtype=np.float64)

zoom_a = probs[(probs >= 0.99) & (probs <= 1.0)]
zoom_b = probs[(probs >= 0.999) & (probs <= 1.0)]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(zoom_a, bins=np.linspace(0.99, 1.0, 60), color="#3b7dd8", edgecolor="white")
axes[0].set_xlim(0.99, 1.0)
axes[0].set_xlabel("Qwen relevance probability")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Zoom [0.99, 1.0]  (n={len(zoom_a):,} of {len(probs):,})")

axes[1].hist(zoom_b, bins=np.linspace(0.999, 1.0, 60), color="#d8743b", edgecolor="white")
axes[1].set_xlim(0.999, 1.0)
axes[1].set_xlabel("Qwen relevance probability")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Zoom [0.999, 1.0]  (n={len(zoom_b):,} of {len(probs):,})")

fig.suptitle("Qwen3-0.6B score distribution — zoomed near 1.0", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
print(f"  >=0.99 : {len(zoom_a):,} ({100*len(zoom_a)/len(probs):.2f}%)")
print(f"  >=0.999: {len(zoom_b):,} ({100*len(zoom_b)/len(probs):.2f}%)")
