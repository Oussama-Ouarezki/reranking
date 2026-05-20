import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

SCORES_PATH = Path("qwen3_0.6b/data/qwen06b_scores_test.jsonl")
OUT_PATH = Path("qwen3_0.6b/qwen06b_score_distribution.png")

probs = []
with SCORES_PATH.open() as f:
    for line in f:
        rec = json.loads(line)
        for s in rec["scores"]:
            probs.append(s["qwen_prob"])

probs = np.asarray(probs, dtype=np.float64)
log_probs = np.log10(np.clip(probs, 1e-12, 1.0))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(probs, bins=60, color="#3b7dd8", edgecolor="white")
axes[0].set_xlabel("Qwen relevance probability")
axes[0].set_ylabel("Count")
axes[0].set_title(f"Linear scale  (n={len(probs):,})")

axes[1].hist(log_probs, bins=60, color="#d8743b", edgecolor="white")
axes[1].set_xlabel("log10(Qwen probability)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log scale")

mean = probs.mean()
median = np.median(probs)
fig.suptitle(
    f"Qwen3-0.6B score distribution — BioASQ test  "
    f"(mean={mean:.3f}, median={median:.3g})",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved {OUT_PATH}  ({len(probs):,} scores)")
