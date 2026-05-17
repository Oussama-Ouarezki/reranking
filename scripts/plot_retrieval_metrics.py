"""
Retrieval Evaluation Metrics Visualization
Per query type: Accuracy, MRR, F1, LLM-as-Judge
Global chart: averaged across 4 metrics per model/k combo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── 1. RAW DATA ────────────────────────────────────────────────────────────────

K_VALUES = [1, 3, 5, 10, 20]

MODELS = [
    "BM25\n(350 tok)",
    "BM25→LiT5\n(350 tok)",
    
    "BM25→LiT5\n(LoRA ep2, 4k)",
    "BM25→Qwen3\n0.6B (pure)",


]

# Shape: (7 models × 5 k-values)
DATA = {
    # ── Factoid · MRR ──────────────────────────────────────────────────────────
    "factoid_mrr": np.array([
        [0.2974, 0.3332, 0.3319, 0.3860, 0.0105],  # BM25 350tok
        [0.3489, 0.3704, 0.3521, 0.3196, 0.0105],  # LiT5 350tok
        [0.3337, 0.3258, 0.3653, 0.3372, 0.0193],  # LiT5 ft ep4
        [0.3470, 0.3526, 0.3705, 0.3256, 0.0105],  # LiT5 LoRA ep2
        [0.3039, 0.3254, 0.3621, 0.2596, 0.0211],  # Qwen3 pure
        [0.3288, 0.3296, 0.3372, 0.2930, 0.0211],  # Qwen3 ft
        [0.2828, 0.3398, 0.3644, 0.2816, 0.0211],  # Qwen3 pure2
    ]),
    # ── Factoid · Strict Accuracy ──────────────────────────────────────────────
    "factoid_acc": np.array([
        [0.2526, 0.3053, 0.3053, 0.3368, 0.0105],
        [0.3053, 0.3263, 0.3053, 0.2842, 0.0105],
        [0.2842, 0.2737, 0.3158, 0.2947, 0.0105],
        [0.2947, 0.3053, 0.3158, 0.2842, 0.0105],
        [0.2632, 0.2947, 0.3158, 0.2211, 0.0211],
        [0.2632, 0.2842, 0.2947, 0.2632, 0.0211],
        [0.2526, 0.2947, 0.3158, 0.2526, 0.0211],
    ]),
    # ── Yes/No · Accuracy ─────────────────────────────────────────────────────
    "yesno_acc": np.array([
        [0.8537, 0.8537, 0.8415, 0.7439, 0.1829],
        [0.9146, 0.8780, 0.8659, 0.7683, 0.1220],
        [0.9146, 0.9024, 0.8780, 0.6951, 0.1585],
        [0.9024, 0.8780, 0.8415, 0.7317, 0.1098],
        [0.9268, 0.9024, 0.8902, 0.7561, 0.1463],
        [0.9146, 0.9146, 0.8902, 0.7561, 0.1585],
        [0.9390, 0.9146, 0.8659, 0.7561, 0.1220],
    ]),
    # ── List · F1 ─────────────────────────────────────────────────────────────
    "list_f1": np.array([
        [0.2522, 0.2425, 0.2489, 0.2327, 0.0078],
        [0.2838, 0.2996, 0.2862, 0.2544, 0.0000],
        [0.2761, 0.3050, 0.2933, 0.2593, 0.0072],
        [0.2649, 0.3000, 0.2396, 0.2350, 0.0119],
        [0.2761, 0.2880, 0.2941, 0.2246, 0.0020],
        [0.2984, 0.2880, 0.2775, 0.2341, 0.0162],
        [0.2740, 0.3021, 0.2724, 0.2385, 0.0056],
    ]),
    # ── Summary · LLM Judge ───────────────────────────────────────────────────
    "summary_llm": np.array([
        [0.5455, 0.6234, 0.7013, 0.6883, 0.3506],
        [0.5844, 0.6364, 0.6623, 0.6364, 0.3896],
        [0.6104, 0.6623, 0.6883, 0.7273, 0.3766],
        [0.6364, 0.6494, 0.6623, 0.6623, 0.3636],
        [0.6234, 0.6753, 0.6623, 0.6494, 0.4026],
        [0.5844, 0.6364, 0.6234, 0.7013, 0.3896],
        [0.6234, 0.6364, 0.6494, 0.6364, 0.4026],
    ]),
}

# ── 2. PALETTE & STYLE ────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.4,
})

PALETTE = sns.color_palette("tab10", n_colors=len(MODELS))
MARKERS     = ["o", "s", "D", "^", "v", "P", "X"]
LINE_STYLES = ["-", "--", "-.", ":", (0, (3,1,1,1)), (0, (5,2)), (0, (1,1))]

x = np.array(K_VALUES)

# ── 3. HELPER ─────────────────────────────────────────────────────────────────

def plot_metric(ax, matrix, title, ylabel, highlight_k=None):
    """Plot one metric panel."""
    for i, (model, color, marker, ls) in enumerate(
        zip(MODELS, PALETTE, MARKERS, LINE_STYLES)
    ):
        ax.plot(
            x, matrix[i],
            color=color, marker=marker, linestyle=ls,
            linewidth=2.0, markersize=6, label=model,
        )
    if highlight_k:
        ax.axvline(highlight_k, color="grey", linestyle=":", alpha=0.5)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("k (retrieved docs)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(K_VALUES)
    ax.set_xlim(0.5, 21)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

# ── 4. PER-QUERY-TYPE FIGURE (2 × 3 layout: 5 panels + legend) ──────────────

fig, axes = plt.subplots(2, 3, figsize=(19, 10), constrained_layout=True)
fig.suptitle(
    "Retrieval Pipeline Evaluation — Per Query Type",
    fontsize=15, fontweight="bold", y=1.01
)

PANELS = [
    # (matrix_key, title, ylabel)
    ("factoid_mrr",  "Factoid — MRR",           "MRR"),
    ("factoid_acc",  "Factoid — Strict Accuracy","Accuracy"),
    ("yesno_acc",    "Yes/No — Accuracy",        "Accuracy"),
    ("list_f1",      "List — F1",                "F1"),
    ("summary_llm",  "Summary — LLM-as-Judge",   "Score"),
]

flat_axes = axes.flatten()
for ax, (key, title, ylabel) in zip(flat_axes, PANELS):
    plot_metric(ax, DATA[key], title, ylabel)

# Shared legend in the last empty cell
legend_ax = flat_axes[-1]
legend_ax.axis("off")
handles = [
    mpatches.Patch(facecolor=PALETTE[i], label=MODELS[i].replace("\n", " "))
    for i in range(len(MODELS))
]
legend_ax.legend(
    handles=handles,
    title="Pipeline",
    title_fontsize=11,
    fontsize=10,
    loc="center",
    frameon=True,
    framealpha=0.9,
    edgecolor="#cccccc",
)

plt.savefig("scripts/per_query_type_metrics.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ per_query_type_metrics.png saved")

# ── 5. GLOBAL FIGURE — averaged across 4 chosen metrics ──────────────────────
#
#  The 4 metrics the user wants averaged:
#    • Accuracy  → (factoid_acc + yesno_acc) / 2   (two accuracy metrics)
#    • MRR       → factoid_mrr
#    • F1        → list_f1
#    • LLM       → summary_llm
#
#  Global score = mean of the 4 above (each already in [0,1]).
# ─────────────────────────────────────────────────────────────────────────────

accuracy_avg = (DATA["factoid_acc"] + DATA["yesno_acc"]) / 2
mrr_vals     = DATA["factoid_mrr"]
f1_vals      = DATA["list_f1"]
llm_vals     = DATA["summary_llm"]

global_score = (accuracy_avg + mrr_vals + f1_vals + llm_vals) / 4   # shape (5,5)

fig2, ax2 = plt.subplots(figsize=(10, 5.5), constrained_layout=True)

for i, (model, color, marker, ls) in enumerate(
    zip(MODELS, PALETTE, MARKERS, LINE_STYLES)
):
    ax2.plot(
        x, global_score[i],
        color=color, marker=marker, linestyle=ls,
        linewidth=2.3, markersize=7.5, label=model.replace("\n", " "),
    )

ax2.set_title(
    "Global Score — Average of Accuracy · MRR · F1 · LLM-as-Judge",
    fontsize=13, fontweight="bold", pad=10
)
ax2.set_xlabel("k (retrieved docs)", fontsize=11)
ax2.set_ylabel("Averaged Score", fontsize=11)
ax2.set_xticks(K_VALUES)
ax2.set_xlim(0.5, 21)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3f}"))

ax2.legend(
    title="Pipeline", title_fontsize=11, fontsize=10,
    loc="upper right", frameon=True, framealpha=0.9, edgecolor="#cccccc",
)

# Annotate best k per model
for i, color in enumerate(PALETTE):
    best_k_idx = np.argmax(global_score[i])
    best_val   = global_score[i][best_k_idx]
    ax2.annotate(
        f"k={K_VALUES[best_k_idx]}",
        xy=(K_VALUES[best_k_idx], best_val),
        xytext=(4, 6), textcoords="offset points",
        fontsize=7.5, color=color, fontweight="bold",
    )

plt.savefig("scripts/global_score.png",
            dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print("✓ global_score.png saved")

# ── 6. ALSO save the script itself ───────────────────────────────────────────
import shutil, os
shutil.copy(__file__, "scripts/plot_retrieval_metrics.py")
print("✓ script copied to outputs")