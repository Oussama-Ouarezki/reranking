"""
Grouped Bar Chart — 4 selected models × 5 k-values
Metrics: Factoid MRR, Yes/No Accuracy, List F1, Summary LLM-as-Judge
Seaborn ggplot-style theme
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── DATA (8 models, only 4 selected below) ────────────────────────────────────

K_VALUES = [1, 3, 5, 10, 20]

ALL_MODELS = [
    "BM25 (350tok)",           # 0
    "BM25→LiT5 (350tok)",      # 1
    "LiT5 ft ep4",             # 2
    "LiT5 LoRA ep2",           # 3
    "Qwen3 pure",              # 4
    "Qwen3 ft",                # 5
    "Qwen3 pure2",             # 6
    "Qwen3+BM25 LF→LiT5",     # 7
]

DATA = {
    "factoid_mrr": np.array([
        [0.2974, 0.3332, 0.3319, 0.3860, 0.0105],
        [0.3489, 0.3704, 0.3521, 0.3196, 0.0105],
        [0.3337, 0.3258, 0.3653, 0.3372, 0.0193],
        [0.3470, 0.3526, 0.3705, 0.3256, 0.0105],
        [0.3039, 0.3254, 0.3621, 0.2596, 0.0211],
        [0.3288, 0.3296, 0.3372, 0.2930, 0.0211],
        [0.2828, 0.3398, 0.3644, 0.2816, 0.0211],
        [0.3288, 0.3296, 0.3372, 0.2930, 0.0211],
    ]),
    "yesno_acc": np.array([
        [0.8537, 0.8537, 0.8415, 0.7439, 0.1829],
        [0.9146, 0.8780, 0.8659, 0.7683, 0.1220],
        [0.9146, 0.9024, 0.8780, 0.6951, 0.1585],
        [0.9024, 0.8780, 0.8415, 0.7317, 0.1098],
        [0.9268, 0.9024, 0.8902, 0.7561, 0.1463],
        [0.9146, 0.9146, 0.8902, 0.7561, 0.1585],
        [0.9390, 0.9146, 0.8659, 0.7561, 0.1220],
        [0.9146, 0.9146, 0.8902, 0.7561, 0.1585],
    ]),
    "list_f1": np.array([
        [0.2522, 0.2425, 0.2489, 0.2327, 0.0078],
        [0.2838, 0.2996, 0.2862, 0.2544, 0.0000],
        [0.2761, 0.3050, 0.2933, 0.2593, 0.0072],
        [0.2649, 0.3000, 0.2396, 0.2350, 0.0119],
        [0.2761, 0.2880, 0.2941, 0.2246, 0.0020],
        [0.2984, 0.2880, 0.2775, 0.2341, 0.0162],
        [0.2740, 0.3021, 0.2724, 0.2385, 0.0056],
        [0.2984, 0.2880, 0.2775, 0.2341, 0.0162],
    ]),
    "summary_llm": np.array([
        [0.5455, 0.6234, 0.7013, 0.6883, 0.3506],
        [0.5844, 0.6364, 0.6623, 0.6364, 0.3896],
        [0.6104, 0.6623, 0.6883, 0.7273, 0.3766],
        [0.6364, 0.6494, 0.6623, 0.6623, 0.3636],
        [0.6234, 0.6753, 0.6623, 0.6494, 0.4026],
        [0.5844, 0.6364, 0.6234, 0.7013, 0.3896],
        [0.6234, 0.6364, 0.6494, 0.6364, 0.4026],
        [0.5844, 0.6364, 0.6234, 0.7013, 0.3896],
    ]),
}

# ── SELECT 4 MODELS ───────────────────────────────────────────────────────────
# 0: BM25, 1: BM25→LiT5, 4: Qwen3 pure, 7: Qwen3+BM25 LF→LiT5
SEL_IDX = [0, 1, 4, 7]
SEL_LABELS = [
    "BM25",
    "BM25 → LiT5",
    "Qwen3-0.6B (pure)",
    "Qwen3+BM25 LF → LiT5",
]

METRICS = {
    "factoid_mrr": "Factoid · MRR",
    "yesno_acc": "Yes/No · Accuracy",
    "list_f1": "List · F1",
    "summary_llm": "Summary · LLM Judge",
}

# Reshape data for seaborn
def prepare_dataframe(metric_key, metric_name):
    """Convert the data array to a tidy DataFrame"""
    rows = []
    matrix = DATA[metric_key]
    for k_idx, k in enumerate(K_VALUES):
        for model_idx, model_name in zip(SEL_IDX, SEL_LABELS):
            rows.append({
                'k': k,
                'Pipeline': model_name,
                'Score': matrix[model_idx][k_idx],
                'Metric': metric_name
            })
    return pd.DataFrame(rows)

# Combine all metrics
all_dfs = []
for key, name in METRICS.items():
    df = prepare_dataframe(key, name)
    all_dfs.append(df)

df_all = pd.concat(all_dfs, ignore_index=True)

# ── SEABORN GGPLOT STYLE PLOTTING ─────────────────────────────────────────────
sns.set_theme(style="darkgrid", font_scale=1.1)
sns.set_palette(["#E07B54", "#5BA8D0", "#6ECC8E", "#C97ED6"])

# Custom ggplot-style with dark background
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.facecolor': '#1C1C1E',
    'axes.facecolor': '#2A2A2E',
    'axes.labelcolor': '#CCCCCC',
    'text.color': '#CCCCCC',
    'xtick.color': '#CCCCCC',
    'ytick.color': '#CCCCCC',
    'grid.color': '#444448',
    'legend.facecolor': '#2A2A2E',
    'legend.edgecolor': '#555555',
})

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 11))
axes = axes.flatten()

for idx, (metric_name, metric_label) in enumerate(METRICS.items()):
    ax = axes[idx]
    
    # Filter data for this metric
    df_subset = df_all[df_all['Metric'] == metric_label]
    
    # Create grouped bar plot
    sns.barplot(
        data=df_subset,
        x='k',
        y='Score',
        hue='Pipeline',
        ax=ax,
        errorbar=None,
        width=0.7,
        palette=["#E07B54", "#5BA8D0", "#6ECC8E", "#C97ED6"],
        edgecolor='none',
        alpha=0.88
    )
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=6.5, 
                    fontweight='bold', color='white',
                    padding=3)
    
    # Customize appearance
    ax.set_title(metric_label, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel('k-value', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_xticklabels([f'k={k}' for k in K_VALUES], fontsize=10)
    
    # Adjust y-limit to accommodate labels
    ymax = df_subset['Score'].max()
    ax.set_ylim(0, ymax * 1.18)
    
    # Improve grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    
    # Move legend to better position
    if idx == 0:  # Legend only on first subplot
        ax.legend(title='Pipeline', title_fontsize=11, fontsize=10,
                 loc='upper right', frameon=True, framealpha=0.15)
    else:
        ax.legend_.remove()

# Add overall title
fig.suptitle('Retrieval Pipeline Comparison — Grouped by k', 
             fontsize=16, fontweight='bold', y=1.01)

# Add shared legend explanation at bottom
fig.text(0.5, 0.01, 'Metrics evaluated across different k-values (top-k retrieved documents)',
         ha='center', fontsize=10, style='italic', color='#AAAAAA')

plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.savefig('scripts/bar_metrics_comparison_seaborn.png', 
            dpi=180, bbox_inches='tight', facecolor='#1C1C1E')
plt.close()

print("✓ bar_metrics_comparison_seaborn.png saved")

# Also create a facet grid version (alternative visualization)
print("\nCreating alternative facet grid visualization...")

# Facet grid version (more compact, shows all metrics together)
g = sns.catplot(
    data=df_all,
    kind='bar',
    x='k',
    y='Score',
    hue='Pipeline',
    col='Metric',
    col_wrap=2,
    height=5,
    aspect=1.5,
    palette=["#E07B54", "#5BA8D0", "#6ECC8E", "#C97ED6"],
    errorbar=None,
    width=0.7,
    edgecolor='none',
    alpha=0.88,
    sharey=False
)

# Customize the facet grid
g.fig.set_size_inches(18, 11)
g.fig.patch.set_facecolor('#1C1C1E')
g.fig.suptitle('Retrieval Pipeline Comparison — All Metrics', 
               fontsize=16, fontweight='bold', y=1.02)

for ax in g.axes.flat:
    ax.set_facecolor('#2A2A2E')
    ax.set_xlabel('k-value', fontsize=10)
    ax.set_ylabel('Score', fontsize=10)
    ax.tick_params(colors='#CCCCCC', labelsize=9)
    ax.set_xticklabels([f'k={k}' for k in K_VALUES], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=6, 
                    fontweight='bold', color='white',
                    padding=2)
    
    # Adjust y-limits
    ymax = max([c.get_height() for c in container]) if container else 0
    ax.set_ylim(0, ymax * 1.15)

# Adjust legend
g.add_legend(title='Pipeline', title_fontsize=11, fontsize=10,
             frameon=True, framealpha=0.15, labelcolor='white')
plt.setp(g._legend.get_texts(), color='white')
plt.setp(g._legend.get_title(), color='white')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('scripts/bar_metrics_comparison_facetgrid.png', 
            dpi=180, bbox_inches='tight', facecolor='#1C1C1E')
plt.close()

print("✓ bar_metrics_comparison_facetgrid.png saved")
print("\n✅ All seaborn plots generated successfully!")