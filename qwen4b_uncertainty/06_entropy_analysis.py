"""H@50 entropy distribution analysis for Qwen3-4B pointwise scores.

Mirrors monoLiT5gate/entropy_analysis.py — but with the *correct* entropy:
divide scores by their sum (no second softmax), Shannon entropy, normalised by log(k).

Plots
  1. Global H@50 distribution (hist + KDE)
  2. H@50 distribution per question type (4-panel hist + KDE)
  3. Violin overlay across types
  4. Two example queries: one low-H50, one high-H50 — bar chart of P(yes) over 50 docs

Reads:
  qwen4b_uncertainty/data/qwen_scores.jsonl
  qwen4b_uncertainty/data/queries_500.jsonl       (for query text)

Writes:
  qwen4b_uncertainty/plots/entropy_h50_global.png
  qwen4b_uncertainty/plots/entropy_h50_by_qtype.png
  qwen4b_uncertainty/plots/entropy_h50_violin.png
  qwen4b_uncertainty/plots/entropy_example_queries.png
"""

import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores.jsonl"
QUERIES_F = BASE / "qwen4b_uncertainty/data/queries_500.jsonl"
OUT_DIR = BASE / "qwen4b_uncertainty/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

QTYPE_COLORS = {
    "factoid": "#4c72b0",
    "list": "#dd8452",
    "yesno": "#2ca02c",
    "summary": "#d62728",
}
ORDERED_TYPES = ["factoid", "list", "yesno", "summary"]


def norm_entropy(vals: list[float]) -> float:
    s = sum(vals)
    if s == 0 or len(vals) < 2:
        return 0.0
    probs = [v / s for v in vals]
    h = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
    return h / math.log(len(probs))


def main() -> None:
    print("Loading scores …")
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))

    qtext: dict[str, str] = {}
    with QUERIES_F.open() as f:
        for line in f:
            q = json.loads(line)
            qtext[q["_id"]] = q["text"]

    entropies: dict[str, float] = {}
    qtypes: dict[str, str] = {}
    sorted_scores: dict[str, list[tuple[str, float]]] = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        ranked = sorted(
            ((s["docid"], s["qwen_prob"]) for s in r["scores"]),
            key=lambda x: x[1], reverse=True,
        )
        sorted_scores[qid] = ranked
        vals = [s for _, s in ranked[:50]]
        entropies[qid] = norm_entropy(vals)

    qids = list(entropies.keys())
    all_h = [entropies[q] for q in qids]
    print(f"H@50 over {len(qids)} queries: "
          f"mean={np.mean(all_h):.4f}  median={np.median(all_h):.4f}  "
          f"std={np.std(all_h):.4f}  min={np.min(all_h):.4f}  max={np.max(all_h):.4f}")

    # ── plot 1: global ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_h, bins=40, color="#4c72b0", edgecolor="white", alpha=0.85,
            density=True, label="H@50 density")
    sns.kdeplot(all_h, ax=ax, color="#d62728", lw=2.2, label="KDE")
    ax.axvline(np.mean(all_h), color="black", ls="--", lw=1.5,
               label=f"mean = {np.mean(all_h):.3f}")
    ax.axvline(np.median(all_h), color="gray", ls=":", lw=1.5,
               label=f"median = {np.median(all_h):.3f}")
    ax.set_xlabel("Normalised H@50 entropy", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"H@50 entropy — Qwen3-4B P(yes) over 50 BM25 docs  ({len(qids)} queries)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = OUT_DIR / "entropy_h50_global.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── plot 2: per qtype ────────────────────────────────────────────────────
    type_groups: dict[str, list[float]] = {t: [] for t in ORDERED_TYPES}
    for q in qids:
        type_groups[qtypes[q]].append(entropies[q])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, t in zip(axes, ORDERED_TYPES):
        v = type_groups[t]
        col = QTYPE_COLORS[t]
        ax.hist(v, bins=20, color=col, edgecolor="white", alpha=0.8,
                density=True, label=f"n={len(v)}")
        sns.kdeplot(v, ax=ax, color="black", lw=2, label="KDE")
        ax.axvline(np.mean(v), color="black", ls="--", lw=1.3,
                   label=f"mean={np.mean(v):.3f}")
        ax.axvline(np.median(v), color="gray", ls=":", lw=1.3,
                   label=f"median={np.median(v):.3f}")
        ax.set_xlabel("Normalised H@50", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{t.capitalize()}  (n={len(v)})", color=col,
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle("H@50 entropy by question type", fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = OUT_DIR / "entropy_h50_by_qtype.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── plot 3: violin ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [type_groups[t] for t in ORDERED_TYPES]
    parts = ax.violinplot(data, positions=range(len(ORDERED_TYPES)),
                          showmedians=True, showextrema=True)
    for i, t in enumerate(ORDERED_TYPES):
        parts["bodies"][i].set_facecolor(QTYPE_COLORS[t])
        parts["bodies"][i].set_alpha(0.65)
    parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
    for k in ("cbars", "cmins", "cmaxes"):
        parts[k].set_color("black")
    ax.set_xticks(range(len(ORDERED_TYPES)))
    ax.set_xticklabels([f"{t}\n(n={len(type_groups[t])})" for t in ORDERED_TYPES])
    ax.set_ylabel("Normalised H@50 entropy", fontsize=12)
    ax.set_title("H@50 entropy by question type (violin)", fontsize=13, fontweight="bold")
    ax.legend(handles=[mpatches.Patch(color=QTYPE_COLORS[t], label=t) for t in ORDERED_TYPES],
              fontsize=9)
    fig.tight_layout()
    p = OUT_DIR / "entropy_h50_violin.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── plot 4: low/high entropy examples ──────────────────────────────────
    interior = sorted([q for q in qids if 0.001 < entropies[q] < 0.9999],
                      key=lambda q: entropies[q])
    low_q = interior[int(len(interior) * 0.10)]
    high_q = interior[int(len(interior) * 0.90)]

    print(f"\nLow-H  query: {low_q}  H={entropies[low_q]:.4f}  type={qtypes[low_q]}")
    print(f"  text: {qtext.get(low_q, '?')[:100]}")
    print(f"High-H query: {high_q}  H={entropies[high_q]:.4f}  type={qtypes[high_q]}")
    print(f"  text: {qtext.get(high_q, '?')[:100]}")

    def example_panel(ax, qid, label, color):
        ranked = sorted_scores[qid]
        n = len(ranked)
        scores = [s for _, s in ranked]
        x = np.arange(n)
        ax.bar(x, scores, color=color, alpha=0.78, edgecolor="white", width=0.75)
        ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.7, label="P(yes)=0.5")
        ax.set_ylim(0, 1.08)
        ax.set_xlim(-0.8, n - 0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"d{i+1}" for i in range(n)], fontsize=6, rotation=90)
        ax.set_xlabel("Document (ranked by Qwen P(yes))", fontsize=10)
        ax.set_ylabel("Qwen P(yes)", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        qt = qtext.get(qid, qid)
        wrapped = (qt[:90] + "…") if len(qt) > 90 else qt
        ax.set_title(
            f"{label}   H@50 = {entropies[qid]:.4f}   [{qtypes[qid]}]\n\"{wrapped}\"",
            fontsize=10, fontweight="bold", loc="left", pad=8,
        )

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(18, 10))
    example_panel(a1, low_q, "Low Entropy", "#4c72b0")
    example_panel(a2, high_q, "High Entropy", "#d62728")
    fig.suptitle(
        "Qwen3-4B P(yes) across 50 BM25 candidates\n"
        "Low H@50 → confident ordering   |   High H@50 → uncertain / flat",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = OUT_DIR / "entropy_example_queries.png"
    fig.savefig(p, dpi=160); plt.close(fig)
    print(f"  → {p}")

    # ── stats ───────────────────────────────────────────────────────────────
    print("\n── H@50 stats by qtype ──")
    for t in ORDERED_TYPES:
        v = type_groups[t]
        print(f"  {t:<8}  n={len(v):>3}  mean={np.mean(v):.4f}  "
              f"median={np.median(v):.4f}  std={np.std(v):.4f}")


if __name__ == "__main__":
    main()
