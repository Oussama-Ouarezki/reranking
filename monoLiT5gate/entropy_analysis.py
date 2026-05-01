"""H@50 entropy analysis for monoLiT5gate.

Produces:
  1. Global H@50 distribution across all queries (histogram + KDE)
  2. H@50 distribution per question type (factoid / list / yesno / summary)
  3. Two example queries: one high-entropy, one low-entropy — bar chart of
     monoT5 P(true) scores across all 50 docs with their titles

Caches / data
  score cache : models/monot5/margin_scores_cache.json   (50 docs × P(true))
  mono cache  : application/cache/runs/monot5/20260427T171106Z.json  (qtype)
  queries     : data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl
  corpus      : data/bioasq/pubmed_full/full/corpus_full.jsonl
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ── theme ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
SCORE_CACHE = ROOT / "models/monot5/margin_scores_cache.json"
MONO_CACHE  = ROOT / "application/cache/runs/monot5/20260427T171106Z.json"
QUERIES_F   = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl"
CORPUS_F    = ROOT / "data/bioasq/pubmed_full/full/corpus_full.jsonl"

OUT_DIR = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading score cache …")
scores_raw = json.loads(SCORE_CACHE.read_text())

print("Loading mono cache (qtypes) …")
mono_run = json.loads(MONO_CACHE.read_text())
mono_pq  = mono_run["per_query"]

print("Loading queries …")
queries: dict[str, dict] = {}
with open(QUERIES_F) as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q

print(f"Loading corpus titles ({(CORPUS_F.stat().st_size / 1e6):.0f} MB) …")
doc_titles: dict[str, str] = {}
with open(CORPUS_F) as f:
    for line in f:
        d = json.loads(line)
        doc_titles[d["_id"]] = d.get("title", "")
print(f"  {len(doc_titles):,} titles loaded")

# ── compute H@50 per query ────────────────────────────────────────────────────

def norm_entropy(vals: list[float]) -> float:
    s = sum(vals)
    if s == 0 or len(vals) < 2:
        return 0.0
    probs = [v / s for v in vals]
    h = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
    return h / math.log(len(probs))


qids = sorted(set(scores_raw) & set(mono_pq))
print(f"Shared queries: {len(qids)}")

entropies:   dict[str, float] = {}
qtypes:      dict[str, str]   = {}
sorted_docs: dict[str, list[tuple[str, float]]] = {}  # qid → [(docid, score)] desc

for qid in qids:
    doc_scores = scores_raw[qid]
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_docs[qid] = ranked
    vals = [s for _, s in ranked[:50]]
    entropies[qid] = norm_entropy(vals)
    qtypes[qid] = mono_pq[qid].get("qtype") or queries.get(qid, {}).get("type", "unknown")

QTYPE_COLORS = {
    "factoid": "#4c72b0",
    "list":    "#dd8452",
    "yesno":   "#2ca02c",
    "summary": "#d62728",
    "unknown": "#9467bd",
}

# ── pick example queries (exclude exact 0 and 1) ─────────────────────────────
# Use p10 and p90 of the strictly-interior distribution so neither example
# is a degenerate spike or perfectly flat distribution.
interior = sorted([q for q in qids if 0.01 < entropies[q] < 0.99],
                  key=lambda q: entropies[q])
low_qid  = interior[int(len(interior) * 0.10)]   # ~p10 of interior
high_qid = interior[int(len(interior) * 0.90)]   # ~p90 of interior

print(f"\nLow-entropy  query : {low_qid}  H={entropies[low_qid]:.4f}  type={qtypes[low_qid]}")
print(f"  text: {queries.get(low_qid, {}).get('text', '?')[:100]}")
print(f"High-entropy query : {high_qid}  H={entropies[high_qid]:.4f}  type={qtypes[high_qid]}")
print(f"  text: {queries.get(high_qid, {}).get('text', '?')[:100]}")

# ── plot 1: global H@50 distribution ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 5))
all_h = list(entropies.values())
ax1.hist(all_h, bins=40, color="#4c72b0", edgecolor="white", alpha=0.85,
         density=True, label="H@50 density")
sns.kdeplot(all_h, ax=ax1, color="#d62728", lw=2.2, label="KDE")
ax1.axvline(np.mean(all_h),   color="black", ls="--", lw=1.5,
            label=f"mean = {np.mean(all_h):.3f}")
ax1.axvline(np.median(all_h), color="gray",  ls=":",  lw=1.5,
            label=f"median = {np.median(all_h):.3f}")
ax1.set_xlabel("Normalised H@50 entropy", fontsize=12)
ax1.set_ylabel("Density", fontsize=12)
ax1.set_title(f"H@50 entropy distribution — all {len(qids)} queries", fontsize=13, fontweight="bold")
ax1.legend(fontsize=9)
fig1.tight_layout()
p1 = OUT_DIR / "entropy_h50_global.png"
fig1.savefig(p1, dpi=160); plt.close(fig1)
print(f"\nPlot → {p1}")

# ── plot 2: H@50 distribution per question type ───────────────────────────────
type_groups: dict[str, list[float]] = {}
for qid in qids:
    t = qtypes[qid]
    type_groups.setdefault(t, []).append(entropies[qid])

ordered_types = [t for t in ["factoid", "list", "yesno", "summary"] if t in type_groups]
n_types = len(ordered_types)

fig2, axes2 = plt.subplots(1, n_types, figsize=(5 * n_types, 5), sharey=False)
if n_types == 1:
    axes2 = [axes2]

for ax, qt in zip(axes2, ordered_types):
    vals = type_groups[qt]
    col  = QTYPE_COLORS.get(qt, "#9467bd")
    ax.hist(vals, bins=25, color=col, edgecolor="white", alpha=0.8,
            density=True, label=f"n={len(vals)}")
    sns.kdeplot(vals, ax=ax, color="black", lw=2, label="KDE")
    ax.axvline(np.mean(vals),   color="black", ls="--", lw=1.3,
               label=f"mean={np.mean(vals):.3f}")
    ax.axvline(np.median(vals), color="gray",  ls=":",  lw=1.3,
               label=f"median={np.median(vals):.3f}")
    ax.set_xlabel("Normalised H@50", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"{qt.capitalize()}  (n={len(vals)})", fontsize=12, fontweight="bold",
                 color=col)
    ax.legend(fontsize=8)

fig2.suptitle("H@50 entropy distribution by question type", fontsize=14, fontweight="bold")
fig2.tight_layout()
p2 = OUT_DIR / "entropy_h50_by_qtype.png"
fig2.savefig(p2, dpi=160); plt.close(fig2)
print(f"Plot → {p2}")

# ── plot 3: violin / box overlay of all types together ────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 5))
data_for_violin = [type_groups[t] for t in ordered_types]
parts = ax3.violinplot(data_for_violin, positions=range(n_types),
                       showmedians=True, showextrema=True)
for i, (pc, qt) in enumerate(zip(parts["bodies"], ordered_types)):
    col = QTYPE_COLORS.get(qt, "#9467bd")
    pc.set_facecolor(col); pc.set_alpha(0.65)
parts["cmedians"].set_color("black"); parts["cmedians"].set_linewidth(2)
parts["cbars"].set_color("black");    parts["cmins"].set_color("black")
parts["cmaxes"].set_color("black")

ax3.set_xticks(range(n_types))
ax3.set_xticklabels([f"{t}\n(n={len(type_groups[t])})" for t in ordered_types], fontsize=11)
ax3.set_ylabel("Normalised H@50 entropy", fontsize=12)
ax3.set_title("H@50 entropy distribution by question type (violin)", fontsize=13, fontweight="bold")
patches = [mpatches.Patch(color=QTYPE_COLORS.get(t, "#9467bd"), label=t) for t in ordered_types]
ax3.legend(handles=patches, fontsize=9)
fig3.tight_layout()
p3 = OUT_DIR / "entropy_h50_violin.png"
fig3.savefig(p3, dpi=160); plt.close(fig3)
print(f"Plot → {p3}")

# ── plot 4: two example queries — vertical bar chart, docs labelled d1…d50 ──

def _example_plot(ax: plt.Axes, qid: str, label: str, bar_color: str) -> None:
    ranked     = sorted_docs[qid]
    query_text = queries.get(qid, {}).get("text", qid)
    qt         = qtypes[qid]
    h          = entropies[qid]

    n      = len(ranked)
    scores = [s for _, s in ranked]
    xlabels = [f"d{i+1}" for i in range(n)]
    x       = np.arange(n)

    bars = ax.bar(x, scores, color=bar_color, alpha=0.78, edgecolor="white", width=0.75)

    ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.7, label="P(true)=0.5")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.8, n - 0.2)

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6, rotation=90)
    ax.set_xlabel("Document (ranked by monoT5 P(true))", fontsize=10)
    ax.set_ylabel("monoT5 P(true)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")

    wrapped_q = (query_text[:90] + "…") if len(query_text) > 90 else query_text
    ax.set_title(
        f"{label}   H@50 = {h:.4f}   [{qt}]\n\"{wrapped_q}\"",
        fontsize=10, fontweight="bold", loc="left", pad=8
    )


fig4, (ax_low, ax_high) = plt.subplots(2, 1, figsize=(18, 10))
_example_plot(ax_low,  low_qid,  "Low Entropy",  "#4c72b0")
_example_plot(ax_high, high_qid, "High Entropy", "#d62728")

fig4.suptitle(
    "monoT5 P(true) score distribution across 50 BM25 candidates\n"
    "Low H@50 → confident ordering   |   High H@50 → uncertain / flat",
    fontsize=13, fontweight="bold"
)
fig4.tight_layout(rect=(0, 0, 1, 0.95))
p4 = OUT_DIR / "entropy_example_queries.png"
fig4.savefig(p4, dpi=160); plt.close(fig4)
print(f"Plot → {p4}")

# ── print summary stats ───────────────────────────────────────────────────────
print("\n── H@50 global stats ──────────────────────────────────────────────")
print(f"  n queries : {len(qids)}")
print(f"  mean      : {np.mean(all_h):.4f}")
print(f"  median    : {np.median(all_h):.4f}")
print(f"  std       : {np.std(all_h):.4f}")
print(f"  min       : {np.min(all_h):.4f}")
print(f"  max       : {np.max(all_h):.4f}")
print(f"  τ=0.832 → {sum(h >= 0.832 for h in all_h)}/{len(all_h)} queries sent to LiT5 "
      f"({sum(h >= 0.832 for h in all_h)/len(all_h)*100:.1f}%)")

print("\n── H@50 stats by question type ────────────────────────────────────")
for qt in ordered_types:
    v = type_groups[qt]
    triggered = sum(h >= 0.832 for h in v)
    print(f"  {qt:<8}  n={len(v):>3}  mean={np.mean(v):.4f}  "
          f"median={np.median(v):.4f}  std={np.std(v):.4f}  "
          f"→ LiT5: {triggered}/{len(v)} ({triggered/len(v)*100:.1f}%)")

print(f"\nAll plots saved to: {OUT_DIR}")
