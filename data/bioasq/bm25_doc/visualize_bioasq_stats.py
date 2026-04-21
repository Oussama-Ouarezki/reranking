"""
Two visualizations for BioASQ processed data:
  1. Number of queries with > N relevant documents (N = 1, 3, 5, 10)
  2. Number of queries per year (from created_at decoded in queries.jsonl)
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

QRELS   = Path("data/bioasq/processed/qrels.tsv")
QUERIES = Path("data/bioasq/processed/queries.jsonl")
OUT_DIR = Path("data/bioasq/bm25_doc/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load qrels → docs-per-query counts ────────────────────────────────────
doc_counts: Counter = Counter()
with QRELS.open() as f:
    next(f)  # skip header
    for line in f:
        qid, *_ = line.strip().split("\t")
        doc_counts[qid] += 1

thresholds = [1, 3, 5, 10]
counts_above = [sum(1 for c in doc_counts.values() if c > t) for t in thresholds]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(
    [f"> {t}" for t in thresholds],
    counts_above,
    color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
    edgecolor="white",
    width=0.5,
)
for bar, val in zip(bars, counts_above):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 15,
        f"{val:,}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
ax.set_title("Queries by minimum number of relevant documents", fontsize=13, pad=12)
ax.set_xlabel("Threshold (relevant docs per query)", fontsize=11)
ax.set_ylabel("Number of queries", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_ylim(0, max(counts_above) * 1.12)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out1 = OUT_DIR / "bioasq_queries_by_doc_threshold.png"
fig.savefig(out1, dpi=150)
print(f"Saved {out1}")

# ── 2. Load queries → year counts ────────────────────────────────────────────
year_counts: Counter = Counter()
with QUERIES.open() as f:
    for line in f:
        rec = json.loads(line)
        if "created_at" in rec:
            year = rec["created_at"][:4]
            year_counts[year] += 1

years  = sorted(year_counts)
totals = [year_counts[y] for y in years]

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
bars2 = ax2.bar(years, totals, color="#4C72B0", edgecolor="white", width=0.6)
for bar, val in zip(bars2, totals):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        f"{val:,}",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )
ax2.set_title("BioASQ queries per year", fontsize=13, pad=12)
ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("Number of queries", fontsize=11)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax2.set_ylim(0, max(totals) * 1.12)
ax2.spines[["top", "right"]].set_visible(False)
fig2.tight_layout()
out2 = OUT_DIR / "bioasq_queries_per_year.png"
fig2.savefig(out2, dpi=150)
print(f"Saved {out2}")

# ── 3. Pie chart: queries with 5+ docs per year ──────────────────────────────
qid_to_year = {}
with QUERIES.open() as f:
    for line in f:
        rec = json.loads(line)
        if "created_at" in rec:
            qid_to_year[rec["_id"]] = rec["created_at"][:4]

year_5plus: Counter = Counter()
for qid, count in doc_counts.items():
    if count >= 5 and qid in qid_to_year:
        year_5plus[qid_to_year[qid]] += 1

years_p   = sorted(year_5plus)
sizes     = [year_5plus[y] for y in years_p]
colors    = sns.color_palette("tab20", len(years_p))

fig3, ax3 = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax3.pie(
    sizes,
    labels=years_p,
    autopct=lambda p: f"{p:.1f}%" if p >= 2 else "",
    colors=colors,
    startangle=140,
    pctdistance=0.78,
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
)
for at in autotexts:
    at.set_fontsize(8.5)
    at.set_fontweight("bold")

total_5plus = sum(sizes)
ax3.set_title(
    f"Queries with 5+ relevant documents by year\n(total: {total_5plus:,})",
    fontsize=13, pad=16,
)
fig3.tight_layout()
out3 = OUT_DIR / "bioasq_5plus_docs_by_year_pie.png"
fig3.savefig(out3, dpi=150)
print(f"Saved {out3}")

# ── 4. Multi-line cumulative queries newest → oldest (all, 2+, 3+, 4+, 5+) ──
years_desc = sorted(year_counts, reverse=True)
thresholds_cum = [0, 2, 3, 4, 5]  # 0 = all queries
labels_cum     = ["all", "2+", "3+", "4+", "5+"]
palette        = sns.color_palette("tab10", len(thresholds_cum))

fig4, ax4 = plt.subplots(figsize=(10, 5.5))
for thresh, label, color in zip(thresholds_cum, labels_cum, palette):
    cumulative, running = [], 0
    for y in years_desc:
        for qid, cnt in doc_counts.items():
            if cnt > thresh and qid in qid_to_year and qid_to_year[qid] == y:
                running += 1
        cumulative.append(running)
    ax4.plot(years_desc, cumulative, marker="o", linewidth=2.2,
             markersize=6, color=color, label=label)

ax4.set_title("Cumulative queries by doc threshold (newest → oldest)", fontsize=13, pad=12)
ax4.set_xlabel("Year", fontsize=11)
ax4.set_ylabel("Cumulative number of queries", fontsize=11)
ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax4.legend(title="Min relevant docs", fontsize=10, title_fontsize=10)
ax4.spines[["top", "right"]].set_visible(False)
fig4.tight_layout()
out4 = OUT_DIR / "bioasq_cumulative_queries_newest_to_oldest.png"
fig4.savefig(out4, dpi=150)
print(f"Saved {out4}")

plt.show()
