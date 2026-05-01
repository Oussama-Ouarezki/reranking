"""
BM25 Recall Distribution — BioASQ test set (13B1–13B4), full corpus.

BM25 params : k1=0.7, b=0.9  (Pyserini / Lucene)
Index       : data/bm25_indexing_full/corpus_full/lucene_index
Qrels       : data/bioasq/raw/Task13BGoldenEnriched/13B{1,2,3,4}_golden.json
              (document URLs  →  PubMed IDs)
Recall@k    : |relevant ∩ top-k| / |relevant|  for k in {10, 20, 50, 100}

Outputs (data/bioasq/bm25_doc/images/):
  bm25_recall_histogram.png   — recall@k histogram: combined (2×2) + per-type KDE
  bm25_recall_cumulative.png  — cumulative recall@k curves per type
  bm25_recall_boxplot.png     — box/violin per type × k
"""

import os, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"]      = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

from pyserini.search.lucene import LuceneSearcher

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parents[3]
TEST_DIR = BASE / "data/bioasq/raw/Task13BGoldenEnriched"
INDEX    = BASE / "data/bm25_indexing_full/corpus_full/lucene_index"
OUT_DIR  = BASE / "data/bioasq/bm25_doc/images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCHES = ["13B1", "13B2", "13B3", "13B4"]
KS      = [10, 20, 50, 100]
TYPES   = ["factoid", "list", "yesno", "summary"]
COLORS  = {"factoid": "#4c72b0", "list": "#dd8452",
           "yesno": "#2ca02c",   "summary": "#d62728"}

# ── load qrels + query text + type from golden JSONs ─────────────────────────
print("Loading test-set queries and qrels …")
queries: dict[str, str] = {}   # qid → text
qtypes:  dict[str, str] = {}   # qid → type
qrels:   dict[str, set] = defaultdict(set)  # qid → set of PMIDs

for batch in BATCHES:
    golden = json.load(open(TEST_DIR / f"{batch}_golden.json"))
    for q in golden["questions"]:
        qid = q["id"]
        queries[qid] = q["body"]
        qtypes[qid]  = q["type"]
        for url in q.get("documents", []):
            qrels[qid].add(url.split("/")[-1])

# only keep queries that have at least one relevant doc
qids = [qid for qid in queries if len(qrels[qid]) > 0]
print(f"  {len(qids)} queries with ≥1 relevant document")

# ── BM25 retrieval ────────────────────────────────────────────────────────────
print(f"Opening index: {INDEX}")
searcher = LuceneSearcher(str(INDEX))
searcher.set_bm25(k1=0.7, b=0.9)

max_k = max(KS)

# recall_data[k][qid] = recall value
recall_data: dict[int, dict[str, float]] = {k: {} for k in KS}

print(f"Retrieving top-{max_k} for {len(qids)} queries …")
for i, qid in enumerate(qids, 1):
    if i % 50 == 0:
        print(f"  {i}/{len(qids)}")
    hits    = searcher.search(queries[qid], k=max_k)
    ret_ids = {h.docid for h in hits}
    rel_ids = qrels[qid]
    for k in KS:
        top_k_ids = {h.docid for h in hits[:k]}
        recall_data[k][qid] = len(rel_ids & top_k_ids) / len(rel_ids)

print("Retrieval done.")

# ── organise by type ──────────────────────────────────────────────────────────
# type_recall[k][type] = list of recall values
type_recall: dict[int, dict[str, list[float]]] = {
    k: {t: [] for t in TYPES} for k in KS
}
all_recall:  dict[int, list[float]] = {k: [] for k in KS}

for qid in qids:
    t = qtypes.get(qid, "unknown")
    for k in KS:
        r = recall_data[k][qid]
        all_recall[k].append(r)
        if t in type_recall[k]:
            type_recall[k][t].append(r)

# ── print summary ─────────────────────────────────────────────────────────────
print(f"\n{'─'*62}")
print(f"  {'':12}" + "".join(f"  Recall@{k:<4}" for k in KS))
print(f"  {'─'*60}")
print(f"  {'Overall':<12}" +
      "".join(f"  {np.mean(all_recall[k]):.3f}      " for k in KS))
for t in TYPES:
    print(f"  {t:<12}" +
          "".join(f"  {np.mean(type_recall[k][t]):.3f}      " for k in KS))
print(f"{'─'*62}")


# ── Plot: grouped bar chart — x=k values, groups=question types ──────────────
fig, ax = plt.subplots(figsize=(12, 6))

n_types = len(TYPES)
n_ks    = len(KS)
width   = 0.16
x       = np.arange(n_ks)

for i, t in enumerate(TYPES):
    means  = [np.mean(type_recall[k][t]) for k in KS]
    offset = (i - n_types / 2 + 0.5) * width
    bars   = ax.bar(x + offset, means, width, label=t.capitalize(),
                    color=COLORS[t], edgecolor="white", alpha=0.87)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.007,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=COLORS[t])

ax.set_xticks(x)
ax.set_xticklabels([f"Recall@{k}" for k in KS], fontsize=12)
ax.set_ylabel("Mean Recall", fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_title(
    f"BM25 Mean Recall by Question Type — BioASQ Test Set (k1=0.7, b=0.9, n={len(qids)})",
    fontsize=13, fontweight="bold"
)
ax.legend(fontsize=11, title="Question type", title_fontsize=10)
fig.tight_layout()
out = OUT_DIR / "bm25_recall_bar.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out}")
