"""
Compare BM25 Recall@K: processed corpus index (corpus_full_processed) vs full corpus index.

Doc corpus  → Pyserini: data/bm25_indexing_full/corpus_full_processed/lucene_index
Full corpus → Pyserini: data/bm25_indexing_full/corpus_full/lucene_index

Writes:
  data/bioasq/bm25_doc/images/compare_recall_doc_vs_full_corpus.png

Usage:
    python data/bioasq/bm25_doc/compare_recall_doc_vs_full_corpus.py
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────────
BASE              = Path(__file__).resolve().parents[3]
QUERIES_F         = BASE / "data" / "bioasq" / "processed" / "queries.jsonl"
QRELS_F           = BASE / "data" / "bioasq" / "processed" / "qrels.tsv"
LUCENE_DOC        = BASE / "data" / "bm25_indexing" / "lucene_index"
LUCENE_FULL       = BASE / "data" / "bm25_indexing_full" / "corpus_full_processed" / "lucene_index"
OUT_DIR           = BASE / "data" / "bioasq" / "bm25_doc" / "images"

CUTOFFS = (5, 10, 20, 50, 100)
K1, B   = 0.7, 0.9


def load_queries() -> dict[str, str]:
    queries = {}
    with QUERIES_F.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels() -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with QRELS_F.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            if int(score) > 0:
                qrels[qid].add(did)
    return qrels


def recall_at_k(results: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[int, float]:
    sums = {k: 0.0 for k in CUTOFFS}
    n = 0
    for qid, ranked in results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        for k in CUTOFFS:
            hits = sum(1 for did in ranked[:k] if did in gold)
            sums[k] += hits / len(gold)
    return {k: 100.0 * sums[k] / n for k in CUTOFFS}


def retrieve(searcher: LuceneSearcher, queries: dict[str, str], label: str) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    for qid, qtext in tqdm(queries.items(), desc=f"  Retrieving ({label})"):
        hits = searcher.search(qtext, k=max(CUTOFFS))
        results[qid] = [h.docid for h in hits]
    return results


queries = load_queries()
qrels   = load_qrels()
print(f"Loaded {len(queries):,} queries, {sum(len(v) for v in qrels.values()):,} relevant pairs\n")

# ── 1. Doc corpus (corpus_full_processed index) ───────────────────────────────
print("=== Doc corpus (corpus_full_processed index) ===")
searcher_doc = LuceneSearcher(str(LUCENE_DOC))
searcher_doc.set_bm25(k1=K1, b=B)
results_doc  = retrieve(searcher_doc, queries, "doc corpus")
recall_doc   = recall_at_k(results_doc, qrels)
print("  Recall@K:", {k: f"{v:.1f}%" for k, v in recall_doc.items()})

# ── 2. Full corpus (corpus_full index) ────────────────────────────────────────
print("\n=== Full corpus (corpus_full index, 257K docs) ===")
searcher_full = LuceneSearcher(str(LUCENE_FULL))
searcher_full.set_bm25(k1=K1, b=B)
results_full  = retrieve(searcher_full, queries, "full corpus")
recall_full   = recall_at_k(results_full, qrels)
print("  Recall@K:", {k: f"{v:.1f}%" for k, v in recall_full.items()})

# ── 3. Print comparison table ─────────────────────────────────────────────────
print(f"\n{'─'*52}")
print(f"  {'K':<6}  {'Doc (49K)':>10}  {'Full (257K)':>12}  {'Diff':>8}")
print(f"  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*8}")
diffs = []
for k in CUTOFFS:
    d = recall_full[k] - recall_doc[k]
    diffs.append(d)
    print(f"  {k:<6}  {recall_doc[k]:>9.1f}%  {recall_full[k]:>11.1f}%  {d:>+7.1f}%")
mean_diff = np.mean(diffs)
print(f"  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*8}")
print(f"  {'Mean':<6}  {'':>10}  {'':>12}  {mean_diff:>+7.1f}%")
print(f"{'─'*52}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

x         = np.arange(len(CUTOFFS))
bar_width = 0.32
cutoffs   = list(CUTOFFS)

fig, ax = plt.subplots(figsize=(11, 6.5))

rec_doc_vals  = [recall_doc[k]  for k in cutoffs]
rec_full_vals = [recall_full[k] for k in cutoffs]
diff_vals     = [recall_full[k] - recall_doc[k] for k in cutoffs]

bars_doc  = ax.bar(x - bar_width / 2, rec_doc_vals,  bar_width,
                   label="Doc corpus  (49K docs)",  color="#4C72B0", alpha=0.88)
bars_full = ax.bar(x + bar_width / 2, rec_full_vals, bar_width,
                   label="Full corpus (257K docs)", color="#55A868", alpha=0.88)

# value labels
for bar, val in zip(bars_doc, rec_doc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#4C72B0")

for bar, val in zip(bars_full, rec_full_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="#2d7a4f")

# diff annotation between each pair
for i, (xi, d) in enumerate(zip(x, diff_vals)):
    mid_y = max(rec_doc_vals[i], rec_full_vals[i]) + 3.5
    sign  = "+" if d >= 0 else ""
    color = "#c0392b" if d < 0 else "#27ae60"
    ax.text(xi, mid_y, f"Δ{sign}{d:.1f}%",
            ha="center", va="bottom", fontsize=8.5,
            color=color, fontweight="bold")

# mean diff label in bottom-right corner
ax.text(0.98, 0.04,
        f"Mean Δ = {mean_diff:+.1f}%",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=10, fontweight="bold",
        color="#27ae60" if mean_diff >= 0 else "#c0392b",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

ax.set_xticks(x)
ax.set_xticklabels([f"@{k}" for k in cutoffs], fontsize=11)
ax.set_xlabel("Top-K cutoff", fontsize=12)
ax.set_ylabel("Mean Recall@K (%)", fontsize=12)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}%"))
ax.set_title(
    f"BM25 Recall@K — Doc corpus (49K) vs Full corpus (257K)\n"
    f"BioASQ training set  ·  k1={K1}, b={B}",
    fontsize=13, pad=12,
)
ax.legend(fontsize=11)

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "compare_recall_doc_vs_full_corpus.png"
plt.tight_layout()
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nPlot saved → {out_path}")
