"""
BM25 retrieval — BioASQ test set, full corpus (257 907 docs).
Produces a dual bar chart (Recall@K + Precision@K) for the test set,
then prints a combined comparison table: test vs training set.

BM25 params : k1=0.7, b=0.9  (Pyserini / Lucene)
Index       : data/bm25_indexing_full/corpus_full/lucene_index

Test set
  queries : data/bioasq/raw/Task13BGoldenEnriched/13B{1..4}_golden.json
Training set
  queries : data/bioasq/processed/queries.jsonl
  qrels   : data/bioasq/processed/qrels.tsv

Output : data/bioasq/bm25_doc/images/recall_precision_at_k_test_full_corpus.png
"""

import os, json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"]      = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")
from pyserini.search.lucene import LuceneSearcher

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE     = Path(__file__).resolve().parents[3]
TEST_DIR = BASE / "data/bioasq/raw/Task13BGoldenEnriched"
TRAIN_Q  = BASE / "data/bioasq/processed/queries.jsonl"
TRAIN_QR = BASE / "data/bioasq/processed/qrels.tsv"
INDEX    = BASE / "data/bm25_indexing_full/corpus_full/lucene_index"
OUT_DIR  = BASE / "data/bioasq/bm25_doc/images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATCHES = ["13B1", "13B2", "13B3", "13B4"]
KS      = [10, 20, 50, 100]
K1, B   = 0.7, 0.9


# ── BM25 searcher ─────────────────────────────────────────────────────────────
print(f"Opening BM25 index (k1={K1}, b={B}) …")
searcher = LuceneSearcher(str(INDEX))
searcher.set_bm25(k1=K1, b=B)
max_k = max(KS)


def retrieve(queries: dict[str, str]) -> dict[str, list[str]]:
    results = {}
    total = len(queries)
    for i, (qid, text) in enumerate(queries.items(), 1):
        hits = searcher.search(text, k=max_k)
        results[qid] = [h.docid for h in hits]
        if i % 200 == 0:
            print(f"  {i}/{total}")
    return results


def compute_metrics(results: dict[str, list[str]],
                    qrels: dict[str, set]) -> dict[str, dict[int, float]]:
    recall_sum = {k: 0.0 for k in KS}
    prec_sum   = {k: 0.0 for k in KS}
    n = 0
    for qid, ranked in results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        for k in KS:
            hits = sum(1 for d in ranked[:k] if d in gold)
            recall_sum[k] += hits / len(gold)
            prec_sum[k]   += hits / k
    return {
        "recall":    {k: 100 * recall_sum[k] / n for k in KS},
        "precision": {k: 100 * prec_sum[k]   / n for k in KS},
        "n":         n,
    }


# ── Test set ──────────────────────────────────────────────────────────────────
print("\nLoading test queries and qrels …")
test_queries: dict[str, str] = {}
test_qrels:   dict[str, set] = defaultdict(set)

for batch in BATCHES:
    golden = json.load(open(TEST_DIR / f"{batch}_golden.json"))
    for q in golden["questions"]:
        qid = q["id"]
        test_queries[qid] = q["body"]
        for url in q.get("documents", []):
            test_qrels[qid].add(url.split("/")[-1])

test_queries = {qid: t for qid, t in test_queries.items() if test_qrels[qid]}
print(f"  {len(test_queries)} test queries with ≥1 relevant doc")

print("Retrieving test set …")
test_results = retrieve(test_queries)
test_m = compute_metrics(test_results, test_qrels)

# ── Training set ──────────────────────────────────────────────────────────────
print("\nLoading training queries and qrels …")
train_queries: dict[str, str] = {}
with open(TRAIN_Q) as f:
    for line in f:
        q = json.loads(line)
        train_queries[q["_id"]] = q["text"]

train_qrels: dict[str, set] = defaultdict(set)
with open(TRAIN_QR) as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 3 and int(parts[-1]) > 0:
            train_qrels[parts[0]].add(parts[1])

train_queries = {qid: t for qid, t in train_queries.items() if train_qrels[qid]}
print(f"  {len(train_queries)} training queries with ≥1 relevant doc")

print("Retrieving training set …")
train_results = retrieve(train_queries)
train_m = compute_metrics(train_results, train_qrels)


# ── Combined table ────────────────────────────────────────────────────────────
def _pct(v): return f"{v:.2f}%"

print(f"\n{'═'*72}")
print(f"  BM25  k1={K1}  b={B}  —  Full corpus (257 907 docs)")
print(f"{'═'*72}")
print(f"  {'':12}" +
      "".join(f"  {'@'+str(k):>8}" for k in KS))

for label, m in [("TEST", test_m), ("TRAIN", train_m)]:
    print(f"{'─'*72}")
    print(f"  {label+' Recall':<12}" +
          "".join(f"  {_pct(m['recall'][k]):>8}" for k in KS))
    print(f"  {label+' Prec':<12}" +
          "".join(f"  {_pct(m['precision'][k]):>8}" for k in KS))

print(f"{'═'*72}")
print(f"  Test  queries : {test_m['n']}   |   Train queries : {train_m['n']}")


# ── Dual bar chart — test set ─────────────────────────────────────────────────
x         = np.arange(len(KS))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars_r = ax.bar(x - bar_width / 2,
                [test_m["recall"][k]    for k in KS], bar_width,
                label="Mean Recall@K",    color="#4C72B0", alpha=0.88)
bars_p = ax.bar(x + bar_width / 2,
                [test_m["precision"][k] for k in KS], bar_width,
                label="Mean Precision@K", color="#DD8452", alpha=0.88)

for bar, val in zip(bars_r, [test_m["recall"][k] for k in KS]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#4C72B0")

for bar, val in zip(bars_p, [test_m["precision"][k] for k in KS]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#DD8452")

ax.set_xticks(x)
ax.set_xticklabels([f"@{k}" for k in KS], fontsize=11)
ax.set_xlabel("Top-K cutoff", fontsize=12)
ax.set_ylabel("Mean %", fontsize=12)
ax.set_ylim(0, 110)
ax.set_title(
    f"BM25 — Mean Recall@K vs Mean Precision@K\n"
    f"(k1={K1}, b={B})  [BioASQ test set — full corpus, 257 907 docs, n={test_m['n']}]",
    fontsize=13, pad=12,
)
ax.legend(fontsize=11)
plt.tight_layout()
out = OUT_DIR / "recall_precision_at_k_test_full_corpus.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved → {out}")
