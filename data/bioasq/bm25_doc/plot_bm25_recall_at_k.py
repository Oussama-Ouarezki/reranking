"""
Plot BM25 Recall@K curve on BioASQ Task13BGoldenEnriched.
Saves: evaluation/bm25_recall_at_k.png
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from rank_bm25 import BM25Okapi
from tqdm import tqdm

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/bioasq/raw/Task13BGoldenEnriched")
CORPUS_FILE  = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"
QRELS_FILE   = DATA_DIR / "qrels.tsv"
OUT_PNG      = Path("evaluation/bm25_recall_at_k.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
MAX_K    = max(K_VALUES)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading corpus …")
corpus, doc_ids = {}, []
with CORPUS_FILE.open() as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = (doc.get("title", "") + " " + doc["text"]).strip()
        doc_ids.append(doc["_id"])
print(f"  {len(corpus):,} documents")

print("Loading queries …")
queries = {}
with QUERIES_FILE.open() as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q["text"]
print(f"  {len(queries):,} queries")

print("Loading qrels …")
relevant: dict[str, set[str]] = defaultdict(set)
with QRELS_FILE.open() as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            qid, doc_id, score = parts
        elif len(parts) == 4:
            qid, _, doc_id, score = parts
        else:
            continue
        if int(score) > 0:
            relevant[qid].add(doc_id)
print(f"  {sum(len(v) for v in relevant.values()):,} relevant pairs across {len(relevant):,} queries")

# ── BM25 index ────────────────────────────────────────────────────────────────
print("\nBuilding BM25 index …")
tokenized_corpus = [corpus[d].lower().split() for d in doc_ids]
bm25 = BM25Okapi(tokenized_corpus, k1=0.9, b=0.4)
print("  BM25 index ready.")

# ── Compute Recall@K per query ────────────────────────────────────────────────
print(f"\nRetrieving top-{MAX_K} for each query …")
recall_at_k: dict[int, list[float]] = defaultdict(list)

for qid, qtext in tqdm(queries.items(), desc="BM25"):
    rel = relevant.get(qid, set())
    if not rel:
        continue

    scores   = bm25.get_scores(qtext.lower().split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:MAX_K]
    retrieved = [doc_ids[i] for i in top_idxs]

    hits = 0
    retrieved_set = set()
    for rank, doc_id in enumerate(retrieved, start=1):
        retrieved_set.add(doc_id)
        if doc_id in rel:
            hits += 1
        if rank in K_VALUES:
            recall_at_k[rank].append(hits / len(rel))

mean_recall = {k: sum(v) / len(v) for k, v in recall_at_k.items()}

# ── Print table ───────────────────────────────────────────────────────────────
print("\n  K       Recall@K")
print("  ─────   ────────")
for k in K_VALUES:
    print(f"  {k:<6}  {mean_recall[k]:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
ks      = K_VALUES
recalls = [mean_recall[k] for k in ks]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, recalls, marker="o", linewidth=2, markersize=6, color="#2196F3", label="BM25 (k1=0.9, b=0.4)")

for k, r in zip(ks, recalls):
    ax.annotate(f"{r:.3f}", (k, r), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

ax.set_xscale("log")
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.set_xlabel("K (number of retrieved documents)", fontsize=12)
ax.set_ylabel("Mean Recall@K", fontsize=12)
ax.set_title("BM25 Recall@K — BioASQ Task13BGoldenEnriched", fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nPlot saved → {OUT_PNG}")
