"""
Plot BM25 Recall@K curve on the left-truncated BioASQ training set.
Saves: data/training/images/bm25_recall_at_k_train_left.png
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
from rank_bm25 import BM25Okapi

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/training/truncated_left")
CORPUS_FILE  = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"
QRELS_FILE   = DATA_DIR / "qrels.tsv"
OUT_PNG      = Path("data/training/images/bm25_recall_at_k_train_left.png")

K_VALUES = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
MAX_K    = max(K_VALUES)

# ── Load corpus ───────────────────────────────────────────────────────────────
print("Loading corpus …")
corpus: dict[str, str] = {}
doc_ids: list[str] = []
with CORPUS_FILE.open() as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = (doc.get("title", "") + " " + doc["text"]).strip()
        doc_ids.append(doc["_id"])
print(f"  {len(corpus):,} documents")

# ── BM25 index (built once) ───────────────────────────────────────────────────
print("\nBuilding BM25 index …")
tokenized_corpus = [corpus[d].lower().split() for d in doc_ids]
bm25 = BM25Okapi(tokenized_corpus, k1=0.7, b=0.9)
print("  BM25 index ready.")

K_SET = set(K_VALUES)


def load_split(queries_file: Path, qrels_file: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    queries: dict[str, str] = {}
    with queries_file.open() as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    relevant: dict[str, set[str]] = defaultdict(set)
    with qrels_file.open() as f:
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
    return queries, relevant


def compute_recall(queries: dict[str, str], relevant: dict[str, set[str]]) -> dict[int, float]:
    recall_at_k: dict[int, list[float]] = defaultdict(list)
    for qid, qtext in queries.items():
        rel = relevant.get(qid, set())
        if not rel:
            continue
        scores   = bm25.get_scores(qtext.lower().split())
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:MAX_K]
        hits = 0
        for rank, idx in enumerate(top_idxs, start=1):
            if doc_ids[idx] in rel:
                hits += 1
            if rank in K_SET:
                recall_at_k[rank].append(hits / len(rel))
    return {k: sum(v) / len(v) for k, v in recall_at_k.items()}


# ── Compute recall ────────────────────────────────────────────────────────────
print("\nProcessing training split …")
queries, relevant = load_split(QUERIES_FILE, QRELS_FILE)
print(f"  {len(queries):,} queries  |  {sum(len(v) for v in relevant.values()):,} relevant pairs")
results = compute_recall(queries, relevant)

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n  {'K':<6}  {'Recall@K':>10}")
print("  " + "─" * 18)
for k in K_VALUES:
    print(f"  {k:<6}  {results.get(k, float('nan')):>10.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(11, 6))

recalls  = [results.get(k, float("nan")) for k in K_VALUES]
x        = range(len(K_VALUES))
bars     = ax.bar(x, recalls, color="#2196F3", alpha=0.85, width=0.6,
                  label="Training set (left-truncated)")

for bar, val in zip(bars, recalls):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(list(x))
ax.set_xticklabels([str(k) for k in K_VALUES], fontsize=10)
ax.set_xlabel("K (number of retrieved documents)", fontsize=12)
ax.set_ylabel("Mean Recall@K", fontsize=12)
ax.set_title("BM25 Recall@K — BioASQ Left-Truncated Training Set", fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nPlot saved → {OUT_PNG}")
