"""
Plot BM25 Recall@K curve on BioASQ Task13BGoldenEnriched test batches (13B1–13B4).
One curve per batch + one combined curve.
Saves: evaluation/bm25_recall_at_k_test.png
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
BASE_DIR    = Path("data/bioasq/raw/Task13BGoldenEnriched")
CORPUS_FILE = BASE_DIR / "corpus.jsonl"
BATCHES     = ["13B1", "13B2", "13B3", "13B4"]
OUT_PNG     = Path("evaluation/bm25_recall_at_k_test.png")
OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
MAX_K    = max(K_VALUES)

BATCH_COLORS = {
    "13B1":    "#2196F3",
    "13B2":    "#E91E63",
    "13B3":    "#4CAF50",
    "13B4":    "#FF9800",
    "Combined": "#673AB7",
}

# ── Load corpus ───────────────────────────────────────────────────────────────
print("Loading corpus …")
corpus, doc_ids = {}, []
with CORPUS_FILE.open() as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = (doc.get("title", "") + " " + doc["text"]).strip()
        doc_ids.append(doc["_id"])
print(f"  {len(corpus):,} documents")

# ── BM25 index (built once, shared across all batches) ────────────────────────
print("\nBuilding BM25 index …")
tokenized_corpus = [corpus[d].lower().split() for d in doc_ids]
bm25 = BM25Okapi(tokenized_corpus, k1=0.9, b=0.4)
print("  BM25 index ready.")

K_SET = set(K_VALUES)


def load_batch(batch: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    batch_dir = BASE_DIR / batch
    queries = {}
    with (batch_dir / "queries.jsonl").open() as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]

    relevant: dict[str, set[str]] = defaultdict(set)
    with (batch_dir / "qrels.tsv").open() as f:
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


# ── Compute per-batch recall ───────────────────────────────────────────────────
batch_results: dict[str, dict[int, float]] = {}
all_queries: dict[str, str] = {}
all_relevant: dict[str, set[str]] = {}

for batch in BATCHES:
    print(f"\nProcessing {batch} …")
    queries, relevant = load_batch(batch)
    print(f"  {len(queries):,} queries  |  {sum(len(v) for v in relevant.values()):,} relevant pairs")
    batch_results[batch] = compute_recall(queries, relevant)
    all_queries.update(queries)
    for qid, docs in relevant.items():
        all_relevant[qid].update(docs)

print("\nProcessing combined …")
batch_results["Combined"] = compute_recall(all_queries, all_relevant)

# ── Print table ───────────────────────────────────────────────────────────────
header = f"  {'K':<6}" + "".join(f"  {b:>10}" for b in BATCHES + ['Combined'])
print(f"\n{header}")
print("  " + "─" * (6 + 12 * len(BATCHES + ['Combined'])))
for k in K_VALUES:
    row = f"  {k:<6}"
    for b in BATCHES + ["Combined"]:
        row += f"  {batch_results[b].get(k, float('nan')):>10.4f}"
    print(row)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

for label in BATCHES + ["Combined"]:
    recalls = [batch_results[label].get(k, float("nan")) for k in K_VALUES]
    lw = 2.5 if label == "Combined" else 1.5
    ls = "-" if label == "Combined" else "--"
    ax.plot(K_VALUES, recalls, marker="o", linewidth=lw, linestyle=ls,
            markersize=5, color=BATCH_COLORS[label], label=label)

ax.set_xscale("log")
ax.set_xticks(K_VALUES)
ax.set_xticklabels([str(k) for k in K_VALUES])
ax.set_xlabel("K (number of retrieved documents)", fontsize=12)
ax.set_ylabel("Mean Recall@K", fontsize=12)
ax.set_title("BM25 Recall@K — BioASQ Task13BGoldenEnriched Test Batches", fontsize=13)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"\nPlot saved → {OUT_PNG}")
