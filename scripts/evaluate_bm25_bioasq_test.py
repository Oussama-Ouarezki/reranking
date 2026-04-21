"""
BM25 evaluation on BioASQ Task13BGoldenEnriched test set.

Metrics at K = 5, 10, 20: nDCG@K, MRR, Recall@K, P@K

Usage:
    python scripts/evaluate_bm25_bioasq_test.py
"""

import json
import re
from pathlib import Path

import numpy as np
import ir_measures
from ir_measures import nDCG, RR, Recall, P, ScoredDoc, Qrel
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
TEST_DIR         = Path("data/bioasq/raw/Task13BGoldenEnriched")
TRAIN_CORPUS     = Path("data/bioasq/processed/corpus.jsonl")
TEST_CORPUS      = TEST_DIR / "corpus.jsonl"
BATCHES          = ["13B1", "13B2", "13B3", "13B4"]
OUT_FILE         = Path("evaluation/scores_bioasq_task13b.tsv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

BM25_TOP_K = 100

METRICS = [nDCG @ 5, nDCG @ 10, nDCG @ 20,
           RR,
           Recall @ 5, Recall @ 10, Recall @ 20,
           P @ 5, P @ 10, P @ 20]

METRIC_LABELS = {
    str(nDCG @ 5):    "nDCG@5",
    str(nDCG @ 10):   "nDCG@10",
    str(nDCG @ 20):   "nDCG@20",
    str(RR):          "MRR",
    str(Recall @ 5):  "Recall@5",
    str(Recall @ 10): "Recall@10",
    str(Recall @ 20): "Recall@20",
    str(P @ 5):       "P@5",
    str(P @ 10):      "P@10",
    str(P @ 20):      "P@20",
}


def tokenize(text):
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading corpus (training + test combined) …")
corpus, doc_ids = {}, []
for corpus_file in [TRAIN_CORPUS, TEST_CORPUS]:
    with corpus_file.open() as f:
        for line in f:
            doc = json.loads(line)
            if doc["_id"] not in corpus:
                corpus[doc["_id"]] = (doc.get("title", "") + " " + doc["text"]).strip()
                doc_ids.append(doc["_id"])
print(f"  {len(corpus):,} documents")

print("Loading test queries and qrels (13B1–13B4) …")
queries = {}
qrels   = []
for batch in BATCHES:
    batch_dir = TEST_DIR / batch
    with (batch_dir / "queries.jsonl").open() as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    with (batch_dir / "qrels.tsv").open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            qrels.append(Qrel(qid, did, int(score)))
print(f"  {len(queries):,} queries  |  {len(qrels):,} qrel entries")

# ── BM25 ──────────────────────────────────────────────────────────────────────
print("\nBuilding BM25 index …")
tokenized_corpus = [tokenize(corpus[d]) for d in doc_ids]
bm25 = BM25Okapi(tokenized_corpus, k1=0.9, b=0.4)
print("  BM25 index ready.")

print(f"\nRetrieving top-{BM25_TOP_K} for {len(queries):,} queries …")
run = []
for qid, qtext in tqdm(queries.items(), desc="BM25"):
    scores   = bm25.get_scores(tokenize(qtext))
    top_idxs = np.argsort(scores)[::-1][:BM25_TOP_K]
    for rank, idx in enumerate(top_idxs, start=1):
        run.append(ScoredDoc(qid, doc_ids[idx], score=float(scores[idx])))

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating …")
results = ir_measures.calc_aggregate(METRICS, qrels, run)
scores  = {str(m): round(v, 4) for m, v in results.items()}

# ── Print table ───────────────────────────────────────────────────────────────
sep = "─" * 40
print(f"\n{sep}")
print(f"  {'Metric':<14} {'Score':>8}")
print(f"  {'──────':<14} {'─────':>8}")
for m_key, label in METRIC_LABELS.items():
    print(f"  {label:<14} {scores.get(m_key, float('nan')):>8.4f}")
print(f"{sep}")
print(f"  k1=0.9  b=0.4  Top-K={BM25_TOP_K}")
print(f"  Test set: 13B1–13B4  ({len(queries)} queries)")
print(f"{sep}\n")

# ── Save ──────────────────────────────────────────────────────────────────────
write_header = not OUT_FILE.exists()
with OUT_FILE.open("a") as f:
    if write_header:
        f.write("model\t" + "\t".join(METRIC_LABELS.values()) + "\tdataset\n")
    vals = "\t".join(str(scores.get(k, "N/A")) for k in METRIC_LABELS)
    f.write(f"BM25 top-{BM25_TOP_K}\t{vals}\tTask13BGoldenEnriched\n")

print(f"Scores saved → {OUT_FILE}")
