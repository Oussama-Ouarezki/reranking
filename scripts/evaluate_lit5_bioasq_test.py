"""
Zero-shot LiT5-Distill (MS MARCO) evaluation on BioASQ Task13BGoldenEnriched test set.

Pipeline:
  1. BM25 top-100 retrieval over Task13BGoldenEnriched corpus
  2. LiT5-Distill sliding window reranking (window=20, stride=10)

Metrics at K = 5, 10, 20:
  nDCG@K, MRR, Recall@K, Precision@K

Output: evaluation/scores_bioasq_task13b.tsv

Usage:
    python scripts/evaluate_lit5_bioasq_test.py
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch
import ir_measures
from ir_measures import nDCG, RR, Recall, P, ScoredDoc, Qrel
from rank_bm25 import BM25Okapi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/bioasq/raw/Task13BGoldenEnriched")
CORPUS_FILE = DATA_DIR / "corpus.jsonl"
BATCHES     = ["13B1", "13B2", "13B3", "13B4"]

LIT5_CKPT   = "castorini/LiT5-Distill-base"
OUT_FILE    = Path("evaluation/scores_bioasq_task13b.tsv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

BM25_TOP_K     = 100
WINDOW_SIZE    = 20
STRIDE         = 10
TEXT_MAXLENGTH = 150
MAX_NEW_TOKENS = 140

QUERY_PREFIX = "Search Query:"
PASS_PREFIX  = "Passage:"
SUFFIX       = " Relevance Ranking:"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── tokeniser ─────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


# ── data loaders ──────────────────────────────────────────────────────────────
def load_corpus():
    corpus, doc_ids = {}, []
    with CORPUS_FILE.open() as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["_id"]] = (doc.get("title", "") + " " + doc["text"]).strip()
            doc_ids.append(doc["_id"])
    return corpus, doc_ids


def load_test_data():
    queries = {}
    qrels   = []
    for batch in BATCHES:
        batch_dir = DATA_DIR / batch
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
    return queries, qrels


# ── BM25 ──────────────────────────────────────────────────────────────────────
def build_bm25(corpus, doc_ids):
    tokenized = [tokenize(corpus[d]) for d in doc_ids]
    return BM25Okapi(tokenized, k1=0.9, b=0.4)


def bm25_retrieve(bm25, corpus, doc_ids, query_text, top_k=BM25_TOP_K):
    scores   = bm25.get_scores(tokenize(query_text))
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], corpus[doc_ids[i]]) for i in top_idxs]


# ── LiT5 sliding window reranking ─────────────────────────────────────────────
def parse_ranking(perm_text, n):
    nums, seen = [], set()
    for tok in perm_text.replace(",", " ").replace(">", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and (v - 1) not in seen:
                nums.append(v - 1)
                seen.add(v - 1)
        except ValueError:
            continue
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums


def rerank_window(model, tokenizer, query, window):
    """window: list of (doc_id, text). Returns reordered list."""
    strings = [
        f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i+1}] {text}{SUFFIX}"
        for i, (_, text) in enumerate(window)
    ]
    enc = tokenizer(
        strings,
        max_length=TEXT_MAXLENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    perm_text = tokenizer.decode(out[0], skip_special_tokens=True)
    perm      = parse_ranking(perm_text, len(window))
    return [window[i] for i in perm]


def sliding_window_rerank(model, tokenizer, query, candidates):
    """candidates: list of (doc_id, text), length = BM25_TOP_K."""
    ranked = list(candidates)
    start  = max(0, len(ranked) - WINDOW_SIZE)
    while True:
        end          = min(start + WINDOW_SIZE, len(ranked))
        ranked[start:end] = rerank_window(model, tokenizer, query, ranked[start:end])
        if start == 0:
            break
        start = max(0, start - STRIDE)
    return ranked


# ── main ──────────────────────────────────────────────────────────────────────
print("\nLoading corpus …")
corpus, doc_ids = load_corpus()
print(f"  {len(corpus):,} documents")

print("Loading test queries and qrels (13B1–13B4) …")
queries, qrels = load_test_data()
print(f"  {len(queries):,} queries  |  {len(qrels):,} qrel entries")

print("\nBuilding BM25 index …")
bm25 = build_bm25(corpus, doc_ids)
print("  BM25 index ready.")

# BM25 baseline run
print("\nRunning BM25 baseline …")
bm25_run = []
bm25_candidates = {}
for qid, qtext in tqdm(queries.items(), desc="BM25"):
    cands = bm25_retrieve(bm25, corpus, doc_ids, qtext)
    bm25_candidates[qid] = cands
    for rank, (did, _) in enumerate(cands, start=1):
        bm25_run.append(ScoredDoc(qid, did, score=BM25_TOP_K - rank + 1))

# LiT5 reranking
print(f"\nLoading LiT5-Distill from {LIT5_CKPT} …")
tokenizer = T5Tokenizer.from_pretrained(LIT5_CKPT, legacy=False, use_fast=True)
model     = T5ForConditionalGeneration.from_pretrained(
    LIT5_CKPT,
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device).eval()

print("\nReranking with LiT5 …")
lit5_run = []
for qid, qtext in tqdm(queries.items(), desc="LiT5"):
    reranked = sliding_window_rerank(model, tokenizer, qtext, bm25_candidates[qid])
    for rank, (did, _) in enumerate(reranked, start=1):
        lit5_run.append(ScoredDoc(qid, did, score=BM25_TOP_K - rank + 1))

# Evaluate
print("\nEvaluating …")
bm25_scores = ir_measures.calc_aggregate(METRICS, qrels, bm25_run)
lit5_scores = ir_measures.calc_aggregate(METRICS, qrels, lit5_run)


def fmt(s):
    return {str(m): round(v, 4) for m, v in s.items()}


bm25_map = fmt(bm25_scores)
lit5_map = fmt(lit5_scores)

# Print table
sep = "─" * 60
print(f"\n{sep}")
print(f"  {'Metric':<14} {'BM25@100':>10} {'LiT5 (zero-shot)':>18}")
print(f"  {'──────':<14} {'────────':>10} {'────────────────':>18}")
for m_key, label in METRIC_LABELS.items():
    bv = bm25_map.get(m_key, float("nan"))
    lv = lit5_map.get(m_key, float("nan"))
    delta = round(lv - bv, 4)
    sign  = "+" if delta > 0 else ""
    print(f"  {label:<14} {bv:>10.4f} {lv:>18.4f}  ({sign}{delta:.4f})")
print(f"{sep}")
print(f"  Checkpoint : {LIT5_CKPT}")
print(f"  BM25 Top-K : {BM25_TOP_K}  |  Window: {WINDOW_SIZE}  Stride: {STRIDE}")
print(f"{sep}\n")

# Save
write_header = not OUT_FILE.exists()
with OUT_FILE.open("a") as f:
    if write_header:
        f.write("model\t" + "\t".join(METRIC_LABELS.values()) + "\tdataset\n")
    def row(name, m):
        vals = "\t".join(str(m.get(k, "N/A")) for k in METRIC_LABELS)
        return f"{name}\t{vals}\tTask13BGoldenEnriched\n"
    f.write(row("BM25 top-100", bm25_map))
    f.write(row("LiT5-Distill (MS MARCO zero-shot)", lit5_map))

print(f"Scores saved → {OUT_FILE}")
