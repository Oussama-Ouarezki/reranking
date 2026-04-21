"""
BM25 top-500 → monoT5 rerank on BioASQ Task13BGoldenEnriched.

Pipeline:
  1. BM25 top-500 retrieval (baseline)
  2. monoT5 reranks all 500 candidates

Evaluation: P@1, MRR, Recall@5, Recall@10 for both stages.
Output: side-by-side table + appends to evaluation/scores_bioasq_task13b.tsv
"""

import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import ir_measures
from ir_measures import P, RR, Recall, ScoredDoc, Qrel
from rank_bm25 import BM25Okapi
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/bioasq/raw/Task13BGoldenEnriched")
CORPUS_FILE  = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"
QRELS_FILE   = DATA_DIR / "qrels.tsv"
OUT_FILE     = Path("evaluation/scores_bioasq_task13b.tsv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MONOT5_CKPT = Path("checkpoints/monot5-base-msmarco-100k")

BM25_TOP_K  = 1000
BATCH_SIZE  = 16
MAX_LENGTH  = 512

TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"

METRICS = [P @ 1, RR, Recall @ 5, Recall @ 10]
METRIC_LABELS = {
    str(P @ 1):       "P@1",
    str(RR):          "MRR",
    str(Recall @ 5):  "Recall@5",
    str(Recall @ 10): "Recall@10",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("\nLoading corpus …")
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
qrels = []
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
        qrels.append(Qrel(qid, doc_id, int(score)))
print(f"  {len(qrels):,} qrel entries")

# ── BM25 ──────────────────────────────────────────────────────────────────────
print("\nBuilding BM25 index …")
tokenized_corpus = [corpus[d].lower().split() for d in doc_ids]
bm25 = BM25Okapi(tokenized_corpus, k1=0.7, b=0.9)
print("  BM25 index ready.")


def bm25_retrieve(query_text: str, top_k: int = BM25_TOP_K) -> list[tuple[str, float]]:
    scores   = bm25.get_scores(query_text.lower().split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_idxs]


# ── monoT5 ────────────────────────────────────────────────────────────────────
print(f"\nLoading monoT5 from {MONOT5_CKPT} …")
tokenizer = AutoTokenizer.from_pretrained(str(MONOT5_CKPT))
model = T5ForConditionalGeneration.from_pretrained(
    str(MONOT5_CKPT),
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device).eval()

true_id  = tokenizer.convert_tokens_to_ids(TOKEN_TRUE)
false_id = tokenizer.convert_tokens_to_ids(TOKEN_FALSE)
print(f"  true_id={true_id}  false_id={false_id}")


def score_batch(query: str, passages: list[str]) -> list[float]:
    inputs = [f"Query: {query} Document: {p} Relevant:" for p in passages]
    enc = tokenizer(
        inputs, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt"
    ).to(device)
    dec = torch.zeros((len(inputs), 1), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=dec,
        ).logits
    probs = torch.softmax(logits[:, 0, [true_id, false_id]], dim=-1)
    return probs[:, 0].cpu().tolist()


def monot5_rerank(query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    ids    = [d for d, _ in candidates]
    texts  = [corpus[d] for d in ids]
    scores: list[float] = []
    for i in range(0, len(texts), BATCH_SIZE):
        scores.extend(score_batch(query, texts[i : i + BATCH_SIZE]))
    return sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)


# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nRunning BM25 top-{BM25_TOP_K} → monoT5 rerank …")
bm25_run   = []
monot5_run = []

for qid, qtext in tqdm(queries.items(), desc="queries"):
    candidates = bm25_retrieve(qtext)
    for doc_id, score in candidates:
        bm25_run.append(ScoredDoc(qid, doc_id, score=score))

    ranked = monot5_rerank(qtext, candidates)
    for doc_id, score in ranked:
        monot5_run.append(ScoredDoc(qid, doc_id, score=score))

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating …")
bm25_scores   = ir_measures.calc_aggregate(METRICS, qrels, bm25_run)
monot5_scores = ir_measures.calc_aggregate(METRICS, qrels, monot5_run)


def fmt(scores_obj) -> dict[str, float]:
    return {str(m): round(v, 4) for m, v in scores_obj.items()}


bm25_map   = fmt(bm25_scores)
monot5_map = fmt(monot5_scores)

# ── Print table ───────────────────────────────────────────────────────────────
sep = "─" * 52
print(f"\n{sep}")
print(f"  {'Metric':<12} {'BM25@500':>10} {'monoT5@500':>12}")
print(f"  {'──────':<12} {'────────':>10} {'──────────':>12}")
for m_key, label in METRIC_LABELS.items():
    bv = bm25_map.get(m_key, float("nan"))
    mv = monot5_map.get(m_key, float("nan"))
    delta = round(mv - bv, 4)
    sign  = "+" if delta > 0 else ""
    print(f"  {label:<12} {bv:>10.4f} {mv:>12.4f}  ({sign}{delta:.4f})")
print(f"{sep}")
print(f"  Checkpoint : {MONOT5_CKPT}")
print(f"  Top-K      : {BM25_TOP_K}")
print(f"{sep}\n")

# ── Save ──────────────────────────────────────────────────────────────────────
write_header = not OUT_FILE.exists()


def row(name: str, m: dict) -> str:
    vals = [str(m.get(k, "N/A")) for k in METRIC_LABELS]
    return f"{name}\t" + "\t".join(vals) + "\tTask13BGoldenEnriched\n"


with OUT_FILE.open("a") as f:
    if write_header:
        f.write("model\t" + "\t".join(METRIC_LABELS.values()) + "\tdataset\n")
    f.write(row(f"BM25 top-{BM25_TOP_K}", bm25_map))
    f.write(row(f"monoT5 (MS MARCO zero-shot) top-{BM25_TOP_K}", monot5_map))

print(f"Scores saved → {OUT_FILE}")
