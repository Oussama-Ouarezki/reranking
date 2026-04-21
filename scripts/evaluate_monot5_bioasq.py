"""
Evaluate monoT5 (MS MARCO pretrained) on BioASQ Task13BGoldenEnriched using nDCG@10.

Pipeline:
  1. BM25 top-20 retrieval over Task13BGoldenEnriched corpus
  2. monoT5 pointwise reranking: scores each (query, passage) pair via P("true")
  3. nDCG@10 via ir-measures

Output: appends to evaluation/scores_bioasq_task13b.tsv
"""

import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import ir_measures
from ir_measures import nDCG, ScoredDoc, Qrel
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

CHECKPOINT  = Path("checkpoints/monot5-base-msmarco-100k")
TOP_K       = 20
BATCH_SIZE  = 8
MAX_LENGTH  = 512
TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load corpus, queries, qrels ───────────────────────────────────────────────
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

# ── BM25 index ────────────────────────────────────────────────────────────────
print("\nBuilding BM25 index …")
tokenized_corpus = [corpus[d].lower().split() for d in doc_ids]
bm25 = BM25Okapi(tokenized_corpus, k1=0.9, b=0.4)
print("  BM25 index ready.")


def bm25_retrieve(query_text: str, top_k: int = TOP_K) -> list:
    scores   = bm25.get_scores(query_text.lower().split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], corpus[doc_ids[i]]) for i in top_idxs]


# ── monoT5 reranker ───────────────────────────────────────────────────────────
print(f"\nLoading monoT5 from {CHECKPOINT} …")
tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
model = T5ForConditionalGeneration.from_pretrained(
    str(CHECKPOINT),
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device).eval()

true_id  = tokenizer.convert_tokens_to_ids(TOKEN_TRUE)
false_id = tokenizer.convert_tokens_to_ids(TOKEN_FALSE)
print(f"  true_id={true_id}  false_id={false_id}")


def score_batch(query: str, passages: list) -> list:
    inputs = [f"Query: {query} Document: {p} Relevant:" for p in passages]
    enc = tokenizer(
        inputs, padding=True, truncation=True,
        max_length=MAX_LENGTH, return_tensors="pt"
    ).to(device)
    decoder_input = torch.zeros((len(inputs), 1), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input,
        ).logits  # (batch, 1, vocab)
    probs = torch.softmax(logits[:, 0, [true_id, false_id]], dim=-1)
    return probs[:, 0].cpu().tolist()


def rerank(query: str, candidates: list) -> list:
    all_scores = []
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i : i + BATCH_SIZE]
        all_scores.extend(score_batch(query, [p for _, p in batch]))
    ranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for (doc_id, _), score in ranked]


# ── Run evaluation ────────────────────────────────────────────────────────────
print("\nReranking …")
run = []
for qid, qtext in tqdm(queries.items(), desc="monoT5"):
    candidates = bm25_retrieve(qtext)
    ranked     = rerank(qtext, candidates)
    for doc_id, score in ranked:
        run.append(ScoredDoc(qid, doc_id, score=score))

score = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)[nDCG @ 10]
print(f"\nmonoT5 (MS MARCO)  nDCG@10 = {score:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
write_header = not OUT_FILE.exists()
with OUT_FILE.open("a") as f:
    if write_header:
        f.write("model\tnDCG@10\tdataset\n")
    f.write(f"monoT5 (MS MARCO)\t{score:.4f}\tTask13BGoldenEnriched\n")

print(f"Saved → {OUT_FILE}")
