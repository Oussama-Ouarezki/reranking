"""
Zero-shot monoT5 (MS MARCO) vs BM25 on BioASQ Task13BGoldenEnriched.

Pipeline:
  1. BM25 top-100 retrieval over Task13BGoldenEnriched corpus
  2. monoT5 pointwise reranking of the same top-100 pool
  3. Evaluate both runs with P@1, MRR (RR), Recall@5, Recall@10

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

CHECKPOINT  = Path("checkpoints/monot5-base-msmarco-100k")
TOP_K       = 100
BATCH_SIZE  = 8
MAX_LENGTH  = 512
TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"

METRICS = [P @ 1, RR, Recall @ 5, Recall @ 10]

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
    next(f)  # skip header
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


def bm25_retrieve(query_text: str, top_k: int = TOP_K) -> list[tuple[str, float]]:
    """Returns list of (doc_id, bm25_score) sorted by score desc."""
    scores   = bm25.get_scores(query_text.lower().split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_idxs]


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


def score_batch(query: str, passages: list[str]) -> list[float]:
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


def rerank(query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Rerank (doc_id, bm25_score) pairs, returns (doc_id, monot5_score) sorted desc."""
    doc_ids_c = [d for d, _ in candidates]
    texts     = [corpus[d] for d in doc_ids_c]
    all_scores: list[float] = []
    for i in range(0, len(texts), BATCH_SIZE):
        all_scores.extend(score_batch(query, texts[i : i + BATCH_SIZE]))
    ranked = sorted(zip(doc_ids_c, all_scores), key=lambda x: x[1], reverse=True)
    return ranked


# ── Run both pipelines ────────────────────────────────────────────────────────
print("\nRunning BM25 + monoT5 reranking …")
bm25_run    = []
monot5_run  = []

for qid, qtext in tqdm(queries.items(), desc="queries"):
    candidates = bm25_retrieve(qtext)

    # BM25 baseline: assign decreasing scores to preserve rank order
    for rank, (doc_id, score) in enumerate(candidates, start=1):
        bm25_run.append(ScoredDoc(qid, doc_id, score=score))

    # monoT5 rerank
    reranked = rerank(qtext, candidates)
    for doc_id, score in reranked:
        monot5_run.append(ScoredDoc(qid, doc_id, score=score))

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating …")
bm25_scores   = ir_measures.calc_aggregate(METRICS, qrels, bm25_run)
monot5_scores = ir_measures.calc_aggregate(METRICS, qrels, monot5_run)

METRIC_LABELS = {
    str(P @ 1):      "P@1",
    str(RR):         "MRR",
    str(Recall @ 5): "Recall@5",
    str(Recall @ 10):"Recall@10",
}

bm25_map   = {str(m): round(v, 4) for m, v in bm25_scores.items()}
monot5_map = {str(m): round(v, 4) for m, v in monot5_scores.items()}

# ── Print comparison table ────────────────────────────────────────────────────
sep = "─" * 52
print(f"\n{sep}")
print(f"  {'Metric':<12} {'BM25':>10} {'monoT5':>10} {'Δ':>10}")
print(f"  {'──────':<12} {'────':>10} {'──────':>10} {'─':>10}")
for m_key, label in METRIC_LABELS.items():
    bv = bm25_map.get(m_key, float("nan"))
    mv = monot5_map.get(m_key, float("nan"))
    delta = round(mv - bv, 4) if isinstance(bv, float) and isinstance(mv, float) else "N/A"
    sign = "+" if isinstance(delta, float) and delta > 0 else ""
    print(f"  {label:<12} {bv:>10.4f} {mv:>10.4f} {sign}{delta:>9.4f}")
print(f"{sep}")
print(f"  Checkpoint : {CHECKPOINT}")
print(f"  Top-K      : {TOP_K}")
print(f"{sep}\n")

# ── Save ──────────────────────────────────────────────────────────────────────
metric_keys  = list(METRIC_LABELS.values())
write_header = not OUT_FILE.exists()
with OUT_FILE.open("a") as f:
    if write_header:
        f.write("model\t" + "\t".join(metric_keys) + "\tdataset\n")
    bm25_row   = "\t".join(str(bm25_map.get(k, ""))   for k in METRIC_LABELS)
    monot5_row = "\t".join(str(monot5_map.get(k, "")) for k in METRIC_LABELS)
    f.write(f"BM25\t{bm25_row}\tTask13BGoldenEnriched\n")
    f.write(f"monoT5 (MS MARCO zero-shot)\t{monot5_row}\tTask13BGoldenEnriched\n")

print(f"Scores saved → {OUT_FILE}")
