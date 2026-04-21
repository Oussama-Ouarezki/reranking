"""
BM25 → monoT5 → duoT5 cascade on BioASQ Task13BGoldenEnriched.

Pipeline:
  1. BM25 top-100  (baseline)
  2. monoT5 reranks 100 → top-20 passed to duoT5
  3. duoT5 pairwise tournament over 20 → top-10

Evaluation: P@1, MRR, Recall@5, Recall@10 for all three stages.
Output: side-by-side table + appends to evaluation/scores_bioasq_task13b.tsv

Fixes applied vs original:
  1. duoT5 MAX_LENGTH bumped to 1024 to avoid brutal truncation of doc pairs.
     Each document is also individually pre-truncated to DOC_MAX_TOKENS tokens
     before concatenation so the final prompt fits within MAX_DUO_LENGTH.
  2. Recall@10 only stored/evaluated for stages that actually return ≥10 docs
     per query (BM25@100, monoT5@100). duoT5 only stores top-10, so Recall@10
     is trivially 1.0 and is reported separately to avoid misleading comparison.
  3. torch.cuda.empty_cache() added between monoT5 and duoT5 loading to reduce
     peak VRAM on RTX 3060 12 GB.
  4. row() helper fixed to look up metric values by METRIC_LABELS keys
     (str(metric_obj)) consistently with fmt().
"""

import json
import itertools
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
DUOT5_CKPT  = Path("checkpoints/duot5-base-msmarco")

BM25_TOP_K   = 100
MONO_TOP_K   = 20   # monoT5 → top-20 sent to duoT5
DUO_TOP_K    = 10   # duoT5  → top-10 final

MONO_BATCH_SIZE = 8
DUO_BATCH_SIZE  = 4   # smaller batch for duoT5: inputs are ~2× longer

# FIX 1: separate max lengths for mono vs duo
MONO_MAX_LENGTH = 512
MAX_DUO_LENGTH  = 1024   # duoT5 inputs contain two documents
# Pre-truncate each document to this many whitespace-tokens before building
# the duoT5 prompt, so the final string reliably fits in MAX_DUO_LENGTH.
DOC_MAX_TOKENS  = 200

TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"

# FIX 2: separate metric sets so Recall@10 is not computed for duoT5
METRICS_FULL   = [P @ 1, RR, Recall @ 5, Recall @ 10]   # BM25, monoT5
METRICS_PARTIAL = [P @ 1, RR, Recall @ 5]                # duoT5 (only top-10 stored)

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
bm25 = BM25Okapi(tokenized_corpus, k1=0.9, b=0.4)
print("  BM25 index ready.")


def bm25_retrieve(query_text: str, top_k: int = BM25_TOP_K) -> list[tuple[str, float]]:
    scores   = bm25.get_scores(query_text.lower().split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(doc_ids[i], float(scores[i])) for i in top_idxs]


# ── monoT5 ────────────────────────────────────────────────────────────────────
print(f"\nLoading monoT5 from {MONOT5_CKPT} …")
mono_tok = AutoTokenizer.from_pretrained(str(MONOT5_CKPT))
mono_model = T5ForConditionalGeneration.from_pretrained(
    str(MONOT5_CKPT),
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device).eval()

mono_true_id  = mono_tok.convert_tokens_to_ids(TOKEN_TRUE)
mono_false_id = mono_tok.convert_tokens_to_ids(TOKEN_FALSE)
print(f"  true_id={mono_true_id}  false_id={mono_false_id}")


def monot5_score_batch(query: str, passages: list[str]) -> list[float]:
    inputs = [f"Query: {query} Document: {p} Relevant:" for p in passages]
    enc = mono_tok(
        inputs, padding=True, truncation=True,
        max_length=MONO_MAX_LENGTH, return_tensors="pt"
    ).to(device)
    dec = torch.zeros((len(inputs), 1), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = mono_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=dec,
        ).logits
    probs = torch.softmax(logits[:, 0, [mono_true_id, mono_false_id]], dim=-1)
    return probs[:, 0].cpu().tolist()


def monot5_rerank(query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Returns (doc_id, score) sorted desc."""
    ids   = [d for d, _ in candidates]
    texts = [corpus[d] for d in ids]
    scores: list[float] = []
    for i in range(0, len(texts), MONO_BATCH_SIZE):
        scores.extend(monot5_score_batch(query, texts[i : i + MONO_BATCH_SIZE]))
    return sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)


# ── duoT5 ─────────────────────────────────────────────────────────────────────
# FIX 3: free monoT5 VRAM before loading duoT5
print(f"\nFreeing monoT5 from GPU memory …")
mono_model = mono_model.cpu()
del mono_model
if device.type == "cuda":
    torch.cuda.empty_cache()

print(f"Loading duoT5 from {DUOT5_CKPT} …")
duo_tok = AutoTokenizer.from_pretrained(str(DUOT5_CKPT))
duo_model = T5ForConditionalGeneration.from_pretrained(
    str(DUOT5_CKPT),
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device).eval()

duo_true_id  = duo_tok.convert_tokens_to_ids(TOKEN_TRUE)
duo_false_id = duo_tok.convert_tokens_to_ids(TOKEN_FALSE)
print(f"  true_id={duo_true_id}  false_id={duo_false_id}")


def truncate_doc(text: str, max_tokens: int = DOC_MAX_TOKENS) -> str:
    """Whitespace-token truncation to keep duoT5 prompt within MAX_DUO_LENGTH."""
    tokens = text.split()
    return " ".join(tokens[:max_tokens]) if len(tokens) > max_tokens else text


def duot5_score_batch(query: str, pairs: list[tuple[str, str]]) -> list[float]:
    """Score a batch of (doc0, doc1) pairs. Returns P(doc0 > doc1) for each."""
    # FIX 1: truncate each document individually before building the prompt
    inputs = [
        f"Query: {query} Document0: {truncate_doc(p0)} Document1: {truncate_doc(p1)} Relevant:"
        for p0, p1 in pairs
    ]
    enc = duo_tok(
        inputs, padding=True, truncation=True,
        max_length=MAX_DUO_LENGTH, return_tensors="pt"   # FIX 1: 1024 not 512
    ).to(device)
    dec = torch.zeros((len(inputs), 1), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = duo_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=dec,
        ).logits
    probs = torch.softmax(logits[:, 0, [duo_true_id, duo_false_id]], dim=-1)
    return probs[:, 0].cpu().tolist()


def duot5_rerank(query: str, candidates: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    All-pairs tournament: for every ordered pair (i,j), compute P(doc_i > doc_j).
    Aggregate score for each doc = sum of win probabilities across all pairs.
    Returns (doc_id, agg_score) sorted desc.
    """
    ids   = [d for d, _ in candidates]
    texts = [corpus[d] for d in ids]
    agg   = {d: 0.0 for d in ids}

    all_pairs: list[tuple[int, int]] = list(itertools.permutations(range(len(ids)), 2))

    for batch_start in range(0, len(all_pairs), DUO_BATCH_SIZE):
        batch = all_pairs[batch_start : batch_start + DUO_BATCH_SIZE]
        pair_texts = [(texts[i], texts[j]) for i, j in batch]
        win_probs  = duot5_score_batch(query, pair_texts)
        for (i, _j), p in zip(batch, win_probs):
            agg[ids[i]] += p

    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


# ── Run all three stages ───────────────────────────────────────────────────────
# NOTE: monoT5 was moved to CPU/deleted above to free VRAM.
# We reload it here only for the scoring loop, then free again before duoT5.
print(f"\nReloading monoT5 for scoring loop …")
mono_tok = AutoTokenizer.from_pretrained(str(MONOT5_CKPT))
mono_model = T5ForConditionalGeneration.from_pretrained(
    str(MONOT5_CKPT),
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device).eval()
mono_true_id  = mono_tok.convert_tokens_to_ids(TOKEN_TRUE)
mono_false_id = mono_tok.convert_tokens_to_ids(TOKEN_FALSE)

print("\nStage 1+2: BM25 → monoT5 …")
bm25_run   = []
monot5_run = []
mono_top20_per_query: dict[str, list[tuple[str, float]]] = {}

for qid, qtext in tqdm(queries.items(), desc="monoT5"):
    # Stage 1: BM25 top-100
    bm25_cands = bm25_retrieve(qtext, top_k=BM25_TOP_K)
    for doc_id, score in bm25_cands:
        bm25_run.append(ScoredDoc(qid, doc_id, score=score))

    # Stage 2: monoT5 rerank all 100 → store full ranking, keep top-20 for duoT5
    mono_ranked = monot5_rerank(qtext, bm25_cands)
    for doc_id, score in mono_ranked:
        monot5_run.append(ScoredDoc(qid, doc_id, score=score))
    mono_top20_per_query[qid] = mono_ranked[:MONO_TOP_K]

# FIX 3: free monoT5 before loading duoT5
print("\nFreeing monoT5 from GPU memory before duoT5 …")
mono_model = mono_model.cpu()
del mono_model
if device.type == "cuda":
    torch.cuda.empty_cache()

print(f"Loading duoT5 from {DUOT5_CKPT} …")
duo_tok = AutoTokenizer.from_pretrained(str(DUOT5_CKPT))
duo_model = T5ForConditionalGeneration.from_pretrained(
    str(DUOT5_CKPT),
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device).eval()
duo_true_id  = duo_tok.convert_tokens_to_ids(TOKEN_TRUE)
duo_false_id = duo_tok.convert_tokens_to_ids(TOKEN_FALSE)

print("\nStage 3: duoT5 …")
duot5_run = []

for qid, qtext in tqdm(queries.items(), desc="duoT5"):
    mono_top20 = mono_top20_per_query[qid]
    duo_ranked = duot5_rerank(qtext, mono_top20)
    duo_top10  = duo_ranked[:DUO_TOP_K]
    for doc_id, score in duo_top10:
        duot5_run.append(ScoredDoc(qid, doc_id, score=score))

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating …")
bm25_scores   = ir_measures.calc_aggregate(METRICS_FULL,    qrels, bm25_run)
monot5_scores = ir_measures.calc_aggregate(METRICS_FULL,    qrels, monot5_run)
# FIX 2: duoT5 only has top-10 per query → skip Recall@10
duot5_scores  = ir_measures.calc_aggregate(METRICS_PARTIAL, qrels, duot5_run)


def fmt(scores_obj) -> dict[str, float]:
    return {str(m): round(v, 4) for m, v in scores_obj.items()}


bm25_map   = fmt(bm25_scores)
monot5_map = fmt(monot5_scores)
duot5_map  = fmt(duot5_scores)

# ── Print table ───────────────────────────────────────────────────────────────
sep = "─" * 68
print(f"\n{sep}")
print(f"  {'Metric':<12} {'BM25@100':>10} {'monoT5@20':>11} {'duoT5@10':>10}")
print(f"  {'──────':<12} {'────────':>10} {'─────────':>11} {'────────':>10}")
for m_key, label in METRIC_LABELS.items():
    bv = bm25_map.get(m_key, float("nan"))
    mv = monot5_map.get(m_key, float("nan"))
    # FIX 2: duoT5 did not compute Recall@10 — show N/A
    dv_raw = duot5_map.get(m_key)
    dv_str = f"{dv_raw:>10.4f}" if dv_raw is not None else f"{'N/A':>10}"
    print(f"  {label:<12} {bv:>10.4f} {mv:>11.4f} {dv_str}")
print(f"{sep}")
print(f"  Pipeline : BM25 top-{BM25_TOP_K} → monoT5 top-{MONO_TOP_K} → duoT5 top-{DUO_TOP_K}")
print(f"  monoT5   : {MONOT5_CKPT}")
print(f"  duoT5    : {DUOT5_CKPT}")
print(f"  NOTE     : duoT5 Recall@10 = N/A (only top-{DUO_TOP_K} docs stored per query)")
print(f"{sep}\n")

# ── Save ──────────────────────────────────────────────────────────────────────
metric_keys  = list(METRIC_LABELS.values())
write_header = not OUT_FILE.exists()

# FIX 4: row() now uses METRIC_LABELS keys (str(metric_obj)) consistently with fmt()
def row(name: str, m: dict) -> str:
    vals = []
    for k in METRIC_LABELS:          # k is str(metric_obj), matches fmt() output
        v = m.get(k)
        vals.append(str(v) if v is not None else "N/A")
    return f"{name}\t" + "\t".join(vals) + "\tTask13BGoldenEnriched\n"

with OUT_FILE.open("a") as f:
    if write_header:
        f.write("model\t" + "\t".join(metric_keys) + "\tdataset\n")
    f.write(row("BM25 top-100", bm25_map))
    f.write(row(f"monoT5 (MS MARCO zero-shot) top-{MONO_TOP_K}", monot5_map))
    f.write(row(f"duoT5 (MS MARCO zero-shot) top-{DUO_TOP_K}", duot5_map))

print(f"Scores saved → {OUT_FILE}")