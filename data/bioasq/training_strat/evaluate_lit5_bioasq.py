"""
Evaluate LiT5 models on Task13BGoldenEnriched using nDCG@10.

Pipeline per query:
  1. BM25 top-20 retrieval over Task13BGoldenEnriched corpus
  2. LiT5 reranking (FiD-style generation → permutation)
  3. nDCG@10 via ir-measures

Models evaluated:
  - BM25 only (baseline)
  - LiT5-Distill-base (MS MARCO pretrained, no fine-tuning)
  - checkpoint1 (fine-tuned on dataset1)
  - checkpoint2 (fine-tuned on dataset2)

Fixes applied:
  - TEXT_MAXLENGTH = 512 (query + passage budget)
  - Correct FiD input shape: (1, n_passages, seq_len) via unsqueeze(0)
  - Proper score assignment using 1/(rank+1) instead of TOP_K - rank
  - Dynamic query-aware truncation reserving tokens for query + overhead
  - OOM guard: falls back to TOP_K=10 if 20 passages OOM

Output: evaluation/scores_bioasq_task13b.tsv
"""

import json
import gc
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import ir_measures
from ir_measures import nDCG, ScoredDoc, Qrel
from rank_bm25 import BM25Okapi
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data/bioasq/raw/Task13BGoldenEnriched")
CORPUS_FILE  = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"
QRELS_FILE   = DATA_DIR / "qrels.tsv"
OUT_FILE     = Path("evaluation/scores_bioasq_task13b.tsv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

MODELS = {
    "LiT5-base (MS MARCO)":    Path("checkpoints/LiT5-Distill-base"),
    "checkpoint1 (dataset1)":  Path("data/bioasq/training_strat/checkpoint1"),
    "checkpoint2 (dataset2)":  Path("data/bioasq/training_strat/checkpoint2"),
}

TOP_K          = 10       # BM25 candidates; 10×512=5120 tokens fits in 12GB GPU; falls back to 5 on OOM
TEXT_MAXLENGTH = 512      # total tokens per (query + passage) input
MAX_NEW_TOKENS = 100      # enough to generate a full permutation of 20 items
QUERY_PREFIX   = "Search Query:"
PASSAGE_PREFIX = "Passage:"
SUFFIX         = " Relevance Ranking:"
OVERHEAD_TOK   = 20       # approximate tokens for prefixes, brackets, suffix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


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
    next(f)  # skip header
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            qid, doc_id, score = parts
        elif len(parts) == 4:
            # TREC format: qid 0 doc_id score
            qid, _, doc_id, score = parts
        else:
            continue
        qrels.append(Qrel(qid, doc_id, int(score)))
print(f"  {len(qrels):,} qrel entries")

# Sanity check: verify qrel doc_ids exist in corpus
qrel_doc_ids  = {q.doc_id for q in qrels}
corpus_doc_ids = set(corpus.keys())
overlap = qrel_doc_ids & corpus_doc_ids
print(f"  Qrel doc_id overlap with corpus: {len(overlap)}/{len(qrel_doc_ids)}")
if len(overlap) == 0:
    raise ValueError(
        "NO qrel doc_ids found in corpus! "
        "Check your doc_id format — qrels and corpus must use identical IDs."
    )


# ── Build BM25 index ──────────────────────────────────────────────────────────
print("\nBuilding BM25 index …")
tokenized_corpus = [corpus[d].lower().split() for d in doc_ids]
bm25 = BM25Okapi(tokenized_corpus, k1=0.9, b=0.4)
print("  BM25 index ready.")


# ── BM25 retrieval ────────────────────────────────────────────────────────────
def bm25_retrieve(query_text: str, top_k: int = TOP_K) -> list:
    scores   = bm25.get_scores(query_text.lower().split())
    top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "doc_id": doc_ids[i],
            "score":  float(scores[i]),
            "text":   corpus[doc_ids[i]],
        }
        for i in top_idxs
    ]


# ── LiT5 helpers ─────────────────────────────────────────────────────────────
def truncate_passage(query: str, passage: str) -> str:
    """
    Truncate passage so that query + passage + overhead fits in TEXT_MAXLENGTH.
    All token counts are approximated by whitespace splitting (fast, good enough).
    """
    query_tok_count = len(query.split())
    budget          = TEXT_MAXLENGTH - query_tok_count - OVERHEAD_TOK
    budget          = max(budget, 10)   # always keep at least 10 passage tokens
    return " ".join(passage.split()[:budget])


def build_fid_input(query: str, passages: list, tokenizer) -> tuple:
    """
    Build FiD-style input.
    Each passage becomes one string: QUERY_PREFIX query PASSAGE_PREFIX [i+1] passage SUFFIX
    Tokenizer encodes each string to TEXT_MAXLENGTH tokens.
    Final shape: input_ids  (1, n_passages * TEXT_MAXLENGTH)
                 attn_mask  (1, n_passages * TEXT_MAXLENGTH)
    """
    strings = [
        f"{QUERY_PREFIX} {query} {PASSAGE_PREFIX} [{i+1}] "
        f"{truncate_passage(query, p['text'])}{SUFFIX}"
        for i, p in enumerate(passages)
    ]
    enc = tokenizer(
        strings,
        max_length=TEXT_MAXLENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # FiD: flatten all passages into a single sequence (1, n_passages * seq_len)
    input_ids = enc.input_ids.view(1, -1).to(device)
    attn_mask = enc.attention_mask.view(1, -1).to(device)
    return input_ids, attn_mask


def parse_permutation(text: str, n: int) -> list:
    """
    Parse a generated permutation string like '[3] [1] [2] ...' into 0-indexed list.
    Any missing indices are appended at the end (fallback to original order).
    """
    nums, seen = [], set()
    for tok in text.replace(",", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and (v - 1) not in seen:
                nums.append(v - 1)
                seen.add(v - 1)
        except ValueError:
            continue
    # append any missing indices (keeps original BM25 order for unranked docs)
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums[:n]


def rerank(model, tokenizer, query: str, candidates: list) -> list:
    input_ids, attn_mask = build_fid_input(query, candidates, tokenizer)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    perm_text = tokenizer.decode(out[0], skip_special_tokens=True)
    perm      = parse_permutation(perm_text, len(candidates))
    return [candidates[i] for i in perm]


# ── Evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(name: str, model_path) -> float:
    """
    Returns nDCG@10.
    model_path=None  →  BM25 only (no reranking).
    """
    print(f"\n{'─'*55}")
    print(f"Evaluating: {name}")

    top_k = TOP_K   # may fall back to 10 on OOM

    if model_path is not None:
        use_bf16  = device.type == "cuda" and torch.cuda.is_bf16_supported()
        tokenizer = T5Tokenizer.from_pretrained(
            str(model_path), legacy=False, use_fast=True
        )
        model = T5ForConditionalGeneration.from_pretrained(str(model_path))
        if use_bf16:
            model = model.bfloat16()
            print("  Using bfloat16")
        model.to(device).eval()
        print(f"  Model loaded from {model_path}")
    else:
        model = tokenizer = None

    run = []
    oom_fallback = False

    for qid, qtext in tqdm(queries.items(), desc=name[:35]):
        candidates = bm25_retrieve(qtext, top_k=top_k)

        if model is not None:
            try:
                ranked = rerank(model, tokenizer, qtext, candidates)
            except torch.cuda.OutOfMemoryError:
                if not oom_fallback:
                    print(f"\n  ⚠ OOM with top_k={top_k} — falling back to top_k=5")
                    oom_fallback = True
                    top_k = 5
                torch.cuda.empty_cache()
                gc.collect()
                candidates = bm25_retrieve(qtext, top_k=top_k)
                ranked     = rerank(model, tokenizer, qtext, candidates)
        else:
            ranked = candidates  # BM25 order unchanged

        # Use 1/(rank+1) as score so nDCG ordering is correct
        for rank, doc in enumerate(ranked):
            run.append(ScoredDoc(qid, doc["doc_id"], score=1.0 / (rank + 1)))

    result = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)
    score  = result[nDCG @ 10]
    print(f"  nDCG@10 = {score:.4f}")

    if model is not None:
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return score


# ── Run all evaluations ───────────────────────────────────────────────────────
results = {}

# BM25 baseline
results["BM25 only"] = evaluate_model("BM25 only", None)

# LiT5 models
for name, path in MODELS.items():
    if path.exists():
        results[name] = evaluate_model(name, path)
    else:
        print(f"\n  Skipping {name} — path not found: {path}")

# ── Print + save results ──────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"{'Model':<38} {'nDCG@10':>10}")
print(f"{'─'*38} {'─'*10}")
for name, score in results.items():
    print(f"{name:<38} {score:>10.4f}")

write_header = not OUT_FILE.exists()
with OUT_FILE.open("a") as f:
    if write_header:
        f.write("model\tnDCG@10\tdataset\n")
    for name, score in results.items():
        f.write(f"{name}\t{score:.4f}\tTask13BGoldenEnriched\n")

print(f"\nResults saved → {OUT_FILE}")