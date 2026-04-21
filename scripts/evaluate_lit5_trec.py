"""
Evaluate BM25 and LiT5-Distill on TREC DL19 and DL20 using nDCG@10.

Fixes vs original:
  - Correct FiD-style encoding: each query-passage pair encoded separately,
    stacked as [1, n_passages, seq_len] — not concatenated into one string
  - text_maxlength raised to 150 (matches paper)
  - Sliding window (size=20, stride=10) over top-100 BM25 candidates
  - N_QUERIES cap for fast smoke-test runs (set to None for full eval)

Models compared:
  - BM25 only
  - LiT5-Distill-base (MS MARCO pretrained)
  - checkpoint1 (BioASQ fine-tuned, dataset1)
  - checkpoint2 (BioASQ fine-tuned, dataset2)

Output: evaluation/scores_trec.tsv
"""

import random
from pathlib import Path

import bm25s
import torch
import ir_measures
from ir_measures import nDCG, ScoredDoc, parse_trec_qrels
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION_PATH = Path("data/msmarco-passage/collection.tsv")
OUT_FILE        = Path("evaluation/scores_trec.tsv")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "dl19": {
        "queries": "data/trec/queries.dl19-passage.tsv",
        "qrels":   "data/trec/qrels.dl19-passage.txt",
    },
    "dl20": {
        "queries": "data/trec/queries.dl20-passage.tsv",
        "qrels":   "data/trec/qrels.dl20-passage.txt",
    },
}

MODELS = {
    "LiT5-base (MS MARCO)":    Path("checkpoints/LiT5-Distill-base"),
    "checkpoint1 (BioASQ-d1)": Path("data/bioasq/training_strat/checkpoint1"),
    "checkpoint2 (BioASQ-d2)": Path("data/bioasq/training_strat/checkpoint2"),
}

# ── Set this to an int (e.g. 10) for a quick smoke-test, None for full eval ──
N_QUERIES      = 10          # <- change to None to run all queries

N_NEGATIVES    = 500_000
TOP_BM25       = 100
WINDOW_SIZE    = 20          # LiT5-Distill uses window of 20 passages
STRIDE         = 10          # sliding window stride
TEXT_MAXLENGTH = 150         # matches the paper (was 64 — too short)
MAX_NEW_TOKENS = 140         # enough to output "[20] > [19] > ... > [1]"
SEED           = 42

# Input format from the paper
QUERY_PREFIX  = "Search Query:"
PASS_PREFIX   = "Passage:"
SUFFIX        = " Relevance Ranking:"

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
print(f"Device: {device}  bf16: {USE_BF16}")
if N_QUERIES:
    print(f"⚠  Smoke-test mode: only {N_QUERIES} queries per dataset")


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_queries(path: str, limit=None) -> dict:
    queries = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
            if limit and len(queries) >= limit:
                break
    return queries


def load_qrel_docids(paths: list) -> set:
    doc_ids = set()
    for path in paths:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    doc_ids.add(parts[2])
    return doc_ids


# ── Build pool ────────────────────────────────────────────────────────────────
def build_corpus_pool(qrel_doc_ids: set) -> tuple:
    print(f"Building corpus pool (qrel docs + {N_NEGATIVES:,} random negatives) …")
    rng = random.Random(SEED)
    total_lines = 8_841_823
    sample_prob = N_NEGATIVES / (total_lines - len(qrel_doc_ids))

    pool_ids, pool_texts = [], []
    qrel_added = 0

    with COLLECTION_PATH.open() as f:
        for line in tqdm(f, total=total_lines, desc="Scanning collection"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            doc_id, text = parts[0], parts[1]
            if doc_id in qrel_doc_ids:
                pool_ids.append(doc_id)
                pool_texts.append(text)
                qrel_added += 1
            elif len(pool_ids) - qrel_added < N_NEGATIVES and rng.random() < sample_prob:
                pool_ids.append(doc_id)
                pool_texts.append(text)

    print(f"  Pool size: {len(pool_ids):,}  (qrel docs: {qrel_added:,})")
    corpus_map = dict(zip(pool_ids, pool_texts))
    return pool_ids, pool_texts, corpus_map


# ── BM25 ──────────────────────────────────────────────────────────────────────
def build_bm25(pool_ids, pool_texts):
    print("Indexing with bm25s …")
    tokenized = bm25s.tokenize(pool_texts, show_progress=True)
    retriever  = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(tokenized, show_progress=True)
    return retriever


def bm25_retrieve_all(retriever, pool_ids, queries, k):
    qids   = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    tok_q  = bm25s.tokenize(qtexts, show_progress=False)
    hits, scores = retriever.retrieve(tok_q, k=min(k, len(pool_ids)),
                                      show_progress=True)
    return {
        qid: [(pool_ids[hits[i, j]], float(scores[i, j]))
              for j in range(hits.shape[1])]
        for i, qid in enumerate(qids)
    }


# ── FiD-style LiT5-Distill reranking ─────────────────────────────────────────
def build_fid_inputs(query: str, passages: list, tokenizer) -> dict:
    """
    Encode each (query, passage) pair independently — this is the FiD pattern.
    passages: list of (doc_id, score, text)
    Returns input_ids and attention_mask shaped [1, n_passages, seq_len].
    """
    strings = [
        f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i+1}] {ptext}{SUFFIX}"
        for i, (_, _, ptext) in enumerate(passages)
    ]
    enc = tokenizer(
        strings,
        max_length=TEXT_MAXLENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    # Shape: [n_passages, seq_len] → [1, n_passages, seq_len]
    return {
        "input_ids":      enc.input_ids.unsqueeze(0).to(device),
        "attention_mask": enc.attention_mask.unsqueeze(0).to(device),
    }


def parse_ranking(perm_text: str, n: int) -> list:
    """Parse '[4] > [2] > ... > [1]' output into 0-indexed permutation."""
    nums, seen = [], set()
    for tok in perm_text.replace(",", " ").replace(">", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and (v - 1) not in seen:
                nums.append(v - 1)
                seen.add(v - 1)
        except ValueError:
            continue
    # append any missing indices at the end (fallback)
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums


def sliding_window_rerank(model, tokenizer, query: str,
                           candidates: list) -> list:
    """
    Apply sliding window reranking over candidates.
    candidates: list of (doc_id, score, text), length = TOP_BM25
    Returns reranked list of (doc_id, score, text).
    """
    ranked = list(candidates)  # current ordering, updated in-place each pass

    # single pass, bottom-up (same as original LiT5 paper)
    start = max(0, len(ranked) - WINDOW_SIZE)
    while True:
        end     = min(start + WINDOW_SIZE, len(ranked))
        window  = ranked[start:end]
        n_win   = len(window)

        enc = build_fid_inputs(query, window, tokenizer)

        with torch.no_grad():
            out = model.generate(
                input_ids=enc["input_ids"].view(1, n_win, -1).squeeze(0)
                    if enc["input_ids"].dim() == 3 else enc["input_ids"],
                attention_mask=enc["attention_mask"].view(1, n_win, -1).squeeze(0)
                    if enc["attention_mask"].dim() == 3 else enc["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        perm_text = tokenizer.decode(out[0], skip_special_tokens=True)
        perm      = parse_ranking(perm_text, n_win)

        ranked[start:end] = [window[i] for i in perm]

        if start == 0:
            break
        start = max(0, start - STRIDE)

    return ranked


def to_scored_docs(results: dict) -> list:
    run = []
    for qid, docs in results.items():
        for rank, (doc_id, _) in enumerate(docs):
            run.append(ScoredDoc(qid, doc_id, score=len(docs) - rank))
    return run


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
all_qrel_ids = load_qrel_docids([ds["qrels"] for ds in DATASETS.values()])
print(f"Qrel doc IDs (both datasets): {len(all_qrel_ids):,}")

pool_ids, pool_texts, corpus_map = build_corpus_pool(all_qrel_ids)
retriever = build_bm25(pool_ids, pool_texts)

all_results = {}

for ds_name, ds_cfg in DATASETS.items():
    queries = load_queries(ds_cfg["queries"], limit=N_QUERIES)
    qrels   = list(parse_trec_qrels(ds_cfg["qrels"]))

    # filter qrels to only the loaded queries (important for subset eval)
    loaded_qids = set(queries.keys())
    qrels_sub   = [q for q in qrels if q.query_id in loaded_qids]

    print(f"\n{'═'*55}")
    print(f"Dataset: {ds_name.upper()}  "
          f"({len(queries)} queries, {len(qrels_sub)} qrel entries)")

    # ── BM25 baseline ─────────────────────────────────────────────────────────
    bm25_results = bm25_retrieve_all(retriever, pool_ids, queries, TOP_BM25)
    score = ir_measures.calc_aggregate(
        [nDCG @ 10], qrels_sub, to_scored_docs(bm25_results))[nDCG @ 10]
    print(f"  {'BM25 only':<32} nDCG@10 = {score:.4f}")
    all_results[("BM25 only", ds_name)] = score

    # prepare (doc_id, score, text) triples
    candidates_per_query = {
        qid: [(d, s, corpus_map.get(d, "")) for d, s in docs]
        for qid, docs in bm25_results.items()
    }

    # ── LiT5 models ───────────────────────────────────────────────────────────
    for model_name, model_path in MODELS.items():
        print(f"\n  Loading {model_name} …")
        tokenizer = T5Tokenizer.from_pretrained(
            str(model_path), legacy=False, use_fast=True)
        model = T5ForConditionalGeneration.from_pretrained(str(model_path))
        if USE_BF16:
            model = model.bfloat16()
        model.to(device).eval()

        reranked = {}
        for qid, qtext in tqdm(queries.items(), desc=f"  Reranking [{model_name[:22]}]"):
            ranked = sliding_window_rerank(
                model, tokenizer, qtext, candidates_per_query[qid])
            reranked[qid] = [(d, s) for d, s, _ in ranked]

        score = ir_measures.calc_aggregate(
            [nDCG @ 10], qrels_sub, to_scored_docs(reranked))[nDCG @ 10]
        print(f"  {model_name:<32} nDCG@10 = {score:.4f}")
        all_results[(model_name, ds_name)] = score

        del model
        torch.cuda.empty_cache()

# ── Print summary & save ──────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"{'Model':<34} {'DL19':>8} {'DL20':>8}")
print(f"{'─'*34} {'─'*8} {'─'*8}")

model_names    = ["BM25 only"] + list(MODELS.keys())
write_header   = not OUT_FILE.exists()

with OUT_FILE.open("a") as f:
    if write_header:
        f.write("model\tdataset\tnDCG@10\n")
    for name in model_names:
        dl19 = all_results.get((name, "dl19"), float("nan"))
        dl20 = all_results.get((name, "dl20"), float("nan"))
        print(f"{name:<34} {dl19:>8.4f} {dl20:>8.4f}")
        f.write(f"{name}\tdl19\t{dl19:.4f}\n")
        f.write(f"{name}\tdl20\t{dl20:.4f}\n")

print(f"\nSaved → {OUT_FILE}")
if N_QUERIES:
    print(f"⚠  Results are from {N_QUERIES} queries only — "
          f"set N_QUERIES = None for full evaluation.")