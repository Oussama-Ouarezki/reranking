"""
Build two listwise reranking distillation datasets from BioASQ using local Ollama.

Dataset 1: newest 200 queries (any doc count)
Dataset 2: newest 200 queries with 5+ relevant documents

Pipeline per query:
  1. BM25 top-20 retrieval (k1=0.7, b=0.9) over BioASQ corpus
  2. Truncate each passage so query + passage <= 512 tokens (whitespace split)
  3. Shuffle passages randomly (avoid positional bias)
  4. Send to local llama3.1:8b via Ollama -> get permutation most to least relevant
  5. Save JSONL with query, shuffled passages, and ranked output

Output:
  data/bioasq/bm25_doc/dataset1_rerank.jsonl
  data/bioasq/bm25_doc/dataset2_rerank.jsonl

Requirements: ollama must be running (`ollama serve`) and llama3.1:8b pulled.
"""

import json
import random
from pathlib import Path

import ollama
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.1:8b"
TOP_K        = 20
MAX_TOKENS   = 512
N_QUERIES    = 200
SEED         = 42

CORPUS_PATH  = Path("data/bioasq/processed/corpus.jsonl")
QUERIES_PATH = Path("data/bioasq/processed/queries.jsonl")
QRELS_PATH   = Path("data/bioasq/processed/qrels.tsv")
OUT_DIR      = Path("data/bioasq/bm25_doc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(SEED)


# ── Helpers ───────────────────────────────────────────────────────────────────
def truncate_passage(query: str, passage: str, max_tokens: int = MAX_TOKENS) -> str:
    q_tokens = query.split()
    p_tokens = passage.split()
    budget   = max_tokens - len(q_tokens)
    return " ".join(p_tokens[:max(budget, 0)])


def build_prompt(query: str, passages: list) -> str:
    numbered = "\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))
    return (
        f"I will provide a biomedical question and {len(passages)} candidate passages "
        f"numbered [1] to [{len(passages)}].\n"
        f"Rank them from most relevant to least relevant for answering the question.\n"
        f"Output ONLY a comma-separated list of the numbers in ranked order, "
        f"e.g.: 3, 1, 5, 2, 4\n\n"
        f"Question: {query}\n\n"
        f"Passages:\n{numbered}\n\n"
        f"Ranked order (most to least relevant):"
    )


def parse_ranking(response_text: str, n: int) -> list:
    nums = []
    seen = set()
    for tok in response_text.replace(",", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and v - 1 not in seen:
                nums.append(v - 1)
                seen.add(v - 1)
        except ValueError:
            continue
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums[:n]


def call_ollama(prompt: str, retries: int = 3) -> str:
    for attempt in range(retries):
        try:
            resp = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
            return resp["response"].strip()
        except Exception as e:
            if attempt < retries - 1:
                continue
            raise e
    return ""


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading corpus …")
corpus = {}
with CORPUS_PATH.open() as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = (doc.get("title", "") + " " + doc["text"]).strip()

print("Loading queries …")
queries = []
with QUERIES_PATH.open() as f:
    for line in f:
        queries.append(json.loads(line))
queries.sort(key=lambda q: q.get("chronological_order", 0), reverse=True)

print("Loading qrels …")
qrels = {}
with QRELS_PATH.open() as f:
    next(f)
    for line in f:
        qid, doc_id, *_ = line.strip().split("\t")
        qrels.setdefault(qid, set()).add(doc_id)

# ── Select query sets ─────────────────────────────────────────────────────────
dataset1_queries = queries[:N_QUERIES]
dataset2_queries = [q for q in queries if len(qrels.get(q["_id"], set())) >= 5][:N_QUERIES]

print(f"Dataset 1: {len(dataset1_queries)} queries (newest {N_QUERIES})")
print(f"Dataset 2: {len(dataset2_queries)} queries (newest {N_QUERIES} with 5+ relevant docs)")
print(f"Estimated time: ~{len(dataset1_queries) + len(dataset2_queries)} requests "
      f"(speed depends on your GPU/CPU)")

# ── Build BM25 index ──────────────────────────────────────────────────────────
print("Building BM25 index (this may take a minute) …")
doc_ids   = list(corpus.keys())
tokenized = [corpus[d].lower().split() for d in doc_ids]
bm25      = BM25Okapi(tokenized, k1=0.7, b=0.9)
print("BM25 index ready.")


# ── Process one dataset ───────────────────────────────────────────────────────
def process_dataset(query_list: list, out_path: Path) -> None:
    results = []
    for q in tqdm(query_list, desc=out_path.name):
        qid   = q["_id"]
        qtext = q["text"]

        # BM25 retrieval
        scores   = bm25.get_scores(qtext.lower().split())
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
        top_docs = [
            {
                "doc_id":     doc_ids[i],
                "bm25_score": float(scores[i]),
                "text":       truncate_passage(qtext, corpus[doc_ids[i]]),
            }
            for i in top_idxs
        ]

        # Shuffle to remove positional bias
        shuffled = top_docs[:]
        rng.shuffle(shuffled)

        # Call local LLM
        prompt       = build_prompt(qtext, shuffled)
        raw_response = call_ollama(prompt)
        ranking      = parse_ranking(raw_response, len(shuffled))
        ranked_docs  = [shuffled[i] for i in ranking]

        results.append({
            "query_id":       qid,
            "query":          qtext,
            "shuffled_input": [{"doc_id": d["doc_id"], "bm25_score": d["bm25_score"]} for d in shuffled],
            "llm_raw":        raw_response,
            "ranked_output":  [{"rank": r + 1, "doc_id": d["doc_id"]} for r, d in enumerate(ranked_docs)],
        })

    with out_path.open("w") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved {len(results)} records → {out_path}")


# ── Run ───────────────────────────────────────────────────────────────────────
process_dataset(dataset1_queries, OUT_DIR / "dataset1_rerank.jsonl")
process_dataset(dataset2_queries, OUT_DIR / "dataset2_rerank.jsonl")
print("Done.")
