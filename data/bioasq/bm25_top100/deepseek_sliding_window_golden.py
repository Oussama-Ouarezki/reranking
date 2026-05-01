"""
BM25 retrieval (k1=0.7, b=0.9, top-50) then sliding-window listwise reranking
with DeepSeek-Chat on all Task13BGoldenEnriched queries.

Algorithm (RankGPT-style, bottom-up):
  window=20, step=10 over 50 BM25 candidates.

Reads:
  data/bioasq/raw/Task13BGoldenEnriched/queries.jsonl
  data/bioasq/pubmed_full/full/corpus_full.jsonl
  Lucene index: data/bm25_indexing_full/corpus_full/lucene_index

Writes:
  data/bioasq/bm25_top100/deepseek_sliding_reranked_golden.jsonl

Supports resume: re-running skips queries already written to the output file.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_sliding_window_golden.py
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_sliding_window_golden.py --n-queries 5
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

# Java must be set before any pyserini import
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import tiktoken
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm

BASE       = Path(__file__).resolve().parents[3]
QUERY_FILE = BASE / "data/bioasq/raw/Task13BGoldenEnriched/queries.jsonl"
CORPUS     = BASE / "data/bioasq/pubmed_full/full/corpus_full.jsonl"
INDEX      = BASE / "data/bm25_indexing_full/corpus_full/lucene_index"
OUT_FILE   = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_golden.jsonl"

BM25_K1     = 0.7
BM25_B      = 0.9
TOP_N       = 50
WINDOW_SIZE = 20
STEP        = 10
MAX_TOKENS  = 512
MODEL       = "deepseek-chat"

TOKENIZER = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = (
    "You are a biomedical literature reranker. Your task is to rank PubMed abstracts "
    "by their in-depth relevance and direct informational value to a given clinical or "
    "research question.\n\n"
    "For each abstract, assess:\n"
    "- Direct topical match (primary): does it address the exact condition, mechanism, "
    "drug, gene, or population in the question? This always outranks other criteria.\n"
    "- Answer completeness: would this abstract directly help answer the question "
    "without further lookup?\n"
    "- Evidence quality (tiebreaker only): prefer primary research over editorials, "
    "but never promote a high-evidence abstract above a more topically relevant one.\n\n"
    "Rank abstracts from most to least relevant by comparing them against each other. "
    "The BM25 score is a meaningful signal — top-ranked BM25 documents are usually correct. "
    "Only demote a high-scoring BM25 document if you are highly confident another abstract "
    "provides substantially more precise or complete information about the question.\n\n"
    "Avoid ranking an abstract highly solely because it shares vocabulary with the "
    "question. Relevance is determined by informational depth and accuracy, not lexical "
    "overlap. When in doubt between two similarly relevant abstracts, prefer the one with "
    "the higher BM25 score.\n\n"
    "Respond ONLY with JSON. No explanation.\n"
    'Format: {"ranking": [<most_relevant_id>, ..., <least_relevant_id>]}\n'
    'Example for 5 documents: {"ranking": [3, 1, 5, 2, 4]}'
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    tokens = TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return TOKENIZER.decode(tokens[:max_tokens]) + " [...]"


def build_user_prompt(query: str, docs: list, bm25_scores: list) -> str:
    n = len(docs)
    lines = [
        f"Biomedical question: {query}\n",
        "Ranking criteria (apply in this order):",
        "  1. Direct topical match to the question's condition, mechanism, drug, gene, or population.",
        "  2. Answer completeness — abstracts that directly answer the question rank above tangentially related ones.",
        "  3. Evidence quality — tiebreaker only; never overrides topical match or completeness.",
        "  4. BM25 score — use as a meaningful signal; only demote a high-scoring document if another is clearly more relevant.\n",
        f"PubMed abstracts (n={n}):",
    ]
    for i, (doc, score) in enumerate(zip(docs, bm25_scores), start=1):
        lines.append(f"[{i}] BM25={score:.1f}")
        lines.append(f"    Title: {doc['title'].rstrip('.')}")
        lines.append(f"    Abstract: {truncate_text(doc['text'])}")
    lines.append(
        f"\nApply the criteria above. Compare abstracts against each other, not just "
        f"against the question. Return a JSON object with key \"ranking\" containing "
        f"all integers 1 through {n} exactly once, best first."
    )
    return "\n".join(lines)


def _coerce_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x == int(x):
        return int(x)
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return None


def _clean(seq: list, n: int) -> list:
    seen = set()
    result = []
    for raw_x in seq:
        x = _coerce_int(raw_x)
        if x is not None and 1 <= x <= n and x not in seen:
            result.append(x)
            seen.add(x)
    for i in range(1, n + 1):
        if i not in seen:
            result.append(i)
    return result


def parse_ranking(raw: str, n: int) -> list:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ("ranking", "ranked", "order", "result"):
                if key in obj and isinstance(obj[key], list):
                    return _clean(obj[key], n)
            for val in obj.values():
                if isinstance(val, list):
                    return _clean(val, n)
        if isinstance(obj, list):
            return _clean(obj, n)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    for m in reversed(list(re.finditer(r'\[[\d\s.,\"\']+\]', raw))):
        try:
            arr = json.loads(m.group())
            if isinstance(arr, list) and len(arr) >= n // 2:
                return _clean(arr, n)
        except (json.JSONDecodeError, TypeError):
            pass

    nums = [int(x) for x in re.findall(r'\b([1-9][0-9]?)\b', raw) if int(x) <= n]
    if nums:
        return _clean(nums, n)

    return list(range(1, n + 1))


def call_api(client, model: str, messages: list) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=128,
    )
    return resp.choices[0].message.content or ""


def sliding_window_rerank(client, model: str, query: str,
                          doc_ids: list, bm25_scores: list,
                          corpus: dict) -> list:
    ranked = list(zip(doc_ids, bm25_scores))
    n = len(ranked)
    end = n
    while True:
        start    = max(0, end - WINDOW_SIZE)
        window   = ranked[start:end]
        w_ids    = [d for d, _ in window]
        w_scores = [s for _, s in window]
        w_docs   = [corpus.get(did, {"title": "", "text": ""}) for did in w_ids]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(query, w_docs, w_scores)},
        ]
        try:
            raw     = call_api(client, model, messages)
            ranking = parse_ranking(raw, len(window))
            ranked[start:end] = [(w_ids[i - 1], w_scores[i - 1]) for i in ranking]
        except Exception:
            pass
        if start == 0:
            break
        end -= STEP
        time.sleep(0.1)
    return [d for d, _ in ranked]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_queries(path: Path) -> dict:
    queries = {}
    with path.open() as f:
        for line in f:
            doc = json.loads(line)
            queries[doc["_id"]] = doc["text"]
    return queries


def bm25_retrieve(searcher: LuceneSearcher, queries: dict, top_n: int) -> dict:
    """Returns {qid: [(docid, score), ...]} for top_n hits per query."""
    results = {}
    for qid, text in tqdm(queries.items(), desc="BM25 retrieval"):
        hits = searcher.search(text, k=top_n)
        results[qid] = [(h.docid, h.score) for h in hits]
    return results


def load_corpus_subset(docids: set, corpus_path: Path) -> dict:
    corpus = {}
    with corpus_path.open() as f:
        for line in f:
            doc = json.loads(line)
            if doc["_id"] in docids:
                corpus[doc["_id"]] = {
                    "title": doc.get("title", ""),
                    "text":  doc["text"],
                }
            if len(corpus) == len(docids):
                break
    return corpus


def load_done_qids(path: Path) -> set:
    done = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument("--model",     default=MODEL)
    parser.add_argument("--top-n",     type=int, default=TOP_N)
    parser.add_argument("--window",    type=int, default=WINDOW_SIZE)
    parser.add_argument("--step",      type=int, default=STEP)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    n_windows = max(1, (args.top_n - args.window) // args.step + 1) if args.top_n > args.window else 1
    print(f"Config: top-{args.top_n} | window={args.window} | step={args.step} | "
          f"~{n_windows} windows/query | {MAX_TOKENS} tok/doc | BM25 k1={BM25_K1} b={BM25_B}")

    # Load queries
    print("Loading queries …")
    queries = load_queries(QUERY_FILE)
    if args.n_queries:
        queries = dict(list(queries.items())[:args.n_queries])
    print(f"  {len(queries)} queries")

    # BM25 retrieval
    print(f"Opening Lucene index at {INDEX} …")
    searcher = LuceneSearcher(str(INDEX))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)
    bm25_results = bm25_retrieve(searcher, queries, args.top_n)

    # Resume
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_qids = load_done_qids(OUT_FILE)
    if done_qids:
        print(f"Resuming: {len(done_qids)} already done, skipping.")
    todo = {qid: hits for qid, hits in bm25_results.items() if qid not in done_qids}
    print(f"{len(todo)} queries to rerank")

    if not todo:
        print("Nothing to do.")
        return

    # Load corpus subset
    needed = {docid for hits in todo.values() for docid, _ in hits}
    print(f"Loading {len(needed)} documents from corpus …")
    corpus = load_corpus_subset(needed, CORPUS)
    missing = len(needed) - len(corpus)
    if missing:
        print(f"  Warning: {missing} docids not found in corpus")
    print(f"  {len(corpus)} found")

    # Rerank
    n_ok = n_fail = 0
    with OUT_FILE.open("a") as out:
        for qid, hits in tqdm(todo.items(), desc="Reranking"):
            doc_ids     = [docid for docid, _ in hits]
            bm25_scores = [score for _, score in hits]
            query       = queries[qid]

            try:
                permutation = sliding_window_rerank(
                    client, args.model, query,
                    doc_ids, bm25_scores, corpus,
                )
                n_ok += 1
            except Exception as e:
                print(f"\n  [FAIL] {qid}: {e}")
                permutation = doc_ids
                n_fail += 1

            out.write(json.dumps({
                "qid":         qid,
                "bm25_order":  doc_ids,
                "permutation": permutation,
            }, ensure_ascii=False) + "\n")

    print(f"\nDone — {n_ok} ok, {n_fail} failed → {OUT_FILE}")


if __name__ == "__main__":
    main()
