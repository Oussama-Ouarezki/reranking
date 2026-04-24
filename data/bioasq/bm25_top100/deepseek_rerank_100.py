"""
Pass top-100 BM25 PubMed abstracts per query to DeepSeek for listwise ranking.
Each document is truncated to 300 tokens using tiktoken.

Reads:
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Writes:
  data/bioasq/bm25_top100/deepseek_reranked_100.jsonl

Output per query:
  {
    "qid":         "...",
    "bm25_order":  ["docid1", ..., "docid100"],   # original BM25 order
    "permutation": ["docid3", "docid1", ...],     # DeepSeek order (best first)
  }

Supports resume: re-running skips queries already written to the output file.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_rerank_100.py --n-queries 10
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_rerank_100.py          # all 2000
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import tiktoken
from openai import OpenAI
from tqdm import tqdm

BASE     = Path(__file__).resolve().parents[3]
IDS_FILE = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
CORPUS   = BASE / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
OUT_FILE = BASE / "data/bioasq/bm25_top100/deepseek_reranked_100.jsonl"

TOP_N      = 100
MAX_TOKENS = 300
MODEL      = "deepseek-reasoner"

# cl100k_base is the closest public tokenizer to DeepSeek's tokenizer
TOKENIZER = tiktoken.get_encoding("cl100k_base")

SYSTEM_PROMPT = (
    "You are an expert biomedical reranker specializing in PubMed literature. "
    "Given a biomedical question and a numbered list of PubMed abstracts with their BM25 "
    "lexical relevance scores, rank all abstracts from most to least relevant to the question. "
    "Use both the BM25 score (lexical match signal) and the semantic content to determine relevance. "
    "Respond with a JSON object with a single key \"ranking\" whose value is a list of ALL "
    "document numbers in order, best first. "
    "Example for 5 documents: {\"ranking\": [3, 1, 5, 2, 4]}"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def truncate_doc(title: str, text: str, max_tokens: int = MAX_TOKENS) -> str:
    passage = f"{title}. {text}".strip() if title else text.strip()
    tokens  = TOKENIZER.encode(passage)
    if len(tokens) <= max_tokens:
        return passage
    return TOKENIZER.decode(tokens[:max_tokens]) + " [...]"


def build_user_prompt(query: str, docs: list[dict], bm25_scores: list[float]) -> str:
    lines = [f"Biomedical question: {query}\n\nPubMed abstracts:"]
    for i, (doc, score) in enumerate(zip(docs, bm25_scores), start=1):
        passage = truncate_doc(doc["title"], doc["text"])
        lines.append(f"[{i}] BM25: {score:.1f} | {passage}")
    lines.append(
        f"\nRank all {len(docs)} abstracts from most to least relevant to the question above. "
        f"Combine BM25 lexical scores and semantic relevance. "
        f"Return a JSON object with key \"ranking\" containing all numbers 1 through "
        f"{len(docs)} exactly once, best first."
    )
    return "\n".join(lines)


def _coerce_int(x) -> int | None:
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x == int(x):
        return int(x)
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return None


def _clean(seq: list, n: int) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for raw_x in seq:
        x = _coerce_int(raw_x)
        if x is not None and 1 <= x <= n and x not in seen:
            result.append(x)
            seen.add(x)
    for i in range(1, n + 1):
        if i not in seen:
            result.append(i)
    return result


def parse_ranking(raw: str, n: int) -> list[int]:
    """
    Extract a permutation of 1..n from the model response.
    Strategy 1: JSON object with "ranking" / "ranked" / "order" key.
    Strategy 2: Last bracket array with >= n//2 items (skips short commentary brackets).
    Strategy 3: Scrape all integers 1..n from text in order.
    Fallback:   Identity (BM25 order unchanged).
    Always returns a complete list of length n.
    """
    # Strategy 1 — JSON object
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

    # Strategy 2 — last valid bracket array with enough items
    for m in reversed(list(re.finditer(r'\[[\d\s.,\"\']+\]', raw))):
        try:
            arr = json.loads(m.group())
            if isinstance(arr, list) and len(arr) >= n // 2:
                return _clean(arr, n)
        except (json.JSONDecodeError, TypeError):
            pass

    # Strategy 3 — scrape integers from text (supports up to 3-digit numbers for n=100)
    nums = [int(x) for x in re.findall(r'\b([1-9][0-9]{0,2})\b', raw) if int(x) <= n]
    if nums:
        return _clean(nums, n)

    # Fallback — identity
    return list(range(1, n + 1))


def load_corpus_subset(docids: set[str]) -> dict[str, dict]:
    corpus: dict[str, dict] = {}
    with CORPUS.open() as f:
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


def load_done_qids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def call_api(client, model: str, messages: list) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=1,   # required for deepseek-reasoner
    )
    return resp.choices[0].message.content or ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Max queries to process. Omit to process all 2000.")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # ── Load queries ───────────────────────────────────────────────────────────
    records = []
    with IDS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
    if args.n_queries:
        records = records[:args.n_queries]

    # ── Resume: skip already-done queries ─────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_qids = load_done_qids(OUT_FILE)
    if done_qids:
        print(f"Resuming: {len(done_qids)} queries already done, skipping.")
    records = [r for r in records if r["qid"] not in done_qids]
    print(f"{len(records)} queries to process  |  model: {args.model}  |  top-{TOP_N} docs  |  {MAX_TOKENS} tokens/doc")

    if not records:
        print("Nothing to do.")
        return

    # ── Load passage texts ─────────────────────────────────────────────────────
    needed = {h["docid"] for r in records for h in r["top100"][:TOP_N]}
    print(f"Loading {len(needed)} documents from corpus …")
    corpus = load_corpus_subset(needed)
    missing = len(needed) - len(corpus)
    if missing:
        print(f"  Warning: {missing} docids not found in corpus (will appear as empty)")
    print(f"  {len(corpus)} found")

    # ── Rerank ─────────────────────────────────────────────────────────────────
    n_ok = n_fail = 0

    with OUT_FILE.open("a") as out:
        for rec in tqdm(records, desc="Reranking"):
            top100      = rec["top100"][:TOP_N]
            top100_ids  = [h["docid"] for h in top100]
            bm25_scores = [h["score"] for h in top100]
            docs        = [corpus.get(did, {"title": "", "text": ""}) for did in top100_ids]
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(rec["query"], docs, bm25_scores)},
            ]

            try:
                raw         = call_api(client, args.model, messages)
                ranking     = parse_ranking(raw, TOP_N)
                permutation = [top100_ids[i - 1] for i in ranking]
                n_ok       += 1
            except Exception as e:
                print(f"\n  [FAIL] {rec['qid']}: {e}")
                permutation = top100_ids
                n_fail     += 1

            out.write(json.dumps({
                "qid":         rec["qid"],
                "bm25_order":  top100_ids,
                "permutation": permutation,
            }, ensure_ascii=False) + "\n")

            time.sleep(0.3)

    print(f"\nDone — {n_ok} ok, {n_fail} failed → {OUT_FILE}")


if __name__ == "__main__":
    main()
