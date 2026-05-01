"""
Sliding-window listwise reranking with DeepSeek-Chat on BioASQ top-20 BM25 docs.
Uses RankGPT-style prompt ([] > [] > ... format) from deepseek_rerank.py.

Algorithm (RankGPT-style, bottom-up):
  For N=20 docs, window=10, step=5:
    Pass 1: rerank positions [10-20]
    Pass 2: rerank positions [5-15]
    Pass 3: rerank positions [0-10]
  Better documents "bubble up" through successive windows.

Each document is truncated to 512 tokens using tiktoken.
Supports resume: re-running skips queries already written to the output file.

Reads:
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Writes:
  data/bioasq/bm25_top100/prompt engineering/deepseek_sliding_reranked_512.jsonl

Output per query:
  {
    "qid":             "...",
    "query":           "...",
    "bm25_top20":      ["docid1", ..., "docid20"],
    "reranked":        ["docid3", "docid1", ...],   # final sliding-window order
    "n_prompt_tokens": int,                          # total tokens across all windows
    "missing_docids":  [...]
  }

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_rerank_sliding.py --n-queries 1 --verbose
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

BASE     = Path(__file__).resolve().parents[4]
IDS_FILE = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
CORPUS   = BASE / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
OUT_FILE = BASE / "data/bioasq/bm25_top100/prompt engineering/deepseek_sliding_reranked_512_rankgpt.jsonl"

TOP_N             = 20
WINDOW_SIZE       = 10
STEP              = 5
MODEL             = "deepseek-chat"
PASSAGE_TOKEN_CAP = 512

# tiktoken has no DeepSeek encoder but cl100k_base is a close proxy
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── RankGPT-style prompt (same as deepseek_rerank.py) ────────────────────────
SYSTEM_PROMPT = (
    "You are RankLLM, an intelligent assistant that ranks biomedical passages "
    "based on their relevance to a query."
)

PRE_PROMPT = (
    "I will provide you with {n} passages, each indicated by a numerical identifier []. "
    "Rank the passages based on their relevance to the biomedical query: {query}"
)

POST_PROMPT = (
    "Biomedical query: {query}\n\n"
    "Rank the {n} passages above based on their relevance to the query. "
    "The passages should be listed in descending order of relevance, using their "
    "identifiers. The most relevant passages should be listed first. "
    "The output format should be [] > [] > ..., e.g., [4] > [2] > [1] > [3]. "
    "Only respond with the ranking results, do not say anything else or explain."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return TOKENIZER.decode(tokens[:max_tokens])


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def build_messages(query: str, docs: list[dict]) -> tuple[list[dict], int]:
    """Build the RankGPT message list for a window of docs.
    Returns (messages, prompt_token_count).
    """
    n = len(docs)

    passage_blocks = []
    for i, doc in enumerate(docs, start=1):
        title   = doc["title"].strip()
        text    = doc["text"].strip()
        passage = f"{title}\n{text}" if title else text
        passage = truncate_to_tokens(passage, PASSAGE_TOKEN_CAP)
        passage_blocks.append(f"[{i}] {passage}")

    user_content = (
        PRE_PROMPT.format(n=n, query=query)
        + "\n\n"
        + "\n\n".join(passage_blocks)
        + "\n\n"
        + POST_PROMPT.format(n=n, query=query)
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    total_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(user_content)
    return messages, total_tokens


def parse_ranking(raw: str, n: int) -> list[int] | None:
    """
    Extract a ranked list of 1-based ints from the model response.
    Handles:
      1. RankGPT format:  [3] > [1] > [7] > [2] ...
      2. JSON-array fallback: [3, 1, 7, 2, ...]
    Appends any missing indices at the end.
    """
    # Try RankGPT bracket format first
    bracket_nums = re.findall(r"\[(\d+)\]", raw)
    ranking: list[int] = []
    if bracket_nums:
        for s in bracket_nums:
            try:
                ranking.append(int(s))
            except ValueError:
                pass

    # Fallback: JSON array
    if not ranking:
        match = re.search(r"\[[\d\s,]+\]", raw)
        if match:
            try:
                ranking = [int(x) for x in json.loads(match.group())]
            except (json.JSONDecodeError, TypeError, ValueError):
                return None

    if not ranking:
        return None

    # Deduplicate and keep only valid 1-based indices
    seen:  set[int]  = set()
    clean: list[int] = []
    for x in ranking:
        if 1 <= x <= n and x not in seen:
            clean.append(x)
            seen.add(x)

    # Append any missing indices in original order
    for i in range(1, n + 1):
        if i not in seen:
            clean.append(i)

    return clean if len(clean) == n else None


def call_deepseek(client, model: str, messages: list) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


# ── Sliding window ────────────────────────────────────────────────────────────
def sliding_window_rerank(
    client,
    model: str,
    query: str,
    doc_ids: list[str],
    corpus: dict[str, dict],
    window_size: int,
    step: int,
    verbose: bool = False,
) -> tuple[list[str], int]:
    """
    Bottom-up sliding window reranking using the RankGPT prompt.

    Returns:
        (reranked_doc_ids, total_prompt_tokens_across_all_windows)
    """
    ranked = list(doc_ids)          # mutable working list
    n      = len(ranked)
    total_tokens = 0
    window_num   = 0

    end = n
    while True:
        start  = max(0, end - window_size)
        window = ranked[start:end]
        w_docs = [corpus.get(did, {"title": "", "text": ""}) for did in window]

        messages, n_tokens = build_messages(query, w_docs)
        total_tokens += n_tokens
        window_num   += 1

        if verbose:
            print(f"\n  [Window {window_num}] positions [{start}:{end}] "
                  f"({len(window)} docs) | {n_tokens} prompt tokens")
            print(f"  Before: {window}")

        try:
            raw     = call_deepseek(client, model, messages)
            ranking = parse_ranking(raw, len(window))
            if ranking is None:
                if verbose:
                    print(f"  [PARSE FAIL] raw={raw[:120]}")
                # keep window order unchanged on parse failure
            else:
                ranked[start:end] = [window[i - 1] for i in ranking]
                if verbose:
                    print(f"  After : {ranked[start:end]}")
                    print(f"  Raw   : {raw[:120]}")
        except Exception as e:
            if verbose:
                print(f"  [API FAIL] {e}")
            # keep window order unchanged on API failure

        if start == 0:
            break
        end -= step
        time.sleep(0.1)   # gentle rate-limit pause between windows

    return ranked, total_tokens


# ── Corpus loader ─────────────────────────────────────────────────────────────
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sliding-window RankGPT reranker for BioASQ"
    )
    parser.add_argument("--n-queries", type=int, default=10,
                        help="Number of queries to process (default: 10)")
    parser.add_argument("--model",   default=MODEL)
    parser.add_argument("--top-n",   type=int, default=TOP_N,
                        help="BM25 candidates to rerank per query (default: 20)")
    parser.add_argument("--window",  type=int, default=WINDOW_SIZE,
                        help="Sliding window size (default: 10)")
    parser.add_argument("--step",    type=int, default=STEP,
                        help="Sliding window step size (default: 5)")
    parser.add_argument("--output",  default=str(OUT_FILE),
                        help="Output JSONL path")
    parser.add_argument("--verbose", action="store_true",
                        help="Print window-level details (useful for n=1 testing)")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")

    client   = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    out_path = Path(args.output)

    n_windows = max(1, (args.top_n - args.window) // args.step + 1) \
                if args.top_n > args.window else 1
    print(f"Config: top-{args.top_n} docs | window={args.window} | step={args.step} | "
          f"~{n_windows} windows/query | {PASSAGE_TOKEN_CAP} tokens/passage | model={args.model}")

    # ── Load queries ───────────────────────────────────────────────────────────
    records: list[dict] = []
    with IDS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
            if args.n_queries and len(records) == args.n_queries:
                break
    print(f"Loaded {len(records)} queries")

    # ── Resume support ─────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_qids = load_done_qids(out_path)
    if done_qids:
        print(f"Resuming: {len(done_qids)} queries already done, skipping.")
    records = [r for r in records if r["qid"] not in done_qids]
    if not records:
        print("Nothing to do.")
        return
    print(f"{len(records)} queries to process")

    # ── Load corpus ────────────────────────────────────────────────────────────
    needed = {h["docid"] for r in records for h in r["top100"][:args.top_n]}
    print(f"Loading {len(needed)} documents from corpus …")
    corpus  = load_corpus_subset(needed)
    missing_global = len(needed) - len(corpus)
    if missing_global:
        print(f"  Warning: {missing_global} docids not found in corpus")
    print(f"  {len(corpus)} found")

    # ── Rerank ─────────────────────────────────────────────────────────────────
    n_ok = n_fail = 0
    token_counts: list[int] = []

    with out_path.open("a") as out:
        for rec in tqdm(records, desc="Reranking"):
            top_ids = [h["docid"] for h in rec["top100"][:args.top_n]]
            missing = [did for did in top_ids if did not in corpus]

            if args.verbose:
                print(f"\n{'='*70}")
                print(f"QID   : {rec['qid']}")
                print(f"Query : {rec['query']}")
                print(f"Missing docids: {missing}")

            try:
                reranked, total_tokens = sliding_window_rerank(
                    client=client,
                    model=args.model,
                    query=rec["query"],
                    doc_ids=top_ids,
                    corpus=corpus,
                    window_size=args.window,
                    step=args.step,
                    verbose=args.verbose,
                )
                token_counts.append(total_tokens)
                n_ok += 1

                if args.verbose:
                    print(f"\nFinal reranked order: {reranked}")
                    print(f"Total prompt tokens (all windows): {total_tokens}")

            except Exception as e:
                print(f"\n  [FAIL] {rec['qid']}: {e}")
                reranked      = top_ids
                total_tokens  = 0
                n_fail       += 1

            out.write(json.dumps({
                "qid":             rec["qid"],
                "query":           rec["query"],
                "bm25_top20":      top_ids,        # always the original BM25 order
                "reranked":        reranked,        # sliding-window output
                "n_prompt_tokens": total_tokens,    # summed across all windows
                "missing_docids":  missing,
            }, ensure_ascii=False) + "\n")

            time.sleep(0.2)   # rate limit between queries

    if token_counts:
        avg_tok = sum(token_counts) / len(token_counts)
        max_tok = max(token_counts)
        print(f"\nPrompt tokens (all windows) — avg: {avg_tok:.0f}, max: {max_tok}")
    print(f"Done — {n_ok} ok, {n_fail} failed → {out_path}")


if __name__ == "__main__":
    main()