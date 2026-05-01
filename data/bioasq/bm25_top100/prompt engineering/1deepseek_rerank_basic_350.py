"""
Pass top-20 BM25 documents per query to DeepSeek for listwise ranking.
Used to generate distillation labels for LiT5-Distill fine-tuning on BioASQ.

Reads:
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Writes:
  data/bioasq/bm25_top100/deepseek_reranked.jsonl

Output per query:
  {
    "qid":           "...",
    "query":         "...",
    "bm25_top20":    ["docid1", ..., "docid20"],
    "reranked":      ["docid3", "docid1", ...],
    "raw_response":  "...",
    "n_prompt_tokens": int,
    "missing_docids": [...]   # docids not found in corpus (should be empty)
  }

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_rerank.py --n-queries 1 --verbose
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
OUT_FILE = BASE / "data/bioasq/bm25_top100/prompt engineering/deepseek_reranked_350.jsonl"

TOP_N             = 20
MODEL             = "deepseek-chat"
PASSAGE_TOKEN_CAP = 350          # per-passage truncation

# tiktoken has no DeepSeek encoder but cl100k_base is a close proxy
# (DeepSeek's tokenizer vocabulary is similar enough for length estimation)
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# RankGPT-style prompt — this is the format LiT5-Distill was trained with
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


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens using cl100k_base tokenizer."""
    tokens = TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return TOKENIZER.decode(tokens[:max_tokens])


def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def build_messages(query: str, docs: list[dict]) -> tuple[list[dict], int]:
    """Build the full message list and return (messages, total_prompt_tokens)."""
    n = len(docs)

    # Format each passage: title + abstract, truncated to 512 tokens
    passage_blocks = []
    for i, doc in enumerate(docs, start=1):
        title = doc["title"].strip()
        text  = doc["text"].strip()
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
    Handles two formats:
      1. RankGPT-style:  [3] > [1] > [7] > [2] ...
      2. JSON-array fallback: [3, 1, 7, 2, ...]
    Appends any missing indices at the end (tail of BM25 order would be better,
    but we don't have access to the original order here — caller handles that).
    """
    # Try RankGPT format first
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
    seen: set[int] = set()
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
    return corpus


def call_deepseek(client, model, messages):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=10,
                        help="Number of queries to process")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--verbose", action="store_true",
                        help="Print full prompt and raw response (useful for n=1 testing)")
    parser.add_argument("--output", default=str(OUT_FILE),
                        help="Output path (default: deepseek_reranked.jsonl)")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    out_path = Path(args.output)

    # Load queries
    records = []
    with IDS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) == args.n_queries:
                break
    print(f"Loaded {len(records)} queries")

    # Collect all needed docids and load their text
    needed = {d["docid"] for r in records for d in r["top100"][:TOP_N]}
    print(f"Loading {len(needed)} documents from corpus …")
    corpus = load_corpus_subset(needed)
    print(f"  {len(corpus)} found ({len(needed) - len(corpus)} missing)")

    # Rerank
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = n_parse_fail = 0
    token_counts = []

    with out_path.open("w") as out:
        for rec in tqdm(records, desc="Reranking"):
            top20_ids = [d["docid"] for d in rec["top100"][:TOP_N]]
            missing = [did for did in top20_ids if did not in corpus]
            docs = [corpus.get(did, {"title": "", "text": ""}) for did in top20_ids]

            messages, n_tokens = build_messages(rec["query"], docs)
            token_counts.append(n_tokens)

            if args.verbose:
                print(f"\n{'='*70}")
                print(f"QID: {rec['qid']}")
                print(f"Query: {rec['query']}")
                print(f"Prompt tokens: {n_tokens}")
                print(f"Missing docids: {missing}")
                print(f"\n--- USER MESSAGE (first 2000 chars) ---")
                print(messages[1]["content"][:2000])
                print(f"... [truncated, total {len(messages[1]['content'])} chars]")

            try:
                raw = call_deepseek(client, args.model, messages)
                ranking = parse_ranking(raw, TOP_N)
                if ranking is None:
                    reranked_ids = top20_ids
                    n_parse_fail += 1
                    print(f"\n  [PARSE FAIL] {rec['qid']}: {raw[:200]}")
                else:
                    reranked_ids = [top20_ids[i - 1] for i in ranking]
                    n_ok += 1
            except Exception as e:
                print(f"\n  [API FAIL] {rec['qid']}: {e}")
                raw = ""
                reranked_ids = top20_ids
                n_fail += 1

            if args.verbose:
                print(f"\n--- RAW RESPONSE ---")
                print(raw)
                print(f"\n--- RERANKED ORDER ---")
                print(reranked_ids)

            out.write(json.dumps({
                "qid":             rec["qid"],
                "query":           rec["query"],
                "bm25_top20":      top20_ids,
                "reranked":        reranked_ids,
                "raw_response":    raw,
                "n_prompt_tokens": n_tokens,
                "missing_docids":  missing,
            }, ensure_ascii=False) + "\n")

            time.sleep(0.3)  # rate limit

    if token_counts:
        avg_tok = sum(token_counts) / len(token_counts)
        max_tok = max(token_counts)
        print(f"\nPrompt tokens — avg: {avg_tok:.0f}, max: {max_tok}")
    print(f"Done — {n_ok} ok, {n_parse_fail} parse-fail, {n_fail} api-fail → {out_path}")


if __name__ == "__main__":
    main()