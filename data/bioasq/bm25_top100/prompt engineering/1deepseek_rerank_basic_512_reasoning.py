"""
Same listwise reranking pipeline as 1deepseek_rerank_basic_512.py but adapted
for DeepSeek's reasoning model (deepseek-reasoner).

Key differences from the chat version:
  - max_tokens set high (16 000) to accommodate the chain-of-thought
  - reasoning_content (the thinking trace) is captured and saved per query
  - temperature is fixed at 1 (required by reasoning models — ignored if passed)
  - system message is merged into the first user turn because some reasoning
    model endpoints do not support the system role

Reads:
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Writes:
  data/bioasq/bm25_top100/prompt engineering/deepseek_reranked_512_reasoning.jsonl

Output per query:
  {
    "qid":               "...",
    "query":             "...",
    "bm25_top20":        ["docid1", ..., "docid20"],
    "reranked":          ["docid3", "docid1", ...],
    "raw_response":      "...",      # final answer only
    "reasoning_content": "...",      # chain-of-thought (may be empty if not exposed)
    "n_prompt_tokens":   int,
    "missing_docids":    [...]
  }

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        "data/bioasq/bm25_top100/prompt engineering/1deepseek_rerank_basic_512_reasoning.py" \\
        --n-queries 5 --verbose
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
OUT_FILE = BASE / "data/bioasq/bm25_top100/prompt engineering/deepseek_reranked_512_reasoning.jsonl"

TOP_N             = 20
MODEL             = "deepseek-reasoner"
PASSAGE_TOKEN_CAP = 512

# cl100k_base is a close proxy for DeepSeek's tokenizer
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ── Prompts (identical to chat version) ───────────────────────────────────────

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
    """
    For reasoning models the system role is folded into the first user message
    to avoid endpoint restrictions. The content is otherwise identical.
    """
    n = len(docs)

    passage_blocks = []
    for i, doc in enumerate(docs, start=1):
        title   = doc["title"].strip()
        text    = doc["text"].strip()
        passage = f"{title}\n{text}" if title else text
        passage = truncate_to_tokens(passage, PASSAGE_TOKEN_CAP)
        passage_blocks.append(f"[{i}] {passage}")

    user_body = (
        PRE_PROMPT.format(n=n, query=query)
        + "\n\n"
        + "\n\n".join(passage_blocks)
        + "\n\n"
        + POST_PROMPT.format(n=n, query=query)
    )

    # Merge system prompt into the user turn so the message list is always valid
    # for both reasoning and non-reasoning endpoints.
    user_content = SYSTEM_PROMPT + "\n\n" + user_body

    messages = [{"role": "user", "content": user_content}]
    total_tokens = count_tokens(user_content)
    return messages, total_tokens


def parse_ranking(raw: str, n: int) -> list[int] | None:
    bracket_nums = re.findall(r"\[(\d+)\]", raw)
    ranking: list[int] = []
    if bracket_nums:
        for s in bracket_nums:
            try:
                ranking.append(int(s))
            except ValueError:
                pass

    if not ranking:
        match = re.search(r"\[[\d\s,]+\]", raw)
        if match:
            try:
                ranking = [int(x) for x in json.loads(match.group())]
            except (json.JSONDecodeError, TypeError, ValueError):
                return None

    if not ranking:
        return None

    seen: set[int] = set()
    clean: list[int] = []
    for x in ranking:
        if 1 <= x <= n and x not in seen:
            clean.append(x)
            seen.add(x)

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


def call_deepseek(client: OpenAI, model: str, messages: list[dict]) -> tuple[str, str]:
    """
    Returns (content, reasoning_content).
    reasoning_content is the chain-of-thought exposed by deepseek-reasoner;
    it will be an empty string for chat models or if the field is absent.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        # temperature is fixed at 1 for reasoning models; passing it is harmless
        temperature=1,
        # max_tokens must be large enough for the full thinking trace + ranking line
        max_tokens=16_000,
    )
    msg = resp.choices[0].message
    content           = msg.content or ""
    reasoning_content = getattr(msg, "reasoning_content", None) or ""
    return content, reasoning_content


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=10,
                        help="Number of queries to process (default: 10)")
    parser.add_argument("--model", default=MODEL,
                        help=f"Model name (default: {MODEL})")
    parser.add_argument("--verbose", action="store_true",
                        help="Print prompt, reasoning trace, and response for each query")
    parser.add_argument("--output", default=str(OUT_FILE),
                        help="Output JSONL path")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")

    client   = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    out_path = Path(args.output)

    records = []
    with IDS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) == args.n_queries:
                break
    print(f"Loaded {len(records)} queries from {IDS_FILE.name}")

    needed = {d["docid"] for r in records for d in r["top100"][:TOP_N]}
    print(f"Loading {len(needed)} documents from corpus …")
    corpus = load_corpus_subset(needed)
    print(f"  {len(corpus)} found  ({len(needed) - len(corpus)} missing)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = n_parse_fail = 0
    token_counts: list[int] = []

    with out_path.open("w") as out:
        for rec in tqdm(records, desc=f"Reranking [{args.model}]"):
            top20_ids = [d["docid"] for d in rec["top100"][:TOP_N]]
            missing   = [did for did in top20_ids if did not in corpus]
            docs      = [corpus.get(did, {"title": "", "text": ""}) for did in top20_ids]

            messages, n_tokens = build_messages(rec["query"], docs)
            token_counts.append(n_tokens)

            if args.verbose:
                print(f"\n{'='*70}")
                print(f"QID   : {rec['qid']}")
                print(f"Query : {rec['query']}")
                print(f"Prompt tokens : {n_tokens}  |  Missing docids: {missing}")
                print(f"\n--- USER MESSAGE (first 2000 chars) ---")
                print(messages[0]["content"][:2000])
                print(f"... [truncated, total {len(messages[0]['content'])} chars]")

            try:
                content, reasoning = call_deepseek(client, args.model, messages)
                ranking = parse_ranking(content, TOP_N)
                if ranking is None:
                    reranked_ids = top20_ids
                    n_parse_fail += 1
                    print(f"\n  [PARSE FAIL] {rec['qid']}: {content[:200]}")
                else:
                    reranked_ids = [top20_ids[i - 1] for i in ranking]
                    n_ok += 1
            except Exception as e:
                print(f"\n  [API FAIL] {rec['qid']}: {e}")
                content = reasoning = ""
                reranked_ids = top20_ids
                n_fail += 1

            if args.verbose:
                print(f"\n--- REASONING (first 1000 chars) ---")
                print((reasoning or "(none)")[:1000])
                print(f"\n--- FINAL RESPONSE ---")
                print(content)
                print(f"\n--- RERANKED ORDER ---")
                print(reranked_ids)

            out.write(json.dumps({
                "qid":               rec["qid"],
                "query":             rec["query"],
                "bm25_top20":        top20_ids,
                "reranked":          reranked_ids,
                "raw_response":      content,
                "reasoning_content": reasoning,
                "n_prompt_tokens":   n_tokens,
                "missing_docids":    missing,
            }, ensure_ascii=False) + "\n")

            time.sleep(0.5)  # reasoning model calls take longer; be conservative

    if token_counts:
        print(f"\nPrompt tokens — avg: {sum(token_counts)/len(token_counts):.0f}  "
              f"max: {max(token_counts)}")
    print(f"Done — {n_ok} ok, {n_parse_fail} parse-fail, {n_fail} api-fail → {out_path}")


if __name__ == "__main__":
    main()
