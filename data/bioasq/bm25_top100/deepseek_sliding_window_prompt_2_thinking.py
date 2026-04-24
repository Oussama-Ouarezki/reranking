"""
Sliding-window listwise reranking with DeepSeek-Chat + enable_thinking on BioASQ top-50.

Same config as prompt_2 (top_n=50, window=20, step=10, 512 tok, title trick) but
with extra_body={"enable_thinking": True} so the model reasons before ranking.

The reasoning trace lands in response.choices[0].message.reasoning (logged but not used).
The final JSON ranking is read from response.choices[0].message.content as usual.

Reads:
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Writes:
  data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2_thinking.jsonl

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_sliding_window_prompt_2_thinking.py --n-queries 1
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
OUT_FILE = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2_thinking.jsonl"

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


def build_user_prompt(query: str, docs: list[dict], bm25_scores: list[float]) -> str:
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


def _coerce_int(x) -> int | None:
    if isinstance(x, int): return x
    if isinstance(x, float) and x == int(x): return int(x)
    if isinstance(x, str) and x.strip().isdigit(): return int(x.strip())
    return None


def _clean(seq: list, n: int) -> list[int]:
    seen: set[int] = set()
    result: list[int] = []
    for raw_x in seq:
        x = _coerce_int(raw_x)
        if x is not None and 1 <= x <= n and x not in seen:
            result.append(x); seen.add(x)
    for i in range(1, n + 1):
        if i not in seen: result.append(i)
    return result


def parse_ranking(raw: str, n: int) -> list[int]:
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            for key in ("ranking", "ranked", "order", "result"):
                if key in obj and isinstance(obj[key], list):
                    return _clean(obj[key], n)
            for val in obj.values():
                if isinstance(val, list): return _clean(val, n)
        if isinstance(obj, list): return _clean(obj, n)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    for m in reversed(list(re.finditer(r'\[[\d\s.,\"\']+\]', raw))):
        try:
            arr = json.loads(m.group())
            if isinstance(arr, list) and len(arr) >= n // 2: return _clean(arr, n)
        except (json.JSONDecodeError, TypeError):
            pass
    nums = [int(x) for x in re.findall(r'\b([1-9][0-9]?)\b', raw) if int(x) <= n]
    if nums: return _clean(nums, n)
    return list(range(1, n + 1))


def call_api(client, model: str, messages: list, log_thinking: bool = False) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=4096,   # extra room for reasoning trace
        extra_body={"enable_thinking": True},
    )
    msg = resp.choices[0].message
    if log_thinking and getattr(msg, "reasoning", None):
        preview = (msg.reasoning or "")[:300].replace("\n", " ")
        print(f"\n  [thinking] {preview} …")
    return msg.content or ""


def sliding_window_rerank(
    client, model: str, query: str,
    doc_ids: list[str], bm25_scores: list[float],
    corpus: dict[str, dict],
    log_thinking: bool = False,
) -> list[str]:
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
            raw     = call_api(client, model, messages, log_thinking=log_thinking)
            ranking = parse_ranking(raw, len(window))
            reranked_window = [(w_ids[i - 1], w_scores[i - 1]) for i in ranking]
        except Exception:
            reranked_window = window
        ranked[start:end] = reranked_window
        if start == 0:
            break
        end -= STEP
        time.sleep(0.5)
    return [d for d, _ in ranked]


def load_corpus_subset(docids: set[str]) -> dict[str, dict]:
    corpus: dict[str, dict] = {}
    with CORPUS.open() as f:
        for line in f:
            doc = json.loads(line)
            if doc["_id"] in docids:
                corpus[doc["_id"]] = {"title": doc.get("title", ""), "text": doc["text"]}
            if len(corpus) == len(docids):
                break
    return corpus


def load_done_qids(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                try: done.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError): pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries",    type=int, default=None)
    parser.add_argument("--model",        default=MODEL)
    parser.add_argument("--log-thinking", action="store_true",
                        help="Print a preview of the reasoning trace for each window")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    print(f"Config: top-{TOP_N} docs | window={WINDOW_SIZE} | step={STEP} | "
          f"max_tokens={MAX_TOKENS} | model={args.model} | thinking=ON")

    records = []
    with IDS_FILE.open() as f:
        for line in f: records.append(json.loads(line))
    if args.n_queries:
        records = records[:args.n_queries]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_qids = load_done_qids(OUT_FILE)
    if done_qids:
        print(f"Resuming: {len(done_qids)} queries already done, skipping.")
    records = [r for r in records if r["qid"] not in done_qids]
    print(f"{len(records)} queries to process")
    if not records:
        print("Nothing to do."); return

    needed = {h["docid"] for r in records for h in r["top100"][:TOP_N]}
    print(f"Loading {len(needed)} documents from corpus …")
    corpus = load_corpus_subset(needed)
    print(f"  {len(corpus)} found")

    n_ok = n_fail = 0
    with OUT_FILE.open("a") as out:
        for rec in tqdm(records, desc="Reranking"):
            top_hits    = rec["top100"][:TOP_N]
            doc_ids     = [h["docid"] for h in top_hits]
            bm25_scores = [h["score"] for h in top_hits]
            try:
                permutation = sliding_window_rerank(
                    client, args.model, rec["query"], doc_ids, bm25_scores, corpus,
                    log_thinking=args.log_thinking,
                )
                n_ok += 1
            except Exception as e:
                print(f"\n  [FAIL] {rec['qid']}: {e}")
                permutation = doc_ids; n_fail += 1
            out.write(json.dumps({
                "qid": rec["qid"], "bm25_order": doc_ids, "permutation": permutation,
            }, ensure_ascii=False) + "\n")

    print(f"\nDone — {n_ok} ok, {n_fail} failed → {OUT_FILE}")


if __name__ == "__main__":
    main()
