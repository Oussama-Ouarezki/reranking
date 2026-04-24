"""
Sliding-window listwise reranking with DeepSeek-Chat on BioASQ top-50 BM25 docs.

Algorithm (RankGPT-style, bottom-up):
  For N=50 docs, window=20, step=10:
    Pass 1: rerank positions [30-50]
    Pass 2: rerank positions [20-40]  (top 10 of pass 1 are in this window)
    Pass 3: rerank positions [10-30]
    Pass 4: rerank positions [0-20]   (final top-20 are locked in)
  Better documents "bubble up" through successive windows.

Each document is truncated to 512 tokens using tiktoken.
Uses deepseek-chat (temperature=0, JSON mode) for fast, deterministic output.

Reads:
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Writes:
  data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_10_v2.jsonl

Output per query:
  {
    "qid":         "...",
    "bm25_order":  ["docid1", ..., "docid50"],
    "permutation": ["docid3", "docid1", ...],   # final sliding-window order
  }

Supports resume: re-running skips queries already written to the output file.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_sliding_window_10_v2.py --n-queries 10
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_sliding_window_10_v2.py          # all 2000
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
OUT_FILE = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_10_v2.jsonl"

TOP_N       = 20
WINDOW_SIZE = 10
STEP        = 5
MAX_TOKENS  = 512
MODEL       = "deepseek-chat"

TOKENIZER = tiktoken.get_encoding("cl100k_base")

# ── System prompt — V4 (precision-first, penalise tangential hits) ─────────────
SYSTEM_PROMPT = (
    "You are an expert biomedical literature ranker. "
    "Your sole objective is to order PubMed abstracts from most to least useful "
    "for answering a specific clinical or research question.\n\n"
    "Ranking criteria — apply strictly in this order:\n\n"
    "1. SPECIFICITY (dominant criterion)\n"
    "   Rank an abstract highest when it addresses the exact entity in the question: "
    "the precise disease subtype, named drug or compound, gene/protein isoform, "
    "experimental model, or patient population. "
    "A abstract about the broad disease family or a related drug ranks BELOW one "
    "that targets the specific entity, even if the broader abstract has higher BM25.\n\n"
    "2. DIRECT ANSWERABILITY\n"
    "   Prefer abstracts that provide a self-contained answer (e.g. a mechanism, "
    "outcome, dosage, association) over those that only acknowledge the topic exists "
    "or call for further research.\n\n"
    "3. EVIDENCE LEVEL (tiebreaker only)\n"
    "   Among equally specific and answerable abstracts, prefer: "
    "RCT > cohort/case-control > case series > review > editorial/opinion. "
    "Never let evidence level override criteria 1 or 2.\n\n"
    "4. BM25 SCORE (weak prior, last resort)\n"
    "   Use BM25 only to break ties when all other criteria are equal. "
    "Override it whenever a lower-scored abstract is more specific or more answerable.\n\n"
    "Penalise abstracts that:\n"
    "  - match question keywords but study a different condition, drug, or population;\n"
    "  - are purely methodological with no domain-specific findings;\n"
    "  - are animal or in-vitro studies when the question is clearly clinical "
    "(unless no clinical evidence is present in the window).\n\n"
    "Respond ONLY with a JSON object. No explanation, no commentary.\n"
    'Format: {"ranking": [<most_relevant_id>, ..., <least_relevant_id>]}\n'
    'Example for 5 documents: {"ranking": [3, 1, 5, 2, 4]}'
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def truncate_doc(title: str, text: str, max_tokens: int = MAX_TOKENS) -> str:
    passage = f"{title}. {text}".strip() if title else text.strip()
    tokens  = TOKENIZER.encode(passage)
    if len(tokens) <= max_tokens:
        return passage
    return TOKENIZER.decode(tokens[:max_tokens]) + " [...]"


def build_user_prompt(query: str, docs: list[dict], bm25_scores: list[float]) -> str:
    n = len(docs)

    lines = [
        f"Biomedical question: {query}\n",
        "Reminder — rank by (1) specificity to the exact entity in the question, "
        "(2) direct answerability, (3) evidence level, (4) BM25 as last resort.\n",
        f"PubMed abstracts (n={n}):",
    ]

    for i, (doc, score) in enumerate(zip(docs, bm25_scores), start=1):
        passage = truncate_doc(doc["title"], doc["text"])
        lines.append(f"[{i}] BM25={score:.1f} | {passage}")

    lines.append(
        f"\nCompare abstracts against each other. "
        f"Return a JSON object with key \"ranking\" containing "
        f"all integers 1 through {n} exactly once, best first."
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
    # Strategy 1 — JSON object (primary path with JSON mode)
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

    # Strategy 3 — scrape integers from text
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


def sliding_window_rerank(
    client, model: str, query: str,
    doc_ids: list[str], bm25_scores: list[float],
    corpus: dict[str, dict],
) -> list[str]:
    """
    Bottom-up sliding window reranking.
    Returns a new ordering of doc_ids.
    """
    ranked = list(zip(doc_ids, bm25_scores))   # mutable working list
    n = len(ranked)

    # Build window start positions: from bottom, step up
    # e.g. n=50, window=20, step=10 → ends: 50, 40, 30, 20
    end = n
    while True:
        start = max(0, end - WINDOW_SIZE)
        window      = ranked[start:end]
        w_ids       = [d for d, _ in window]
        w_scores    = [s for _, s in window]
        w_docs      = [corpus.get(did, {"title": "", "text": ""}) for did in w_ids]
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(query, w_docs, w_scores)},
        ]

        try:
            raw     = call_api(client, model, messages)
            ranking = parse_ranking(raw, len(window))
            reranked_window = [(w_ids[i - 1], w_scores[i - 1]) for i in ranking]
        except Exception:
            reranked_window = window  # keep original on failure

        ranked[start:end] = reranked_window

        if start == 0:
            break
        end -= STEP
        time.sleep(0.1)

    return [d for d, _ in ranked]


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument("--model",     default=MODEL)
    parser.add_argument("--top-n",     type=int, default=TOP_N,
                        help="Number of BM25 candidates to rerank (default 50)")
    parser.add_argument("--window",    type=int, default=WINDOW_SIZE)
    parser.add_argument("--step",      type=int, default=STEP)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    n_windows = max(1, (args.top_n - args.window) // args.step + 1) if args.top_n > args.window else 1
    print(f"Config: top-{args.top_n} docs | window={args.window} | step={args.step} | "
          f"~{n_windows} windows/query | {MAX_TOKENS} tokens/doc | model={args.model}")

    # ── Load queries ───────────────────────────────────────────────────────────
    records = []
    with IDS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
    if args.n_queries:
        records = records[:args.n_queries]

    # ── Resume ─────────────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    done_qids = load_done_qids(OUT_FILE)
    if done_qids:
        print(f"Resuming: {len(done_qids)} queries already done, skipping.")
    records = [r for r in records if r["qid"] not in done_qids]
    print(f"{len(records)} queries to process")

    if not records:
        print("Nothing to do.")
        return

    # ── Load corpus ────────────────────────────────────────────────────────────
    needed = {h["docid"] for r in records for h in r["top100"][:args.top_n]}
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
            top_hits    = rec["top100"][:args.top_n]
            doc_ids     = [h["docid"] for h in top_hits]
            bm25_scores = [h["score"] for h in top_hits]

            try:
                permutation = sliding_window_rerank(
                    client, args.model, rec["query"],
                    doc_ids, bm25_scores, corpus,
                )
                n_ok += 1
            except Exception as e:
                print(f"\n  [FAIL] {rec['qid']}: {e}")
                permutation = doc_ids
                n_fail     += 1

            out.write(json.dumps({
                "qid":         rec["qid"],
                "bm25_order":  doc_ids,
                "permutation": permutation,
            }, ensure_ascii=False) + "\n")

    print(f"\nDone — {n_ok} ok, {n_fail} failed → {OUT_FILE}")


if __name__ == "__main__":
    main()