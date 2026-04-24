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
  data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt.jsonl

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
        data/bioasq/bm25_top100/deepseek_sliding_window.py --n-queries 10
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_sliding_window.py          # all 2000
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
OUT_FILE = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_3.jsonl"

TOP_N       = 50
WINDOW_SIZE = 20
STEP        = 10
MAX_TOKENS  = 512
MODEL       = "deepseek-chat"

TOKENIZER = tiktoken.get_encoding("cl100k_base")
# ── System prompt — BioASQ-tuned V4 ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a biomedical literature reranker specializing in PubMed abstracts for "
    "expert biomedical question answering (BioASQ task). Your goal is to rank "
    "candidate abstracts so that the single most relevant document appears first, "
    "maximising Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR).\n\n"

    "BioASQ questions come in four types — adapt your relevance judgment accordingly:\n"
    "- Yes/No: rank abstracts that contain direct evidence supporting OR refuting the "
    "claim above those that only discuss related topics.\n"
    "- Factoid: rank abstracts that name the specific entity (gene, drug, disease, "
    "protein, number) that answers the question above broader background abstracts.\n"
    "- List: rank abstracts that enumerate or collectively cover the full set of "
    "entities asked for; prefer abstracts that together maximise coverage.\n"
    "- Summary: rank abstracts that provide the most comprehensive mechanistic or "
    "clinical explanation of the topic; breadth and depth both matter.\n\n"

    "Apply these criteria in strict priority order:\n"
    "  1. Direct topical match (primary): exact match on condition, mechanism, drug, "
    "gene, or population named in the question. This always outranks other criteria.\n"
    "  2. Answer completeness: does the abstract directly answer the question without "
    "requiring the reader to consult further sources?\n"
    "  3. Evidence quality (tiebreaker only): prefer primary research, RCTs, and "
    "systematic reviews over editorials — but never promote a high-evidence abstract "
    "above a more topically relevant one.\n"
    "  4. BM25 score (final tiebreaker): use only when all semantic criteria are equal.\n\n"

    "Important:\n"
    "- Do NOT let the order abstracts appear in the input influence your ranking. "
    "Evaluate each abstract on its own merit.\n"
    "- Do NOT rank an abstract highly because it shares vocabulary with the question. "
    "Lexical overlap without informational depth is not relevance.\n"
    "- Compare abstracts against each other, not just against the question.\n\n"

    "Respond ONLY with JSON. No explanation, no preamble.\n"
    'Format: {"ranking": [<most_relevant_id>, ..., <least_relevant_id>]}\n'
    'Example for 5 documents: {"ranking": [3, 1, 5, 2, 4]}'
)


# ── Question type detector ────────────────────────────────────────────────────
def detect_question_type(query: str) -> str:
    """Heuristic BioASQ question type detector."""
    q = query.lower().strip()
    yes_no_signals = [
        "is ", "are ", "does ", "do ", "has ", "have ", "can ", "could ",
        "was ", "were ", "will ", "would ", "should ", "did "
    ]
    list_signals = [
        "which ", "what are ", "list ", "name ", "enumerate ", "what genes",
        "what proteins", "what drugs", "what diseases", "what factors",
        "what types", "what mutations", "what biomarkers"
    ]
    if any(q.startswith(s) for s in yes_no_signals):
        return "yes_no"
    if any(s in q for s in list_signals):
        return "list"
    if q.startswith("what is ") or q.startswith("what was ") or q.startswith("who "):
        return "factoid"
    return "summary"


_TYPE_HINTS = {
    "yes_no":  (
        "This is a YES/NO question. Prioritise abstracts that contain direct "
        "experimental or clinical evidence that confirms or refutes the claim. "
        "An abstract with a clear result statement outranks one that merely discusses the topic."
    ),
    "factoid": (
        "This is a FACTOID question. Prioritise abstracts that explicitly name "
        "the specific entity (gene, drug, disease, number) that is the answer. "
        "Precision matters more than breadth."
    ),
    "list":    (
        "This is a LIST question. Prioritise abstracts that enumerate or together "
        "cover the full set of entities asked for. An abstract naming multiple relevant "
        "entities outranks one naming only a single entity."
    ),
    "summary": (
        "This is a SUMMARY question. Prioritise abstracts that provide the most "
        "mechanistic or clinical depth on the topic. Breadth and explanatory detail "
        "both matter; review articles and well-designed studies rank above brief reports."
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def truncate_doc(title: str, text: str, max_tokens: int = MAX_TOKENS) -> str:
    passage = f"{title}. {text}".strip() if title else text.strip()
    tokens  = TOKENIZER.encode(passage)
    if len(tokens) <= max_tokens:
        return passage
    return TOKENIZER.decode(tokens[:max_tokens]) + " [...]"


def build_user_prompt(query: str, docs: list[dict], bm25_scores: list[float]) -> str:
    n        = len(docs)
    qtype    = detect_question_type(query)
    type_hint = _TYPE_HINTS[qtype]

    lines = [
        f"Biomedical question ({qtype.replace('_', '/')} type): {query}\n",
        f"Question-type guidance: {type_hint}\n",
        "Ranking criteria (apply in strict priority order):",
        "  1. Direct topical match — exact match on the condition, mechanism, drug, gene, or population in the question.",
        "  2. Answer completeness — does this abstract directly answer the question?",
        "  3. Evidence quality — tiebreaker only; never overrides topical match or completeness.",
        "  4. BM25 score — final tiebreaker only.\n",
        f"PubMed abstracts to rank (n={n}). Do NOT let their order below influence your ranking:",
    ]

    for i, (doc, score) in enumerate(zip(docs, bm25_scores), start=1):
        passage = truncate_doc(doc["title"], doc["text"])
        lines.append(f"[{i}] BM25={score:.1f} | {passage}")

    lines.append(
        f"\nRank all {n} abstracts. Compare them against each other. The abstract most "
        f"likely to directly answer the question must be ranked first. Return a JSON "
        f'object with key \"ranking\" containing all integers 1 through {n} exactly once, best first.'
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
