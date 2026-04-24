"""
Sliding-window listwise reranking — prompt_2 + enhanced_gold patterns.

Same pipeline as prompt_2 (top-50, window=20, step=10, 512 tok/doc) but with
a system prompt and user prompt enriched by patterns extracted from gold documents:

  1. Title is the strongest single relevance signal — check it first.
  2. Core entity co-occurrence triad: [entity] + [causal verb] + [outcome].
  3. Frequency/prevalence language directly answers "which is most common" queries.
  4. Penalise false positives: vocabulary overlap without a causal/defining link.
  5. BM25 score is a meaningful prior — demote only when criteria 1-4 clearly disagree.

Prints nDCG@1/5/10/20/50 every 100 queries (and at the very end).

Reads:
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl
  data/bioasq/processed/qrels.tsv            (for inline metric printing)

Writes:
  data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2_enhanced_gold.jsonl

Supports resume.

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_sliding_window_prompt_2_enhanced_gold.py --n-queries 1
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/deepseek_sliding_window_prompt_2_enhanced_gold.py
"""

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import tiktoken
from openai import OpenAI
from tqdm import tqdm

BASE     = Path(__file__).resolve().parents[3]
IDS_FILE = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
CORPUS   = BASE / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
QRELS    = BASE / "data/bioasq/processed/qrels.tsv"
OUT_FILE = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_prompt_2_enhanced_gold.jsonl"

TOP_N       = 50
WINDOW_SIZE = 20
STEP        = 10
MAX_TOKENS  = 512
MODEL       = "deepseek-chat"
PRINT_EVERY = 100   # print nDCG checkpoint every N queries

TOKENIZER = tiktoken.get_encoding("cl100k_base")
"""
Enhanced biomedical reranking prompts.

Design principles:
- Follows InsertRank (Seetharaman et al., 2025): BM25 scores are injected inline
  per-document as a grounding signal, documents are passed in DECREASING BM25
  order (their Table 4 ablation shows shuffling hurts), and the prompt frames
  BM25 as an anti-overthinking anchor rather than a tiebreaker rule.
- Incorporates DeepSeek's cross-query findings: title-as-primary-signal,
  entity+causal-verb+outcome triad, frequency/prevalence language, and the
  lexical-overlap-without-causal-link trap.
- Uses a reasoning-then-JSON output contract (matches the paper's Appendix
  7.4 prompt template: "First identify the essential problem. Think step by
  step. Then output JSON.").
"""

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a biomedical literature reranker. You rank PubMed abstracts by "
    "their direct informational value in answering a clinical or research "
    "question.\n\n"

    "## How to judge relevance\n"
    "Rank on these criteria, in this order of priority:\n\n"

    "1. **Title-level entity match (strongest signal).** Gold-relevant "
    "abstracts almost always name the query's key entity — the specific gene, "
    "chromosome, drug, mechanism, subtype, or population — in the title "
    "itself. A title that contains the exact answer entity is a very strong "
    "positive signal.\n\n"

    "2. **Causal / definitional link, not lexical overlap.** The abstract "
    "must connect the query entity to the query outcome through a causal or "
    "definitional verb: *drives, defines, causes, characterizes, results in, "
    "is associated with, is a hallmark of, is required for*. An abstract that "
    "mentions the same vocabulary as the query but only in passing — with no "
    "causal link between the entity and the outcome — is a FALSE POSITIVE and "
    "should be ranked low, even if it shares many words with the query.\n\n"

    "3. **Frequency / prevalence language for 'which is most common' "
    "questions.** If the query asks what is most frequent, common, typical, "
    "or characteristic, prefer abstracts that use words like *most frequent, "
    "most common, hallmark, predominant, canonical*. These phrases directly "
    "answer prevalence questions.\n\n"

    "4. **Answer completeness.** Prefer abstracts that would let a reader "
    "answer the question without needing to look elsewhere.\n\n"

    "5. **Evidence quality (tiebreaker only).** Among abstracts that are "
    "equally on-topic, prefer primary research over editorials or commentary. "
    "Never promote a weakly-relevant abstract above a strongly-relevant one "
    "on evidence grounds.\n\n"

    "## How to use BM25 scores\n"
    "Each abstract is annotated with its BM25 score from a lexical retriever. "
    "Documents are given in decreasing BM25 order. Treat BM25 as a grounding "
    "signal:\n"
    "- A high BM25 score means strong lexical overlap. This is often — but "
    "not always — a good proxy for relevance.\n"
    "- Use BM25 to avoid drifting: when your reasoning leans toward promoting "
    "a low-BM25 abstract over a high-BM25 one, that promotion requires a "
    "clear substantive reason (a stronger title match, a causal link the "
    "high-BM25 doc lacks, or frequency language the high-BM25 doc lacks). "
    "Without such a reason, keep the BM25 order.\n"
    "- A high BM25 score alone is NOT enough to rank an abstract first if it "
    "fails the causal-link test. Lexical overlap without a causal link is the "
    "classic BM25 false positive.\n\n"

    "## Output\n"
    "Respond with a single JSON object. No prose, no code fences, no "
    "explanation.\n"
    'Format: {"ranking": [<most_relevant_id>, ..., <least_relevant_id>]}\n'
    'Example for 5 docs: {"ranking": [3, 1, 5, 2, 4]}'
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def truncate_text(text: str, max_tokens: int = MAX_TOKENS) -> str:
    tokens = TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return TOKENIZER.decode(tokens[:max_tokens]) + " [...]"


def build_user_prompt(query: str, docs: list[dict], bm25_scores: list[float]) -> str:
    """
    Build the per-query user prompt.

    IMPORTANT: `docs` and `bm25_scores` must be pre-sorted in DECREASING order
    of BM25 score before being passed in. The InsertRank paper's Table 4
    ablation shows that shuffled input order degrades performance
    substantially (BRIGHT drops from .345 to .322, R2MED drops from .484 to
    .445). Decreasing-BM25 order is load-bearing.
    """
    n = len(docs)

    lines = [
        f"# Biomedical question",
        query.strip(),
        "",
        f"# PubMed abstracts (n={n}, given in decreasing BM25 order)",
        "",
    ]

    for i, (doc, score) in enumerate(zip(docs, bm25_scores), start=1):
        title = doc["title"].rstrip(".")
        abstract = truncate_text(doc["text"])
        lines.append(f"[{i}] BM25={score:.2f}")
        lines.append(f"    Title:    {title}")
        lines.append(f"    Abstract: {abstract}")
        lines.append("")

    lines.extend([
        "# Task",
        "First, identify the essential problem in the question — what "
        "specific entity (gene, drug, mechanism, chromosome, subtype, "
        "population) and what specific outcome is being asked about.",
        "",
        "Then, for each abstract, silently check:",
        "  - Does the TITLE name the query's key entity?",
        "  - Does the abstract connect that entity to the query's outcome "
        "through a causal or definitional verb (drives, defines, causes, "
        "characterizes, is a hallmark of, ...)?",
        "  - If the question is about frequency/prevalence, does the abstract "
        "use frequency language (most common, hallmark, predominant)?",
        "  - Or is the abstract only lexically similar — sharing words with "
        "the query but lacking the causal link? (If so, rank it low.)",
        "",
        "Use the BM25 ordering as your starting anchor. Only move an abstract "
        "up past a higher-BM25 abstract if it clearly wins on title match, "
        "causal link, or frequency language.",
        "",
        f"Return a JSON object with key \"ranking\" containing every integer "
        f"from 1 to {n} exactly once, ordered from most to least relevant. "
        f"No other text.",
    ])

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
    seen: set = set()
    result: list = []
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
            reranked_window = [(w_ids[i - 1], w_scores[i - 1]) for i in ranking]
        except Exception:
            reranked_window = window
        ranked[start:end] = reranked_window
        if start == 0:
            break
        end -= STEP
        time.sleep(0.1)
    return [d for d, _ in ranked]


def load_corpus_subset(docids: set) -> dict:
    corpus: dict = {}
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


def load_done_qids(path: Path) -> set:
    done: set = set()
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def load_qrels(path: Path) -> dict:
    qrels: dict = defaultdict(set)
    with path.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            if int(score) > 0:
                qrels[qid].add(did)
    return dict(qrels)


# ── Inline metrics ────────────────────────────────────────────────────────────
KS = [1, 5, 10, 20, 50]


def ndcg_at_k(ranked: list, gold: set, k: int) -> float:
    dcg  = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def print_metrics(results: dict, qrels: dict, label: str) -> None:
    qids = [q for q in results if qrels.get(q)]
    if not qids:
        return
    sums: dict = defaultdict(float)
    for qid in qids:
        ranked = results[qid]
        gold   = qrels[qid]
        for k in KS:
            sums[k] += ndcg_at_k(ranked, gold, k)
    n = len(qids)
    scores = {k: 100 * sums[k] / n for k in KS}
    print(f"\n  [{label} — {n} queries]  "
          + "  ".join(f"nDCG@{k}={scores[k]:.2f}%" for k in KS))


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
          f"~{n_windows} windows/query | {MAX_TOKENS} tok/doc | model={args.model}")

    # ── Load data ──────────────────────────────────────────────────────────────
    records = []
    with IDS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
    if args.n_queries:
        records = records[:args.n_queries]

    print("Loading qrels for inline metrics …")
    qrels = load_qrels(QRELS)

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

    needed = {h["docid"] for r in records for h in r["top100"][:args.top_n]}
    print(f"Loading {len(needed)} documents from corpus …")
    corpus = load_corpus_subset(needed)
    missing = len(needed) - len(corpus)
    if missing:
        print(f"  Warning: {missing} docids not found in corpus")
    print(f"  {len(corpus)} found\n")

    # ── Rerank ─────────────────────────────────────────────────────────────────
    n_ok = n_fail = 0
    in_memory: dict = {}   # qid → permutation, for inline metrics

    # Load already-done results into memory for accurate checkpoints
    if OUT_FILE.exists():
        with OUT_FILE.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    in_memory[r["qid"]] = r["permutation"]
                except (json.JSONDecodeError, KeyError):
                    pass

    with OUT_FILE.open("a") as out:
        for i, rec in enumerate(tqdm(records, desc="Reranking"), start=1):
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
                n_fail += 1

            out.write(json.dumps({
                "qid":         rec["qid"],
                "bm25_order":  doc_ids,
                "permutation": permutation,
            }, ensure_ascii=False) + "\n")
            out.flush()

            in_memory[rec["qid"]] = permutation

            if i % PRINT_EVERY == 0:
                print_metrics(in_memory, qrels, f"checkpoint @{len(in_memory)}")

    print(f"\nDone — {n_ok} ok, {n_fail} failed → {OUT_FILE}")
    print_metrics(in_memory, qrels, "FINAL")


if __name__ == "__main__":
    main()
