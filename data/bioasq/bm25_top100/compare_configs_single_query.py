"""
Single-query ablation: compare 6 sliding-window reranking configs vs BM25 baseline.

Configurations (all use top-50 BM25 candidates, window=20, step=10):
  Prompt V2 × {150, 350, 512} tokens/doc
  Prompt V3 × {150, 350, 512} tokens/doc

Metrics: NDCG@10, NDCG@20, NDCG@50, MAP@20, Recall@10, Recall@20, Recall@50

Prompt V2 — balanced criteria (topical match, evidence quality, completeness equal weight)
Prompt V3 — topical match primary, evidence quality demoted to tiebreaker

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/compare_configs_single_query.py --qid 55031181e9bde69634000014
"""

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from collections import defaultdict

import tiktoken
from openai import OpenAI

BASE     = Path(__file__).resolve().parents[3]
IDS_FILE = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
CORPUS   = BASE / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
QRELS    = BASE / "data/bioasq/processed/qrels.tsv"

TOP_N       = 50
WINDOW_SIZE = 20
STEP        = 10
MODEL       = "deepseek-chat"
TOKEN_SIZES = [150, 350, 512]

TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_V2 = (
    "You are a biomedical literature reranker. Your task is to rank PubMed abstracts "
    "by their in-depth relevance and direct informational value to a given clinical or "
    "research question.\n\n"
    "For each abstract, assess:\n"
    "- Direct topical match: does it address the exact condition, mechanism, drug, gene, "
    "or population in the question?\n"
    "- Accuracy and evidence quality: does it report original data, validated findings, "
    "or systematic analysis?\n"
    "- Answer completeness: would this abstract help a clinician or researcher answer the "
    "question without needing further lookup?\n\n"
    "Rank abstracts from most to least relevant by comparing them against each other. "
    "Use the BM25 score as a weak prior — override it when an abstract with a lower BM25 "
    "score provides more precise or complete information about the question.\n\n"
    "Avoid ranking an abstract highly solely because it shares vocabulary with the "
    "question. Relevance is determined by informational depth and accuracy, not lexical "
    "overlap.\n\n"
    "Respond ONLY with JSON. No explanation.\n"
    'Format: {"ranking": [<most_relevant_id>, ..., <least_relevant_id>]}\n'
    'Example for 5 documents: {"ranking": [3, 1, 5, 2, 4]}'
)

SYSTEM_PROMPT_V3 = (
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
    "Use the BM25 score as a weak prior — override it when an abstract with a lower BM25 "
    "score provides more precise or complete information about the question.\n\n"
    "Avoid ranking an abstract highly solely because it shares vocabulary with the "
    "question. Relevance is determined by informational depth and accuracy, not lexical "
    "overlap.\n\n"
    "Respond ONLY with JSON. No explanation.\n"
    'Format: {"ranking": [<most_relevant_id>, ..., <least_relevant_id>]}\n'
    'Example for 5 documents: {"ranking": [3, 1, 5, 2, 4]}'
)

USER_CRITERIA_V2 = [
    "  1. Direct topical match to the question's condition, mechanism, drug, gene, or population.",
    "  2. Evidence quality — prefer primary research, RCTs, and systematic reviews over editorials.",
    "  3. Answer completeness — abstracts that directly answer the question rank above tangentially related ones.",
    "  4. BM25 score — use as a weak tiebreaker only when semantic relevance is equal.",
]

USER_CRITERIA_V3 = [
    "  1. Direct topical match to the question's condition, mechanism, drug, gene, or population.",
    "  2. Answer completeness — abstracts that directly answer the question rank above tangentially related ones.",
    "  3. Evidence quality (tiebreaker only) — never overrides topical match or completeness.",
    "  4. BM25 score — use as a weak tiebreaker only when semantic relevance is equal.",
]

PROMPTS = {
    "V2": (SYSTEM_PROMPT_V2, USER_CRITERIA_V2),
    "V3": (SYSTEM_PROMPT_V3, USER_CRITERIA_V3),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def truncate_doc(title: str, text: str, max_tokens: int) -> str:
    passage = f"{title}. {text}".strip() if title else text.strip()
    tokens  = TOKENIZER.encode(passage)
    if len(tokens) <= max_tokens:
        return passage
    return TOKENIZER.decode(tokens[:max_tokens]) + " [...]"


def build_user_prompt(
    query: str,
    docs: list[dict],
    bm25_scores: list[float],
    criteria: list[str],
    max_tokens: int,
) -> str:
    n = len(docs)
    lines = [
        f"Biomedical question: {query}\n",
        "Ranking criteria (apply in this order):",
        *criteria,
        "",
        f"PubMed abstracts (n={n}):",
    ]
    for i, (doc, score) in enumerate(zip(docs, bm25_scores), start=1):
        passage = truncate_doc(doc["title"], doc["text"], max_tokens)
        lines.append(f"[{i}] BM25={score:.1f} | {passage}")
    lines.append(
        f"\nApply the criteria above. Compare abstracts against each other, not just "
        f"against the question. Return a JSON object with key \"ranking\" containing "
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


def call_api(client, messages: list) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=128,
    )
    return resp.choices[0].message.content or ""


def sliding_window_rerank(
    client,
    query: str,
    doc_ids: list[str],
    bm25_scores: list[float],
    corpus: dict[str, dict],
    system_prompt: str,
    criteria: list[str],
    max_tokens: int,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
) -> list[str]:
    ranked = list(zip(doc_ids, bm25_scores))
    n = len(ranked)
    end = n
    while True:
        start    = max(0, end - window_size)
        window   = ranked[start:end]
        w_ids    = [d for d, _ in window]
        w_scores = [s for _, s in window]
        w_docs   = [corpus.get(did, {"title": "", "text": ""}) for did in w_ids]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": build_user_prompt(
                query, w_docs, w_scores, criteria, max_tokens
            )},
        ]
        try:
            raw     = call_api(client, messages)
            ranking = parse_ranking(raw, len(window))
            ranked[start:end] = [(w_ids[i - 1], w_scores[i - 1]) for i in ranking]
        except Exception:
            pass  # keep original window order on failure
        if start == 0:
            break
        end -= step
        time.sleep(0.1)
    return [d for d, _ in ranked]


# ── Metrics ────────────────────────────────────────────────────────────────────

def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg   = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in gold)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def map_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    hits = 0
    score = 0.0
    for i, d in enumerate(ranked[:k]):
        if d in gold:
            hits += 1
            score += hits / (i + 1)
    return score / len(gold) if gold else 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in gold) / len(gold) if gold else 0.0


METRIC_FNS: list[tuple] = [
    ("NDCG@10",   lambda r, g: ndcg_at_k(r, g, 10)),
    ("NDCG@20",   lambda r, g: ndcg_at_k(r, g, 20)),
    ("NDCG@50",   lambda r, g: ndcg_at_k(r, g, 50)),
    ("MAP@20",    lambda r, g: map_at_k(r, g, 20)),
    ("Recall@10", lambda r, g: recall_at_k(r, g, 10)),
    ("Recall@20", lambda r, g: recall_at_k(r, g, 20)),
    ("Recall@50", lambda r, g: recall_at_k(r, g, 50)),
]


def compute_metrics(ranked: list[str], gold: set[str]) -> dict[str, float]:
    return {name: fn(ranked, gold) * 100 for name, fn in METRIC_FNS}


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
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
    return qrels


def find_query_record(qid: str) -> dict | None:
    with IDS_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            if r["qid"] == qid:
                return r
    return None


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


# ── Table printer ──────────────────────────────────────────────────────────────

def build_table_lines(
    query: str,
    gold: set[str],
    col_order: list[str],
    scores: dict[str, dict[str, float]],
    top_n: int,
    window_size: int,
    step: int,
) -> list[str]:
    metric_names = [name for name, _ in METRIC_FNS]
    metric_w = 12
    col_w    = 11

    total_w = metric_w + 2 + len(col_order) * (col_w + 2)
    sep = "─" * total_w

    lines = [
        f"",
        f"Query  : {query}",
        f"Gold   : {len(gold)} relevant doc(s)  |  top-{top_n} BM25 candidates evaluated",
        f"Window : {window_size}, step={step}",
        f"",
        f"  {sep}",
        f"  {'Metric':<{metric_w}}" + "".join(f"  {c:>{col_w}}" for c in col_order),
        f"  {sep}",
    ]

    for m in metric_names:
        bm25_val = scores["BM25"][m]
        row = f"  {m:<{metric_w}}"
        for col in col_order:
            val  = scores[col][m]
            mark = "*" if col != "BM25" and val > bm25_val else " "
            row += f"  {val:>{col_w - 1}.2f}{mark}"
        lines.append(row)

    lines.append(f"  {sep}")
    lines.append(f"")
    lines.append(f"  * = beats BM25 baseline")

    rerank_cols = [c for c in col_order if c != "BM25"]
    wins = {c: sum(1 for m in metric_names if scores[c][m] > scores["BM25"][m])
            for c in rerank_cols}
    best = max(wins, key=wins.get)
    lines.append(f"")
    lines.append(f"  Best config: {best} ({wins[best]}/{len(metric_names)} metrics beat BM25)")
    lines.append(f"")
    return lines


def save_tsv(
    qid: str,
    col_order: list[str],
    scores: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    metric_names = [name for name, _ in METRIC_FNS]
    with out_path.open("w") as f:
        f.write("metric\t" + "\t".join(col_order) + "\n")
        for m in metric_names:
            row = m + "\t" + "\t".join(f"{scores[col][m]:.4f}" for col in col_order)
            f.write(row + "\n")


def print_table(
    query: str,
    gold: set[str],
    col_order: list[str],
    scores: dict[str, dict[str, float]],
    top_n: int,
    window_size: int = WINDOW_SIZE,
    step: int = STEP,
    out_dir: Path | None = None,
    qid: str = "",
) -> None:
    lines = build_table_lines(query, gold, col_order, scores, top_n, window_size, step)
    for line in lines:
        print(line)

    if out_dir is not None and qid:
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / f"compare_configs_{qid}.txt"
        tsv_path = out_dir / f"compare_configs_{qid}.tsv"

        txt_path.write_text("\n".join(lines) + "\n")
        print(f"\n  Saved table  → {txt_path}")

        save_tsv(qid, col_order, scores, tsv_path)
        print(f"  Saved scores → {tsv_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 6 reranking configs on a single BioASQ query."
    )
    parser.add_argument("--qid",   required=True, help="Query ID from bm25_top100_ids.jsonl")
    parser.add_argument("--top-n", type=int, default=TOP_N,
                        help=f"Number of BM25 candidates to rerank (default {TOP_N})")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--step",   type=int, default=STEP)
    args = parser.parse_args()

    window_size = args.window
    step        = args.step

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("Loading qrels …")
    qrels = load_qrels(QRELS)
    gold  = qrels.get(args.qid, set())
    if not gold:
        raise SystemExit(f"No gold labels found for qid={args.qid}")
    print(f"  Gold docs: {len(gold)}")

    print(f"Finding query {args.qid} …")
    rec = find_query_record(args.qid)
    if not rec:
        raise SystemExit(f"Query {args.qid} not found in {IDS_FILE}")

    query       = rec["query"]
    top_hits    = rec["top100"][:args.top_n]
    doc_ids     = [h["docid"] for h in top_hits]
    bm25_scores = [h["score"] for h in top_hits]
    print(f"  Query: {query}")

    print("Loading corpus …")
    corpus = load_corpus_subset(set(doc_ids))
    missing = len(doc_ids) - len(corpus)
    if missing:
        print(f"  Warning: {missing} docids not found in corpus (will appear as empty)")
    print(f"  {len(corpus)}/{len(doc_ids)} documents found")

    # ── Run all 6 configs ──────────────────────────────────────────────────────

    configs = (
        [(f"V2-{t}tok", "V2", t) for t in TOKEN_SIZES] +
        [(f"V3-{t}tok", "V3", t) for t in TOKEN_SIZES]
    )

    results: dict[str, list[str]] = {"BM25": doc_ids}

    n_windows = max(1, (args.top_n - window_size) // step + 1) if args.top_n > window_size else 1
    total_calls = len(configs) * n_windows
    print(f"\n{len(configs)} configs × ~{n_windows} windows = ~{total_calls} API calls\n")

    for label, prompt_ver, max_tok in configs:
        sys_prompt, criteria = PROMPTS[prompt_ver]
        print(f"  [{label}] prompt={prompt_ver}, max_tokens={max_tok} …", end="", flush=True)
        try:
            perm = sliding_window_rerank(
                client, query, doc_ids, bm25_scores, corpus,
                sys_prompt, criteria, max_tok,
                window_size, step,
            )
            results[label] = perm
            print(" done")
        except Exception as e:
            print(f" FAILED: {e}")
            results[label] = doc_ids  # fall back to BM25 order

    # ── Compute metrics & display ──────────────────────────────────────────────

    col_order = ["BM25"] + [label for label, _, _ in configs]
    scores    = {col: compute_metrics(results[col], gold) for col in col_order}

    out_dir = BASE / "data/bioasq/bm25_top100/results"
    print_table(query, gold, col_order, scores, args.top_n, window_size, step,
                out_dir=out_dir, qid=args.qid)


if __name__ == "__main__":
    main()
