"""DeepSeek zero-shot listwise reranking on the BioASQ application set.

Runs offline (slow). Outputs a run JSON in the same schema the FastAPI app
uses for its native rerankers, so the Dashboard / Generation pages pick it
up automatically as another model named ``deepseek``.

Usage
-----
    /home/oussama/miniconda3/envs/pyml/bin/python \
        scripts/rerank_deepseek_zeroshot.py [--model deepseek-r1:7b] [--n 20]

Requirements
------------
- Ollama running locally with the picked DeepSeek tag pulled.
- The same Lucene index + queries + qrels used by the application backend.

Output
------
    application/cache/runs/deepseek/<UTC_TS>.json
"""

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from application.backend import config, deps
from application.backend.evaluation.ranking import per_query_metrics, aggregate_metrics
from application.backend.generation import ollama_client


WINDOW_SIZE = 20
STEP = 10
SAVE_TOPN = 20
PASSAGE_CHARS = 800
PERM_RE = re.compile(r"\[(\d+)\]")
RUNS_DIR = config.CACHE_DIR / "runs" / "deepseek"

SYSTEM_PROMPT = (
    "You are RankLLM, an intelligent assistant that ranks biomedical "
    "PubMed-style passages by relevance to a user's question."
)


def _user_prompt(query: str, passages: list[str]) -> str:
    n = len(passages)
    body = "\n".join(
        f"[{i + 1}] {p[:PASSAGE_CHARS]}" for i, p in enumerate(passages)
    )
    return (
        f"I will provide you with {n} passages, each indicated by number "
        f"identifier []. Rank the passages based on their relevance to the "
        f"search query: {query}.\n\n"
        f"{body}\n\n"
        f"Search Query: {query}.\n"
        f"Rank the {n} passages above based on their relevance to the search "
        f"query. The passages should be listed in descending order using "
        f"identifiers, the most relevant first. Output format: [] > [] > [] "
        f"(e.g. [2] > [3] > [1]). Only respond with the ranking, no other text."
    )


def _parse_permutation(text: str, n: int) -> list[int]:
    seen: list[int] = []
    used: set[int] = set()
    for m in PERM_RE.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n and idx not in used:
            seen.append(idx)
            used.add(idx)
    for i in range(n):
        if i not in used:
            seen.append(i)
    return seen


def _rank_window(model: str, query: str, passages: list[str]) -> list[int]:
    text = ollama_client.chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(query, passages)},
        ],
        model=model,
        temperature=0.0,
        max_tokens=300,
        num_ctx=8192,
    )
    return _parse_permutation(text, len(passages))


def _rerank_listwise(model: str, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    if not candidates:
        return []
    order = list(range(len(candidates)))
    texts = [c[1] for c in candidates]
    end = len(order)
    while end > 0:
        start = max(0, end - WINDOW_SIZE)
        window_idxs = order[start:end]
        window_texts = [texts[i] for i in window_idxs]
        try:
            perm = _rank_window(model, query, window_texts)
        except Exception:
            traceback.print_exc()
            perm = list(range(len(window_idxs)))
        order[start:end] = [window_idxs[p] for p in perm]
        if start == 0:
            break
        end -= STEP

    n = len(order)
    return [(candidates[i][0], float(n - rank)) for rank, i in enumerate(order)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-r1:7b", help="Ollama tag to call")
    ap.add_argument("--n", type=int, default=0, help="Limit to first N queries (0 = all)")
    args = ap.parse_args()

    bm25 = deps.get_bm25()
    corpus = deps.get_corpus()
    qrels = deps.get_qrels()
    queries = deps.get_queries()
    if args.n:
        queries = queries[: args.n]

    started_at = time.time()
    full_run: list[tuple[str, str, float]] = []
    per_query_model: dict[str, dict] = {}

    for i, q in enumerate(queries, start=1):
        qid = q["_id"]
        qtext = q["text"]
        qtype = q.get("type")

        hits = bm25.search(qtext, k=config.BM25_RETRIEVE_K)
        cands: list[tuple[str, str]] = []
        for h in hits:
            docid = str(h["docid"])
            text = corpus.get_text(docid)
            if text:
                cands.append((docid, text))

        ranked = _rerank_listwise(args.model, qtext, cands)
        for docid, score in ranked:
            full_run.append((qid, str(docid), float(score)))

        metrics = per_query_metrics(qid, ranked, qrels) if qid in qrels else None
        top_docids = [str(d) for d, _ in ranked[:SAVE_TOPN]]
        entry: dict = {"top_docids": top_docids}
        if metrics is not None:
            entry["metrics"] = metrics
        if qtype is not None:
            entry["qtype"] = qtype
        per_query_model[qid] = entry

        if i % 5 == 0 or i == len(queries):
            print(f"[{i}/{len(queries)}] {qid}", flush=True)

    agg = aggregate_metrics(full_run, qrels)
    ended_at = time.time()
    elapsed = round(ended_at - started_at, 1)

    ts = datetime.fromtimestamp(ended_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"deepseek_{ts}"
    payload = {
        "model": "deepseek",
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_s": elapsed,
        "config": {
            "save_topn": SAVE_TOPN,
            "bm25_retrieve_k": config.BM25_RETRIEVE_K,
            "deepseek_model": args.model,
            "window_size": WINDOW_SIZE,
            "step": STEP,
            "n_questions": len(queries),
        },
        "comment": f"DeepSeek zero-shot ({args.model}) listwise sliding window",
        "aggregate": agg,
        "per_query": per_query_model,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"{ts}.json"
    with open(out, "w") as f:
        json.dump(payload, f)

    print(f"\nSaved {out}")
    print(f"run_id: {run_id}")
    print(f"aggregate: {json.dumps(agg, indent=2)}")


if __name__ == "__main__":
    main()
