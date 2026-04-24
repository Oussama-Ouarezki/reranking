"""
Pass top-20 BM25 documents per query to a local Ollama model for listwise ranking.

Reads:
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Writes:
  data/bioasq/bm25_top100/ollama_reranked.jsonl

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/ollama_rerank.py --n-queries 10
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/ollama_rerank.py --n-queries 10 --model llama3.1:8b
"""

import argparse
import json
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

BASE      = Path(__file__).resolve().parents[3]
IDS_FILE  = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
CORPUS    = BASE / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
OUT_FILE  = BASE / "data/bioasq/bm25_top100/ollama_reranked.jsonl"

TOP_N     = 20
DEFAULT_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = (
    "You are a biomedical information retrieval expert. "
    "Your task is to rank a list of documents by their relevance to a given query. "
    "Return ONLY a JSON array of document numbers in order from most to least relevant. "
    "Example output: [3, 1, 7, 2, 5, ...]  — nothing else."
)


def build_user_prompt(query: str, docs: list[dict]) -> str:
    lines = [f"Query: {query}\n"]
    for i, doc in enumerate(docs, start=1):
        title   = doc["title"].strip()
        text    = doc["text"].strip()
        passage = f"{title}. {text}" if title else text
        lines.append(f"[{i}] {passage}")
    lines.append(
        "\nRank these documents from most to least relevant to the query. "
        "Return ONLY a JSON array of the document numbers, e.g. [3,1,5,...]"
    )
    return "\n".join(lines)


def parse_ranking(raw: str, n: int) -> list[int] | None:
    match = re.search(r"\[[\d\s,]+\]", raw)
    if not match:
        return None
    try:
        ranking = json.loads(match.group())
        seen: set[int] = set()
        clean = []
        for x in ranking:
            if isinstance(x, int) and 1 <= x <= n and x not in seen:
                clean.append(x)
                seen.add(x)
        for i in range(1, n + 1):
            if i not in seen:
                clean.append(i)
        return clean
    except (json.JSONDecodeError, TypeError):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--model",     default=DEFAULT_MODEL)
    args = parser.parse_args()

    client = OpenAI(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
    )

    # ── Load queries ───────────────────────────────────────────────────────────
    records = []
    with IDS_FILE.open() as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) == args.n_queries:
                break
    print(f"Loaded {len(records)} queries  |  model: {args.model}")

    # ── Load passage texts ─────────────────────────────────────────────────────
    needed = {h["docid"] for r in records for h in r["top100"][:TOP_N]}
    print(f"Loading {len(needed)} documents from corpus …")
    corpus = load_corpus_subset(needed)
    print(f"  {len(corpus)} found")

    # ── Rerank ─────────────────────────────────────────────────────────────────
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0

    with OUT_FILE.open("w") as out:
        for rec in tqdm(records, desc="Reranking"):
            top20_ids = [h["docid"] for h in rec["top100"][:TOP_N]]
            docs      = [corpus.get(did, {"title": "", "text": ""}) for did in top20_ids]
            prompt    = build_user_prompt(rec["query"], docs)

            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=256,
                )
                raw     = resp.choices[0].message.content or ""
                ranking = parse_ranking(raw, TOP_N)
                reranked_ids = [top20_ids[i - 1] for i in ranking] if ranking else top20_ids
                n_ok += 1
            except Exception as e:
                print(f"\n  [WARN] {rec['qid']}: {e}")
                raw          = ""
                reranked_ids = top20_ids
                n_fail      += 1

            out.write(json.dumps({
                "qid":          rec["qid"],
                "query":        rec["query"],
                "bm25_top20":   top20_ids,
                "reranked":     reranked_ids,
                "raw_response": raw,
            }, ensure_ascii=False) + "\n")

    print(f"\nDone — {n_ok} ok, {n_fail} failed → {OUT_FILE}")


if __name__ == "__main__":
    main()
