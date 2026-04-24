"""
Pattern analysis: show DeepSeek each query + its gold documents and ask
what linguistic/semantic patterns connect them. Helps design better prompts.

Reads:
  data/bioasq/processed/qrels.tsv
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl   (query text)
  data/bioasq/pubmed_full/full/corpus_full_processed.jsonl

Writes:
  data/bioasq/bm25_top100/query_gold_patterns.txt

Usage:
    export DEEPSEEK_API_KEY="sk-..."
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/analyze_query_gold_patterns.py --n-queries 10
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

BASE       = Path(__file__).resolve().parents[3]
QRELS      = BASE / "data/bioasq/processed/qrels.tsv"
IDS_FILE   = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
CORPUS     = BASE / "data/bioasq/pubmed_full/full/corpus_full_processed.jsonl"
OUT_FILE   = BASE / "data/bioasq/bm25_top100/query_gold_patterns.txt"

MODEL = "deepseek-chat"

SYSTEM_PROMPT = (
    "You are an expert in biomedical information retrieval. "
    "Your task is to analyze what makes PubMed abstracts relevant to a given biomedical question.\n\n"
    "Look carefully at the query and ALL the gold (known-relevant) abstracts together. "
    "Identify concrete, generalizable patterns — not just for this query, but patterns that "
    "could help a reranker distinguish relevant from irrelevant documents across many queries.\n\n"
    "Focus on:\n"
    "- Lexical patterns: shared terms, synonyms, abbreviations\n"
    "- Semantic patterns: concepts, mechanisms, entities (drugs, genes, diseases, populations)\n"
    "- Structural patterns: where in the abstract the relevant signal appears (title, first sentence, methods, conclusion)\n"
    "- What irrelevant-looking documents might share with the query but lack (false positive traps)\n\n"
    "Be specific and concise. Give actionable patterns, not generic observations."
)


def build_prompt(query: str, gold_docs: list) -> str:
    lines = [
        f"Biomedical question: {query}\n",
        f"Known-relevant PubMed abstracts ({len(gold_docs)} total):\n",
    ]
    for i, doc in enumerate(gold_docs, start=1):
        title = doc.get("title", "").strip()
        text  = doc.get("text",  "").strip()
        lines.append(f"[{i}] Title: {title}")
        lines.append(f"    Abstract: {text[:600]}{'[...]' if len(text) > 600 else ''}\n")

    lines.append(
        "What concrete patterns do you recognize that make these abstracts relevant "
        "to this question? What should a reranker look for?"
    )
    return "\n".join(lines)


def load_qrels(path: Path) -> dict:
    qrels = defaultdict(set)
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


def load_queries(path: Path) -> dict:
    queries = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            queries[r["qid"]] = r["query"]
    return queries


def load_corpus_subset(docids: set) -> dict:
    corpus = {}
    with CORPUS.open() as f:
        for line in f:
            doc = json.loads(line)
            if doc["_id"] in docids:
                corpus[doc["_id"]] = {
                    "title": doc.get("title", ""),
                    "text":  doc.get("text", ""),
                }
            if len(corpus) == len(docids):
                break
    return corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-queries", type=int, default=10)
    parser.add_argument("--model",     default=MODEL)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable first.")

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print("Loading qrels …")
    qrels = load_qrels(QRELS)

    print("Loading queries …")
    queries = load_queries(IDS_FILE)

    # Pick first N queries that have both a query text and gold docs
    qids = [qid for qid in queries if qid in qrels and qrels[qid]][:args.n_queries]
    print(f"Analyzing {len(qids)} queries …\n")

    # Load only the gold docs we need
    needed = {did for qid in qids for did in qrels[qid]}
    print(f"Loading {len(needed)} gold documents from corpus …")
    corpus = load_corpus_subset(needed)
    missing = len(needed) - len(corpus)
    if missing:
        print(f"  Warning: {missing} gold docids not found in corpus")
    print(f"  {len(corpus)} found\n")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for i, qid in enumerate(qids, start=1):
        query     = queries[qid]
        gold_ids  = qrels[qid]
        gold_docs = [corpus[did] for did in gold_ids if did in corpus]

        print(f"[{i}/{len(qids)}] Query: {query}")
        print(f"         Gold docs: {len(gold_docs)} (of {len(gold_ids)} in qrels)")

        if not gold_docs:
            print("         Skipping — no gold docs found in corpus\n")
            continue

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(query, gold_docs)},
        ]

        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=0,
                max_tokens=600,
            )
            analysis = resp.choices[0].message.content or ""
        except Exception as e:
            analysis = f"[API ERROR: {e}]"

        print(f"\n{analysis}\n")
        print("─" * 80 + "\n")

        results.append({
            "qid":       qid,
            "query":     query,
            "n_gold":    len(gold_docs),
            "analysis":  analysis,
        })

    # Save to file
    with OUT_FILE.open("w") as f:
        for r in results:
            f.write(f"Query ({r['qid']}): {r['query']}\n")
            f.write(f"Gold docs: {r['n_gold']}\n\n")
            f.write(r["analysis"] + "\n")
            f.write("=" * 80 + "\n\n")

    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
