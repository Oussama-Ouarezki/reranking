"""Exp 1: Pure Qwen3-0.6B baseline.

Rank BM25 top-50 by Qwen3-Reranker-0.6B probability.
"""

import json
from pathlib import Path

from ir_measures import ScoredDoc

from _common import (
    BASE, load_qrels, load_qwen_scores, report, write_trec,
)

SCORES_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
OUT_TREC = BASE / "qwen3_0.6b/results/run_pure_qwen.tsv"
OUT_JSON = BASE / "qwen3_0.6b/results/metrics_pure_qwen.json"


def main() -> None:
    rows = load_qwen_scores(SCORES_F)
    qrels = load_qrels()
    qtypes = {r["qid"]: r["type"] for r in rows}

    run = [
        ScoredDoc(r["qid"], s["docid"], float(s["qwen_prob"]))
        for r in rows for s in r["scores"]
    ]

    metrics = report("Exp 1: Pure Qwen3-0.6B", run, qrels, qtypes)

    write_trec(OUT_TREC, run, tag="qwen06b")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {OUT_TREC}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
