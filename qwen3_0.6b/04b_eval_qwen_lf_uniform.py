"""Exp 2b: Qwen3-0.6B + linear fusion with uniform alpha=0.975 across query types.

score(d) = 0.975 * qwen_prob(d) + 0.025 * bm25_minmax(d)
"""

import json

import numpy as np
from ir_measures import ScoredDoc

from _common import (
    BASE, load_qrels, load_qwen_scores, minmax, report, write_trec,
)

import sys
ALPHA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.975
SCORES_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
OUT_TREC = BASE / f"qwen3_0.6b/results/run_qwen_lf_uniform_{ALPHA}.tsv"
OUT_JSON = BASE / f"qwen3_0.6b/results/metrics_qwen_lf_uniform_{ALPHA}.json"


def main() -> None:
    rows = load_qwen_scores(SCORES_F)
    qrels = load_qrels()
    qtypes = {r["qid"]: r["type"] for r in rows}

    run: list[ScoredDoc] = []
    for r in rows:
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        fused = ALPHA * q + (1.0 - ALPHA) * minmax(b)
        for s, fv in zip(items, fused):
            run.append(ScoredDoc(r["qid"], s["docid"], float(fv)))

    metrics = report(f"Exp 2b: Qwen + uniform LF (alpha={ALPHA})", run, qrels, qtypes)

    write_trec(OUT_TREC, run, tag="qwen06b_lf_uniform")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"alpha": ALPHA, "metrics": metrics}, indent=2))
    print(f"\nwrote {OUT_TREC}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
