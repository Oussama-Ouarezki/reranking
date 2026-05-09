"""Exp 2: Qwen3-0.6B + per-type linear fusion with BM25.

score(d) = alpha_t * qwen_prob(d) + (1 - alpha_t) * bm25_minmax(d)
where alpha_t comes from QWEN_LF_DYNAMIC10_ALPHA, keyed by query type.
"""

import json

import numpy as np
from ir_measures import ScoredDoc

from _common import (
    BASE, QWEN_LF_DYNAMIC10_ALPHA, load_qrels, load_qwen_scores,
    minmax, report, write_trec,
)

SCORES_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
OUT_TREC = BASE / "qwen3_0.6b/results/run_qwen_lf.tsv"
OUT_JSON = BASE / "qwen3_0.6b/results/metrics_qwen_lf.json"
OUT_FUSED = BASE / "qwen3_0.6b/data/qwen_lf_fused_test.jsonl"


def main() -> None:
    rows = load_qwen_scores(SCORES_F)
    qrels = load_qrels()
    qtypes = {r["qid"]: r["type"] for r in rows}

    run: list[ScoredDoc] = []
    fused_records = []

    for r in rows:
        qid = r["qid"]
        t = r["type"]
        alpha = QWEN_LF_DYNAMIC10_ALPHA.get(t, 1.0)
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        b_norm = minmax(b)
        fused = alpha * q + (1.0 - alpha) * b_norm

        per_doc = []
        for s, qv, bn, fv in zip(items, q, b_norm, fused):
            run.append(ScoredDoc(qid, s["docid"], float(fv)))
            per_doc.append({
                "docid": s["docid"],
                "qwen_prob": float(qv),
                "bm25_score": float(s["bm25_score"]),
                "bm25_minmax": float(bn),
                "fused_score": float(fv),
            })
        per_doc.sort(key=lambda x: x["fused_score"], reverse=True)
        fused_records.append({"qid": qid, "type": t, "alpha": alpha, "ranked": per_doc})

    metrics = report("Exp 2: Qwen3-0.6B + per-type linear fusion", run, qrels, qtypes)

    write_trec(OUT_TREC, run, tag="qwen06b_lf")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "alphas": QWEN_LF_DYNAMIC10_ALPHA,
        "metrics": metrics,
    }, indent=2))
    OUT_FUSED.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FUSED.open("w") as f:
        for rec in fused_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {OUT_TREC}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_FUSED}")


if __name__ == "__main__":
    main()
