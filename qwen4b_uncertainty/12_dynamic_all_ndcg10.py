"""Dynamic linear fusion where every query type optimises α for nDCG@10.

Compares with the previous mixed-target dynamic config (list→@3, summary→@10,
yesno→@1, factoid→@5).

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        qwen4b_uncertainty/data/cascade_grid.tsv  (precomputed α sweep)
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/cascade_eval_ndcg10.tsv
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import ir_measures
from ir_measures import nDCG, Qrel, ScoredDoc

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores.jsonl"
QRELS_F = BASE / "data/bioasq/processed/qrels.tsv"
GRID_F = BASE / "qwen4b_uncertainty/data/cascade_grid.tsv"
OUT_TSV = BASE / "qwen4b_uncertainty/data/cascade_eval_ndcg10.tsv"

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, nDCG @ 20]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "ndcg@20"]

# Previous "mixed-target" dynamic config (for comparison)
ALPHA_MIXED = {"list": 0.875, "summary": 0.800, "yesno": 0.750, "factoid": 0.875}
ALPHA_GLOBAL = 0.825


def minmax(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_qrels():
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def load_scores():
    return [json.loads(l) for l in SCORES_F.open()]


def evaluate(run, qrels, qid_set):
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main():
    rows = load_scores()
    qrels = load_qrels()
    grid = pd.read_csv(GRID_F, sep="\t")
    print(f"{len(rows)} queries / {len(qrels)} qrel rows")

    # Pick α* per type by maximising ndcg@10 on the precomputed grid.
    alpha_ndcg10 = {}
    for t in TYPES:
        sub = grid[grid["scope"] == t]
        best = sub.loc[sub["ndcg@10"].idxmax()]
        alpha_ndcg10[t] = float(best["alpha"])
    print("\nNew per-type α (all targeting nDCG@10):")
    for t in TYPES:
        print(f"  {t:<8}  α* = {alpha_ndcg10[t]:.3f}")
    print("\nPrevious per-type α (mixed targets):")
    for t in TYPES:
        print(f"  {t:<8}  α* = {ALPHA_MIXED[t]:.3f}")

    # Build runs and evaluate
    qwen_arr, qwen_norm, bm25_norm, docids, qtypes = {}, {}, {}, {}, {}
    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = sorted(r["scores"], key=lambda s: s["qwen_prob"], reverse=True)
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_arr[qid] = q
        qwen_norm[qid] = minmax(q)
        bm25_norm[qid] = minmax(b)
        docids[qid] = [s["docid"] for s in items]

    qids = list(qwen_arr.keys())
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    def build(alpha_fn):
        run = []
        for qid in qids:
            a = alpha_fn(qid)
            s = a * qwen_norm[qid] + (1 - a) * bm25_norm[qid] if a is not None \
                else qwen_arr[qid]
            for d, sv in zip(docids[qid], s):
                run.append(ScoredDoc(qid, d, float(sv)))
        return run

    runs = {
        "pure_qwen":              build(lambda q: None),
        "linear_global_α0.825":   build(lambda q: ALPHA_GLOBAL),
        "linear_dyn_mixed":       build(lambda q: ALPHA_MIXED[qtypes[q]]),
        "linear_dyn_ndcg10":      build(lambda q: alpha_ndcg10[qtypes[q]]),
    }

    rows_out = []
    for cfg, run in runs.items():
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            rows_out.append({"config": cfg, "scope": sc, **m})

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_TSV, sep="\t", index=False)

    print("\n" + "=" * 100)
    print(f"{'config':<26}  {'scope':<8}  " +
          "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    print("=" * 100)
    for cfg in ["pure_qwen", "linear_global_α0.825",
                "linear_dyn_mixed", "linear_dyn_ndcg10"]:
        for sc in ["global"] + TYPES:
            row = df[(df["config"] == cfg) & (df["scope"] == sc)].iloc[0]
            print(f"{cfg:<26}  {sc:<8}  " +
                  "  ".join(f"{row[m]:>8.4f}" for m in METRIC_NAMES))
        print("-" * 100)

    # Side-by-side: dyn_mixed vs dyn_ndcg10
    print("\n" + "=" * 100)
    print("Δ  (linear_dyn_ndcg10  −  linear_dyn_mixed)")
    print("=" * 100)
    print(f"{'scope':<8}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    for sc in ["global"] + TYPES:
        m_mixed = df[(df["config"] == "linear_dyn_mixed") &
                     (df["scope"] == sc)].iloc[0]
        m_n10 = df[(df["config"] == "linear_dyn_ndcg10") &
                   (df["scope"] == sc)].iloc[0]
        deltas = [m_n10[m] - m_mixed[m] for m in METRIC_NAMES]
        print(f"{sc:<8}  " + "  ".join(f"{d:>+8.4f}" for d in deltas))

    print(f"\nwrote {OUT_TSV}")


if __name__ == "__main__":
    main()
