"""Shared utilities for the qwen3_0.6b experiments."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import ir_measures
from ir_measures import nDCG, RR, Qrel, ScoredDoc

BASE = Path(__file__).resolve().parents[1]
QRELS_F = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"

METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, RR @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@10"]
TYPES = ["summary", "factoid", "list", "yesno"]

QWEN_LF_DYNAMIC10_ALPHA: dict[str, float] = {
    "list":    0.975,
    "summary": 1.000,
    "yesno":   0.750,
    "factoid": 0.975,
}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def load_qwen_scores(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate(run: list[ScoredDoc], qrels: list[Qrel], qid_set: set[str] | None = None) -> dict[str, float]:
    if qid_set is not None:
        sub_run = [r for r in run if r.query_id in qid_set]
        sub_q = [q for q in qrels if q.query_id in qid_set]
    else:
        sub_run, sub_q = run, qrels
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def report(label: str, run: list[ScoredDoc], qrels: list[Qrel], qtypes: dict[str, str]) -> dict:
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    out = {"global": evaluate(run, qrels)}
    for t in TYPES:
        out[t] = evaluate(run, qrels, type_qids[t])

    print(f"\n=== {label} ===")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    for sc in ["global"] + TYPES:
        vals = "  ".join(f"{out[sc][m]:>8.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {vals}")
    return out


def write_trec(path: Path, run: list[ScoredDoc], tag: str) -> None:
    by_qid: dict[str, list[ScoredDoc]] = defaultdict(list)
    for r in run:
        by_qid[r.query_id].append(r)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for qid, recs in by_qid.items():
            recs_sorted = sorted(recs, key=lambda x: x.score, reverse=True)
            for rank, r in enumerate(recs_sorted, start=1):
                f.write(f"{qid}\tQ0\t{r.doc_id}\t{rank}\t{r.score:.6f}\t{tag}\n")
