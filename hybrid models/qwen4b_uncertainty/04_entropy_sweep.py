"""Entropy threshold sweep — H50 over Qwen probabilities, fuse with BM25 when uncertain.

For each query:
  H50 = Shannon entropy of softmax(qwen_probs).
  If H50 > tau: re-rank by fused score.
    - linear: alpha*qwen_norm + (1-alpha)*bm25_norm  (per-query min-max norm)
    - rrf:    sum 1/(k + rank) over the two ranklists
  Else: keep pure-Qwen ranking.

Sweep tau on a grid and report nDCG@1/@5/@10 globally and per-type.

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/sweep_metrics.tsv
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

import ir_measures
from ir_measures import nDCG, ScoredDoc, Qrel

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores.jsonl"
QRELS_F = BASE / "data/bioasq/processed/qrels.tsv"
OUT = BASE / "qwen4b_uncertainty/data/sweep_metrics.tsv"

TYPES = ["summary", "factoid", "list", "yesno"]
ALPHAS = [0.3, 0.5, 0.7]
RRF_K = 60
N_TAU = 41
METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10"]


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def load_scores() -> list[dict]:
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def shannon_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def rank_array(x: np.ndarray) -> np.ndarray:
    order = np.argsort(-x, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def build_run(rows: list[dict], taus_use_fusion: dict[str, np.ndarray], H: dict[str, float],
              tau: float, fusion: str, alpha: float) -> list[ScoredDoc]:
    run = []
    for r in rows:
        qid = r["qid"]
        docids = [s["docid"] for s in r["scores"]]
        qwen = np.array([s["qwen_prob"] for s in r["scores"]], dtype=float)
        bm25 = np.array([s["bm25_score"] for s in r["scores"]], dtype=float)
        if H[qid] > tau:
            if fusion == "linear":
                fused = alpha * minmax(qwen) + (1 - alpha) * minmax(bm25)
            elif fusion == "rrf":
                rq = rank_array(qwen)
                rb = rank_array(bm25)
                fused = 1.0 / (RRF_K + rq) + 1.0 / (RRF_K + rb)
            else:
                raise ValueError(fusion)
            scores = fused
        else:
            scores = qwen
        for d, s in zip(docids, scores):
            run.append(ScoredDoc(qid, d, float(s)))
    return run


def evaluate_scope(run: list[ScoredDoc], qrels: list[Qrel], qids: set[str]) -> dict[str, float]:
    sub_run = [r for r in run if r.query_id in qids]
    sub_qrels = [q for q in qrels if q.query_id in qids]
    if not sub_run or not sub_qrels:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_qrels, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main() -> None:
    rows = load_scores()
    qrels = load_qrels()
    print(f"{len(rows)} queries  /  {len(qrels)} qrel rows")

    # Precompute entropy
    H: dict[str, float] = {}
    for r in rows:
        probs = np.array([s["qwen_prob"] for s in r["scores"]], dtype=float)
        H[r["qid"]] = shannon_entropy(softmax(probs))
    h_max = math.log(50)
    print(f"H50 range: {min(H.values()):.3f} .. {max(H.values()):.3f}  (max possible = {h_max:.3f})")

    type_qids: dict[str, set[str]] = defaultdict(set)
    all_qids: set[str] = set()
    for r in rows:
        type_qids[r["type"]].add(r["qid"])
        all_qids.add(r["qid"])

    taus = list(np.linspace(0.0, h_max, N_TAU))
    taus.append(float("inf"))  # baseline: pure Qwen everywhere

    variants: list[tuple[str, float | None]] = [("linear", a) for a in ALPHAS] + [("rrf", None)]
    scopes: list[tuple[str, set[str]]] = [("global", all_qids)] + [(t, type_qids[t]) for t in TYPES]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        f.write("tau\tfusion\talpha\tscope\tn_uncertain\t" + "\t".join(METRIC_NAMES) + "\n")
        for tau in tqdm(taus, desc="tau"):
            n_unc = sum(1 for v in H.values() if v > tau)
            for fusion, alpha in variants:
                run = build_run(rows, {}, H, tau, fusion, alpha if alpha is not None else 0.0)
                for scope_name, qids in scopes:
                    m = evaluate_scope(run, qrels, qids)
                    f.write(
                        f"{tau}\t{fusion}\t{alpha if alpha is not None else ''}\t"
                        f"{scope_name}\t{n_unc}\t"
                        + "\t".join(f"{m[n]:.6f}" for n in METRIC_NAMES) + "\n"
                    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
