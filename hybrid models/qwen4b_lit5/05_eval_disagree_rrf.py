"""Disagreement-gated RRF between Qwen4B and LiT5/BM25-top-50.

Strategy: for queries where Qwen and LiT5 disagree on the top, fuse their
rankings via Reciprocal Rank Fusion. For agreeing queries, keep Qwen.

Disagreement signal: 1 - |Qwen_top10 ∩ LiT5_top10| / 10
Gate:                use RRF iff disagreement >= tau

Sweeps:
    tau in {0.0, 0.1, ..., 1.0}     (0.0 = always RRF, 1.0 = never RRF)
    rrf_k in {10, 30, 60, 100}      (RRF constant)

For each (tau, rrf_k), reports nDCG@{1,3,5,10} globally and per question type,
saves a TSV, and prints the best config per scope.

Reads:  qwen4b_uncertainty/data/qwen_scores_test.jsonl
        qwen4b_lit5/data/lit5_bm25_top50_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_lit5/data/05_disagree_rrf_sweep.tsv
        qwen4b_lit5/data/05_disagree_rrf_best.json
"""

import json
from pathlib import Path

import ir_measures
from ir_measures import Qrel, ScoredDoc, nDCG

BASE = Path(__file__).resolve().parents[1]
QWEN_F  = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
LIT5_F  = BASE / "qwen4b_lit5/data/lit5_bm25_top50_test.jsonl"
QRELS_F = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV = BASE / "qwen4b_lit5/data/05_disagree_rrf_sweep.tsv"
OUT_JSON= BASE / "qwen4b_lit5/data/05_disagree_rrf_best.json"

TYPES        = ["summary", "factoid", "list", "yesno"]
METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]
TARGET_PER_TYPE = {"global": "ndcg@10", "summary": "ndcg@10",
                   "factoid": "ndcg@5", "list": "ndcg@3", "yesno": "ndcg@1"}

TAUS  = [round(0.1 * i, 1) for i in range(11)]      # 0.0 .. 1.0
RRFKS = [10, 30, 60, 100]
TOPK_OVERLAP = 10


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def evaluate(run: list[ScoredDoc], qrels: list[Qrel], qid_set: set[str]) -> dict[str, float]:
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q   = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def rrf(qwen_rank: list[str], lit5_rank: list[str], k: int) -> list[str]:
    """Return docids fused by Reciprocal Rank Fusion (descending score)."""
    docs = set(qwen_rank) | set(lit5_rank)
    qpos = {d: i for i, d in enumerate(qwen_rank)}
    lpos = {d: i for i, d in enumerate(lit5_rank)}
    big = max(len(qwen_rank), len(lit5_rank)) + 1
    scored = []
    for d in docs:
        rq = qpos.get(d, big)
        rl = lpos.get(d, big)
        s = 1.0 / (k + rq + 1) + 1.0 / (k + rl + 1)
        scored.append((d, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored]


def main() -> None:
    qwen_by_qid: dict[str, list[dict]] = {}
    qtype: dict[str, str] = {}
    with QWEN_F.open() as f:
        for line in f:
            r = json.loads(line)
            qwen_by_qid[r["qid"]] = r["scores"]
            qtype[r["qid"]] = r["type"]

    lit5_by_qid: dict[str, list[str]] = {}
    with LIT5_F.open() as f:
        for line in f:
            r = json.loads(line)
            lit5_by_qid[r["qid"]] = r["perm"]

    qids = sorted(set(qwen_by_qid) & set(lit5_by_qid))
    print(f"{len(qids)} queries with both sources")

    qrels = load_qrels()
    print(f"{len(qrels)} qrel rows")

    # Pre-compute per-qid Qwen ranking, LiT5 ranking, disagreement
    qwen_rank: dict[str, list[str]] = {}
    lit5_rank: dict[str, list[str]] = {}
    disagree:  dict[str, float]     = {}
    for qid in qids:
        qr = sorted(qwen_by_qid[qid], key=lambda s: s["qwen_prob"], reverse=True)
        qwen_rank[qid] = [s["docid"] for s in qr]
        lit5_rank[qid] = list(lit5_by_qid[qid])
        a = set(qwen_rank[qid][:TOPK_OVERLAP])
        b = set(lit5_rank[qid][:TOPK_OVERLAP])
        disagree[qid] = 1.0 - len(a & b) / TOPK_OVERLAP

    # Disagreement distribution
    print("\ntop-10 disagreement distribution:")
    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        n = sum(1 for d in disagree.values() if lo <= d < hi)
        print(f"  [{lo:.1f}, {hi:.2f}):  {n:>4}")

    # Pure Qwen baseline (tau >= 1.0 is equivalent)
    qwen_run = [ScoredDoc(qid, d, float(50 - i))
                for qid in qids for i, d in enumerate(qwen_rank[qid])]

    scopes: list[tuple[str, set[str]]] = [("global", set(qids))]
    for t in TYPES:
        scopes.append((t, {q for q in qids if qtype[q] == t}))

    # Baseline metrics
    base = {sc: evaluate(qwen_run, qrels, qset) for sc, qset in scopes}
    print(f"\nQwen baseline:")
    for sc, qset in scopes:
        print(f"  {sc:<8} n={len(qset):>3}  " +
              "  ".join(f"{m}={base[sc][m]:.4f}" for m in METRIC_NAMES))

    # Sweep
    rows: list[dict] = []
    for tau in TAUS:
        for k in RRFKS:
            run: list[ScoredDoc] = []
            n_fused = 0
            for qid in qids:
                if disagree[qid] >= tau:
                    fused = rrf(qwen_rank[qid], lit5_rank[qid], k)
                    n_fused += 1
                else:
                    fused = qwen_rank[qid]
                for i, d in enumerate(fused):
                    run.append(ScoredDoc(qid, d, float(len(fused) - i)))
            for sc, qset in scopes:
                m = evaluate(run, qrels, qset)
                rows.append({"tau": tau, "rrf_k": k, "scope": sc,
                             "n": len(qset), "n_fused": n_fused, **m})

    # Save sweep
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w") as f:
        f.write("tau\trrf_k\tscope\tn\tn_fused\t" + "\t".join(METRIC_NAMES) + "\n")
        for r in rows:
            f.write(f"{r['tau']}\t{r['rrf_k']}\t{r['scope']}\t{r['n']}\t{r['n_fused']}\t" +
                    "\t".join(f"{r[m]:.4f}" for m in METRIC_NAMES) + "\n")
    print(f"\nsaved sweep -> {OUT_TSV}")

    # Best per scope at the scope's target metric, plus delta vs Qwen baseline
    best: dict[str, dict] = {}
    for sc, _ in scopes:
        target = TARGET_PER_TYPE[sc]
        sc_rows = [r for r in rows if r["scope"] == sc]
        top = max(sc_rows, key=lambda r: r[target])
        best[sc] = {
            "tau": top["tau"], "rrf_k": top["rrf_k"], "n_fused": top["n_fused"],
            "target_metric": target, "value": top[target],
            "baseline": base[sc][target],
            "delta": top[target] - base[sc][target],
            **{m: top[m] for m in METRIC_NAMES},
        }

    print(f"\nBest config per scope (target metric in parentheses):")
    print(f"  {'scope':<8} {'target':<8} {'tau':>4} {'rrf_k':>5} {'n_fused':>7} {'value':>8} {'base':>8} {'delta':>8}")
    for sc, _ in scopes:
        b = best[sc]
        print(f"  {sc:<8} {b['target_metric']:<8} {b['tau']:>4.1f} {b['rrf_k']:>5} "
              f"{b['n_fused']:>7} {b['value']:>8.4f} {b['baseline']:>8.4f} {b['delta']:>+8.4f}")

    OUT_JSON.write_text(json.dumps(best, indent=2))
    print(f"saved best -> {OUT_JSON}")


if __name__ == "__main__":
    main()
