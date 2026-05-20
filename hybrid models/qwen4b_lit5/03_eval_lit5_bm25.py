"""Evaluate pure LiT5 reranking of BM25 top-50 vs BM25 and Qwen baselines.

Compares three rankings on the Task13BGoldenEnriched test set:
    - BM25       (bm25_score)
    - Qwen4B     (qwen_prob)
    - LiT5/BM25  (LiT5 sliding-window over BM25 top-50)

Reports nDCG@{1,3,5,10}, globally and per question type.

Reads:  qwen4b_uncertainty/data/bm25_top50_test.jsonl
        qwen4b_uncertainty/data/qwen_scores_test.jsonl
        qwen4b_lit5/data/lit5_bm25_top50_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_lit5/data/03_eval_lit5_bm25.tsv
"""

import json
from pathlib import Path

import ir_measures
from ir_measures import Qrel, ScoredDoc, nDCG

BASE = Path(__file__).resolve().parents[1]
HITS_F  = BASE / "qwen4b_uncertainty/data/bm25_top50_test.jsonl"
QWEN_F  = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
LIT5_F  = BASE / "qwen4b_lit5/data/lit5_bm25_top50_test.jsonl"
QRELS_F = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV = BASE / "qwen4b_lit5/data/03_eval_lit5_bm25.tsv"

TYPES        = ["summary", "factoid", "list", "yesno"]
METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]


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


def main() -> None:
    # Load BM25 hits + qtype
    hits_by_qid: dict[str, list[dict]] = {}
    qtype: dict[str, str] = {}
    with HITS_F.open() as f:
        for line in f:
            r = json.loads(line)
            hits_by_qid[r["qid"]] = r["hits"]
            qtype[r["qid"]] = r["type"]

    # Load Qwen scores
    qwen_by_qid: dict[str, list[dict]] = {}
    with QWEN_F.open() as f:
        for line in f:
            r = json.loads(line)
            qwen_by_qid[r["qid"]] = r["scores"]

    # Load LiT5 permutations
    lit5_by_qid: dict[str, list[str]] = {}
    with LIT5_F.open() as f:
        for line in f:
            r = json.loads(line)
            lit5_by_qid[r["qid"]] = r["perm"]

    qids = sorted(set(hits_by_qid) & set(qwen_by_qid) & set(lit5_by_qid))
    print(f"{len(qids)} queries with all three sources")

    qrels = load_qrels()
    print(f"{len(qrels)} qrel rows")

    # Build runs
    bm25_run: list[ScoredDoc] = []
    qwen_run: list[ScoredDoc] = []
    lit5_run: list[ScoredDoc] = []

    for qid in qids:
        for h in hits_by_qid[qid]:
            bm25_run.append(ScoredDoc(qid, h["docid"], float(h["bm25_score"])))
        for s in qwen_by_qid[qid]:
            qwen_run.append(ScoredDoc(qid, s["docid"], float(s["qwen_prob"])))
        perm = lit5_by_qid[qid]
        n = len(perm)
        for i, did in enumerate(perm):
            lit5_run.append(ScoredDoc(qid, did, float(n - i)))

    # Scopes: global + per-type
    scopes: list[tuple[str, set[str]]] = [("global", set(qids))]
    for t in TYPES:
        scopes.append((t, {q for q in qids if qtype[q] == t}))

    runs = [
        ("BM25",          bm25_run),
        ("Qwen4B",        qwen_run),
        ("LiT5/BM25-50",  lit5_run),
    ]

    # Compute and print
    results: dict[tuple[str, str], dict[str, float]] = {}
    for name, run in runs:
        for sc, qset in scopes:
            results[(name, sc)] = evaluate(run, qrels, qset)

    header = f"{'method':<14} {'scope':<8} {'n':>4}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
    print()
    print(header)
    print("-" * len(header))
    for name, _ in runs:
        for sc, qset in scopes:
            r = results[(name, sc)]
            row = f"{name:<14} {sc:<8} {len(qset):>4}  " + "  ".join(f"{r[m]:>8.4f}" for m in METRIC_NAMES)
            print(row)
        print()

    # Save TSV
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w") as f:
        f.write("method\tscope\tn\t" + "\t".join(METRIC_NAMES) + "\n")
        for name, _ in runs:
            for sc, qset in scopes:
                r = results[(name, sc)]
                f.write(f"{name}\t{sc}\t{len(qset)}\t" +
                        "\t".join(f"{r[m]:.4f}" for m in METRIC_NAMES) + "\n")
    print(f"saved -> {OUT_TSV}")


if __name__ == "__main__":
    main()
