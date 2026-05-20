"""Evaluate LiT5 reordering of Qwen-top-20 vs Qwen alone.

For each query the top-20 by qwen_prob is reordered by LiT5 (single window);
docs ranked 21-50 by Qwen are appended below unchanged. Compares this hybrid
against pure Qwen using nDCG@{1,3,5,10}, globally and per question type.

Also reports per-query nDCG@10 delta so we can see which queries LiT5 helps,
hurts, or leaves unchanged.

Reads:  qwen4b_uncertainty/data/qwen_scores_test.jsonl
        qwen4b_lit5/data/lit5_qwen_top20_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_lit5/data/04_eval_lit5_qwen_top20.tsv          (aggregate)
        qwen4b_lit5/data/04_per_query_delta.tsv               (per-qid delta)
"""

import json
from collections import defaultdict
from pathlib import Path

import ir_measures
from ir_measures import Qrel, ScoredDoc, nDCG

BASE = Path(__file__).resolve().parents[1]
QWEN_F  = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
LIT5_F  = BASE / "qwen4b_lit5/data/lit5_qwen_top20_test.jsonl"
QRELS_F = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV = BASE / "qwen4b_lit5/data/04_eval_lit5_qwen_top20.tsv"
DELTA_F = BASE / "qwen4b_lit5/data/04_per_query_delta.tsv"

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


def per_query_ndcg10(run: list[ScoredDoc], qrels: list[Qrel]) -> dict[str, float]:
    out: dict[str, float] = {}
    for measurement in ir_measures.iter_calc([nDCG @ 10], qrels, run):
        out[measurement.query_id] = float(measurement.value)
    return out


def main() -> None:
    qwen_by_qid: dict[str, list[dict]] = {}
    qtype: dict[str, str] = {}
    with QWEN_F.open() as f:
        for line in f:
            r = json.loads(line)
            qwen_by_qid[r["qid"]] = r["scores"]
            qtype[r["qid"]] = r["type"]

    lit5_by_qid: dict[str, dict] = {}
    with LIT5_F.open() as f:
        for line in f:
            r = json.loads(line)
            lit5_by_qid[r["qid"]] = r

    qids = sorted(set(qwen_by_qid) & set(lit5_by_qid))
    print(f"{len(qids)} queries with both sources")

    qrels = load_qrels()
    print(f"{len(qrels)} qrel rows")

    # Pure Qwen run: qwen_prob descending, all 50 docs
    qwen_run: list[ScoredDoc] = []
    # LiT5/Qwen-top20: LiT5 perm of top-20 by Qwen, then docs 21-50 by Qwen score
    hybrid_run: list[ScoredDoc] = []

    for qid in qids:
        ranked = sorted(qwen_by_qid[qid], key=lambda s: s["qwen_prob"], reverse=True)
        for s in ranked:
            qwen_run.append(ScoredDoc(qid, s["docid"], float(s["qwen_prob"])))

        lit5 = lit5_by_qid[qid]
        top20_set = set(lit5["qwen_top20"])
        # Top 20 by LiT5 perm — assign descending integer scores 50..31
        for i, did in enumerate(lit5["perm"]):
            hybrid_run.append(ScoredDoc(qid, did, float(50 - i)))
        # Tail: ranks 21..50 by Qwen score, score 30..1
        tail = [s for s in ranked if s["docid"] not in top20_set]
        for i, s in enumerate(tail):
            hybrid_run.append(ScoredDoc(qid, s["docid"], float(30 - i)))

    # Aggregate per scope
    scopes: list[tuple[str, set[str]]] = [("global", set(qids))]
    for t in TYPES:
        scopes.append((t, {q for q in qids if qtype[q] == t}))

    runs = [
        ("Qwen4B",            qwen_run),
        ("LiT5/Qwen-top20",   hybrid_run),
    ]

    results: dict[tuple[str, str], dict[str, float]] = {}
    for name, run in runs:
        for sc, qset in scopes:
            results[(name, sc)] = evaluate(run, qrels, qset)

    header = f"{'method':<18} {'scope':<8} {'n':>4}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
    print()
    print(header)
    print("-" * len(header))
    for name, _ in runs:
        for sc, qset in scopes:
            r = results[(name, sc)]
            print(f"{name:<18} {sc:<8} {len(qset):>4}  " + "  ".join(f"{r[m]:>8.4f}" for m in METRIC_NAMES))
        print()

    # Delta vs Qwen
    print(f"{'delta vs Qwen':<18} {'scope':<8} {'n':>4}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    print("-" * len(header))
    for sc, qset in scopes:
        rb = results[("Qwen4B", sc)]
        rh = results[("LiT5/Qwen-top20", sc)]
        delta = {m: rh[m] - rb[m] for m in METRIC_NAMES}
        print(f"{'(LiT5 - Qwen)':<18} {sc:<8} {len(qset):>4}  " + "  ".join(f"{delta[m]:>+8.4f}" for m in METRIC_NAMES))

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w") as f:
        f.write("method\tscope\tn\t" + "\t".join(METRIC_NAMES) + "\n")
        for name, _ in runs:
            for sc, qset in scopes:
                r = results[(name, sc)]
                f.write(f"{name}\t{sc}\t{len(qset)}\t" +
                        "\t".join(f"{r[m]:.4f}" for m in METRIC_NAMES) + "\n")
    print(f"saved -> {OUT_TSV}")

    # Per-query delta
    qwen_pq   = per_query_ndcg10(qwen_run, qrels)
    hybrid_pq = per_query_ndcg10(hybrid_run, qrels)

    helped = hurt = same = 0
    by_type_helped: dict[str, int] = defaultdict(int)
    by_type_hurt:   dict[str, int] = defaultdict(int)
    by_type_total:  dict[str, int] = defaultdict(int)

    with DELTA_F.open("w") as f:
        f.write("qid\ttype\tqwen_ndcg10\tlit5_ndcg10\tdelta\n")
        for qid in qids:
            qv = qwen_pq.get(qid, float("nan"))
            hv = hybrid_pq.get(qid, float("nan"))
            d = hv - qv
            f.write(f"{qid}\t{qtype[qid]}\t{qv:.4f}\t{hv:.4f}\t{d:+.4f}\n")
            by_type_total[qtype[qid]] += 1
            if d > 1e-6:
                helped += 1
                by_type_helped[qtype[qid]] += 1
            elif d < -1e-6:
                hurt += 1
                by_type_hurt[qtype[qid]] += 1
            else:
                same += 1
    print(f"\nper-query nDCG@10 deltas (LiT5/Qwen-top20 - Qwen):")
    print(f"  helped: {helped}   hurt: {hurt}   unchanged: {same}")
    print(f"  {'type':<8}  {'n':>4}  {'helped':>7}  {'hurt':>5}  {'unchanged':>9}")
    for t in TYPES:
        n = by_type_total[t]
        h = by_type_helped[t]
        u = by_type_hurt[t]
        s = n - h - u
        print(f"  {t:<8}  {n:>4}  {h:>7}  {u:>5}  {s:>9}")
    print(f"saved -> {DELTA_F}")


if __name__ == "__main__":
    main()
