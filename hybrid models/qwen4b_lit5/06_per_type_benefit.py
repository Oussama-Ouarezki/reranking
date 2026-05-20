"""Per-type benefit summary: does LiT5 help over Qwen4B alone?

Compares all candidate methods at the metric that matters per type
(global=ndcg@10, summary=ndcg@10, factoid=ndcg@5, list=ndcg@3, yesno=ndcg@1):

    Qwen                Qwen-prob descending
    LiT5/BM25-50        Pure LiT5 sliding window over BM25 top-50
    LiT5/Qwen-top20     LiT5 reorders Qwen-top-20, Qwen tail appended
    RRF (always)        RRF(Qwen, LiT5/BM25-50) on every query, k=60
    RRF (yesno-only)    RRF only when type==yesno, else Qwen

The last row is what the sweep in 05 suggests as the recommended policy:
keep Qwen everywhere except yesno, where fusing with LiT5 gives a clear lift.

Reads:  qwen4b_uncertainty/data/qwen_scores_test.jsonl
        qwen4b_lit5/data/lit5_bm25_top50_test.jsonl
        qwen4b_lit5/data/lit5_qwen_top20_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_lit5/data/06_per_type_benefit.tsv
"""

import json
from pathlib import Path

import ir_measures
from ir_measures import Qrel, ScoredDoc, nDCG

BASE = Path(__file__).resolve().parents[1]
QWEN_F  = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
L_BM25  = BASE / "qwen4b_lit5/data/lit5_bm25_top50_test.jsonl"
L_QWEN  = BASE / "qwen4b_lit5/data/lit5_qwen_top20_test.jsonl"
QRELS_F = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV = BASE / "qwen4b_lit5/data/06_per_type_benefit.tsv"

TYPES        = ["summary", "factoid", "list", "yesno"]
METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]
TARGET = {"global": "ndcg@10", "summary": "ndcg@10",
          "factoid": "ndcg@5", "list": "ndcg@3", "yesno": "ndcg@1"}
RRF_K = 60


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


def rrf(a: list[str], b: list[str], k: int = RRF_K) -> list[str]:
    docs = set(a) | set(b)
    pa = {d: i for i, d in enumerate(a)}
    pb = {d: i for i, d in enumerate(b)}
    big = max(len(a), len(b)) + 1
    scored = [(d, 1.0/(k + pa.get(d, big) + 1) + 1.0/(k + pb.get(d, big) + 1))
              for d in docs]
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

    lit5_bm25: dict[str, list[str]] = {}
    with L_BM25.open() as f:
        for line in f:
            r = json.loads(line)
            lit5_bm25[r["qid"]] = r["perm"]

    lit5_qwen: dict[str, dict] = {}
    with L_QWEN.open() as f:
        for line in f:
            r = json.loads(line)
            lit5_qwen[r["qid"]] = r

    qids = sorted(set(qwen_by_qid) & set(lit5_bm25) & set(lit5_qwen))
    print(f"{len(qids)} queries")
    qrels = load_qrels()

    qwen_rank: dict[str, list[str]] = {}
    for qid in qids:
        ranked = sorted(qwen_by_qid[qid], key=lambda s: s["qwen_prob"], reverse=True)
        qwen_rank[qid] = [s["docid"] for s in ranked]

    def mk_run(rank_fn) -> list[ScoredDoc]:
        out: list[ScoredDoc] = []
        for qid in qids:
            r = rank_fn(qid)
            for i, d in enumerate(r):
                out.append(ScoredDoc(qid, d, float(len(r) - i)))
        return out

    runs = {
        "Qwen":             mk_run(lambda q: qwen_rank[q]),
        "LiT5/BM25-50":     mk_run(lambda q: lit5_bm25[q]),
        "LiT5/Qwen-top20":  mk_run(lambda q: lit5_qwen[q]["perm"] +
                                    [d for d in qwen_rank[q] if d not in set(lit5_qwen[q]["perm"])]),
        "RRF (always)":     mk_run(lambda q: rrf(qwen_rank[q], lit5_bm25[q])),
        "RRF (yesno-only)": mk_run(lambda q: rrf(qwen_rank[q], lit5_bm25[q])
                                    if qtype[q] == "yesno" else qwen_rank[q]),
    }

    scopes: list[tuple[str, set[str]]] = [("global", set(qids))]
    for t in TYPES:
        scopes.append((t, {q for q in qids if qtype[q] == t}))

    print()
    header = f"{'method':<18} " + "  ".join(f"{sc + '/' + TARGET[sc]:>16}" for sc, _ in scopes)
    print(header)
    print("-" * len(header))
    rows = []
    qwen_metrics: dict[str, dict[str, float]] = {sc: evaluate(runs["Qwen"], qrels, qset) for sc, qset in scopes}

    for name, run in runs.items():
        line_parts = [f"{name:<18}"]
        row = {"method": name}
        for sc, qset in scopes:
            m = evaluate(run, qrels, qset)
            tgt = TARGET[sc]
            base = qwen_metrics[sc][tgt]
            delta = m[tgt] - base
            line_parts.append(f"{m[tgt]:.4f} ({delta:+.4f})")
            row[f"{sc}_{tgt}"]       = m[tgt]
            row[f"{sc}_{tgt}_delta"] = delta
            for mn in METRIC_NAMES:
                row[f"{sc}_{mn}"] = m[mn]
        rows.append(row)
        print(f"{name:<18} " + "  ".join(f"{p:>16}" for p in line_parts[1:]))

    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method"]
    for sc, _ in scopes:
        cols.append(f"{sc}_{TARGET[sc]}")
        cols.append(f"{sc}_{TARGET[sc]}_delta")
    with OUT_TSV.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(
                f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                for c in cols
            ) + "\n")
    print(f"\nsaved -> {OUT_TSV}")

    # Bottom line
    print("\nbottom line (target-metric delta vs Qwen):")
    for sc, _ in scopes:
        tgt = TARGET[sc]
        best_method = max(runs.keys(),
                          key=lambda n: next(r for r in rows if r["method"] == n)[f"{sc}_{tgt}"])
        d = next(r for r in rows if r["method"] == best_method)[f"{sc}_{tgt}_delta"]
        verdict = "Qwen alone is best" if best_method == "Qwen" else f"{best_method} > Qwen by {d:+.4f}"
        print(f"  {sc:<8} ({tgt}):  {verdict}")


if __name__ == "__main__":
    main()
