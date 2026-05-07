"""Cascade monoT5 -> Qwen4b based on top-20 monoT5 entropy. Sweep entropy thresholds."""
import json
import math
from pathlib import Path
import ir_measures
from ir_measures import nDCG, MRR

ROOT = Path("/home/oussama/Desktop/reranking_project")
DIR = ROOT / "monot5_qwen"
QRELS_PATH = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"

ENTROPY_TOPK = 20


def load_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["qid"]] = d["scores"]
    return out


def entropy_topk(scores, k=ENTROPY_TOPK, key="monot5_prob"):
    probs = sorted([s[key] for s in scores], reverse=True)[:k]
    total = sum(probs)
    if total <= 0:
        return 0.0
    p = [x / total for x in probs]
    return -sum(x * math.log(x) for x in p if x > 0)


def ranking_from(scores, key):
    return sorted(scores, key=lambda s: s[key], reverse=True)


def to_run(rankings, run_name="run"):
    """rankings: dict qid -> list of dicts with docid + score key"""
    run = []
    for qid, ranked in rankings.items():
        for rank, doc in enumerate(ranked):
            run.append(ir_measures.ScoredDoc(qid, doc["docid"], -float(rank)))
    return run


def load_qrels(path):
    qrels = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            qid, did, s = line.rstrip("\n").split("\t")
            qrels.append(ir_measures.Qrel(qid, did, int(s)))
    return qrels


def main():
    monot5 = load_jsonl(DIR / "monot5_scores_test.jsonl")
    qwen = load_jsonl(DIR / "qwen_scores_test.jsonl")
    qrels = load_qrels(QRELS_PATH)

    qids = sorted(set(monot5) & set(qwen))
    entropies = {q: entropy_topk(monot5[q]) for q in qids}

    e_vals = sorted(entropies.values())
    print(f"#queries: {len(qids)}")
    print(f"entropy stats: min={e_vals[0]:.4f} med={e_vals[len(e_vals)//2]:.4f} "
          f"max={e_vals[-1]:.4f} (max possible ln{ENTROPY_TOPK}={math.log(ENTROPY_TOPK):.4f})")

    metrics = [nDCG @ 10, MRR @ 10]

    # Baselines
    for name, key, src in [
        ("BM25 only",   "bm25_score",   monot5),
        ("monoT5 only", "monot5_prob",  monot5),
        ("Qwen only",   "qwen_prob",    qwen),
    ]:
        rankings = {q: ranking_from(src[q], key) for q in qids}
        run = to_run(rankings)
        res = ir_measures.calc_aggregate(metrics, qrels, run)
        print(f"{name:14s} nDCG@10={res[nDCG@10]:.4f} MRR@10={res[MRR@10]:.4f}")

    print("\n--- Threshold sweep (route to Qwen if entropy > tau) ---")
    print(f"{'tau':>8} {'%->Qwen':>9} {'nDCG@10':>9} {'MRR@10':>9}")

    # Threshold candidates: span percentiles of entropy
    taus = []
    n = len(e_vals)
    for pct in range(0, 101, 5):
        idx = min(n - 1, int(n * pct / 100))
        taus.append(e_vals[idx])
    taus = sorted(set([round(t, 6) for t in taus] + [0.0, 1e9]))

    results = []
    for tau in taus:
        rankings = {}
        n_qwen = 0
        for q in qids:
            if entropies[q] > tau:
                rankings[q] = ranking_from(qwen[q], "qwen_prob")
                n_qwen += 1
            else:
                rankings[q] = ranking_from(monot5[q], "monot5_prob")
        run = to_run(rankings)
        res = ir_measures.calc_aggregate(metrics, qrels, run)
        pct = 100.0 * n_qwen / len(qids)
        results.append((tau, pct, res[nDCG @ 10], res[MRR @ 10]))
        print(f"{tau:8.4f} {pct:8.1f}% {res[nDCG@10]:9.4f} {res[MRR@10]:9.4f}")

    best = max(results, key=lambda r: r[2])
    print(f"\nBest by nDCG@10: tau={best[0]:.4f}  routed={best[1]:.1f}%  "
          f"nDCG@10={best[2]:.4f}  MRR@10={best[3]:.4f}")
    best_mrr = max(results, key=lambda r: r[3])
    print(f"Best by MRR@10:  tau={best_mrr[0]:.4f}  routed={best_mrr[1]:.1f}%  "
          f"nDCG@10={best_mrr[2]:.4f}  MRR@10={best_mrr[3]:.4f}")


if __name__ == "__main__":
    main()
