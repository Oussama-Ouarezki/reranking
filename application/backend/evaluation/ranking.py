"""Ranking metrics via ir_measures."""

from ir_measures import nDCG, RR, Recall, P, AP, ScoredDoc, Qrel
import ir_measures

K_VALUES = [1, 5, 10, 20]


def _qrels_to_objects(qrels: dict[str, dict[str, int]]) -> list[Qrel]:
    out: list[Qrel] = []
    for qid, docs in qrels.items():
        for docid, score in docs.items():
            out.append(Qrel(qid, docid, int(score)))
    return out


def per_query_metrics(
    qid: str,
    ranked_docs: list[tuple[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict:
    """Compute nDCG/MRR/P/R @ {1,5,10,20} for a single query.

    Returns dict shaped like RankingMetrics in schemas.py.
    """
    if qid not in qrels or not qrels[qid]:
        return {
            "ndcg_at": {k: 0.0 for k in K_VALUES},
            "mrr_at": {k: 0.0 for k in K_VALUES},
            "p_at": {k: 0.0 for k in K_VALUES},
            "r_at": {k: 0.0 for k in K_VALUES},
            "map_at": {k: 0.0 for k in K_VALUES},
        }

    qrel_objs = [Qrel(qid, docid, int(s)) for docid, s in qrels[qid].items()]
    run = [ScoredDoc(qid, docid, score=float(score)) for docid, score in ranked_docs]

    metrics = []
    for k in K_VALUES:
        metrics.extend([nDCG @ k, RR @ k, P @ k, Recall @ k, AP @ k])

    agg = ir_measures.calc_aggregate(metrics, qrel_objs, run)

    out = {"ndcg_at": {}, "mrr_at": {}, "p_at": {}, "r_at": {}, "map_at": {}}
    for k in K_VALUES:
        out["ndcg_at"][k] = round(float(agg.get(nDCG @ k, 0.0)), 4)
        out["mrr_at"][k] = round(float(agg.get(RR @ k, 0.0)), 4)
        out["p_at"][k] = round(float(agg.get(P @ k, 0.0)), 4)
        out["r_at"][k] = round(float(agg.get(Recall @ k, 0.0)), 4)
        out["map_at"][k] = round(float(agg.get(AP @ k, 0.0)), 4)
    return out


def aggregate_metrics(
    run: list[tuple[str, str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict:
    """Compute aggregate metrics over a full run.

    run: list of (qid, docid, score)
    """
    qrel_objs = _qrels_to_objects(qrels)
    run_objs = [ScoredDoc(qid, docid, score=float(score)) for qid, docid, score in run]

    metrics = []
    for k in K_VALUES:
        metrics.extend([nDCG @ k, RR @ k, P @ k, Recall @ k, AP @ k])

    agg = ir_measures.calc_aggregate(metrics, qrel_objs, run_objs)

    out = {"ndcg_at": {}, "mrr_at": {}, "p_at": {}, "r_at": {}, "map_at": {}}
    for k in K_VALUES:
        out["ndcg_at"][k] = round(float(agg.get(nDCG @ k, 0.0)), 4)
        out["mrr_at"][k] = round(float(agg.get(RR @ k, 0.0)), 4)
        out["p_at"][k] = round(float(agg.get(P @ k, 0.0)), 4)
        out["r_at"][k] = round(float(agg.get(Recall @ k, 0.0)), 4)
        out["map_at"][k] = round(float(agg.get(AP @ k, 0.0)), 4)
    return out
