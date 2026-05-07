"""
Convert deepseek_sliding_reranked_golden copy.jsonl into the application's
cached-run format so it appears alongside other models in the UI.

Writes to:
  application/cache/runs/deepseek_sliding_window/<timestamp>.json

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/bm25_top100/import_deepseek_to_cache.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ir_measures import nDCG, RR, Recall, P, AP, ScoredDoc, Qrel
import ir_measures

BASE      = Path(__file__).resolve().parents[3]
JSONL     = BASE / "data/bioasq/bm25_top100/deepseek_sliding_reranked_golden copy.jsonl"
QRELS_P   = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
QUERIES_P = BASE / "data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl"
OUT_DIR   = BASE / "application/cache/runs/deepseek_sliding_window"

KS        = [1, 5, 10, 20]
SAVE_TOPN = 20
MODEL     = "deepseek_sliding_window"


def load_qrels(path):
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        header = f.readline()
        if not header.startswith("query-id"):
            f.seek(0)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def load_qtypes(path):
    qtypes: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            q = json.loads(line)
            if q.get("type"):
                qtypes[q["_id"]] = q["type"]
    return qtypes


def per_query_metrics(qid, ranked_docids, qrels):
    if qid not in qrels or not qrels[qid]:
        return None
    n = len(ranked_docids)
    run_objs  = [ScoredDoc(qid, d, score=float(n - i)) for i, d in enumerate(ranked_docids)]
    qrel_objs = [Qrel(qid, d, int(s)) for d, s in qrels[qid].items()]
    measures  = [m @ k for k in KS for m in (nDCG, RR, P, Recall, AP)]
    agg = ir_measures.calc_aggregate(measures, qrel_objs, run_objs)
    return {
        "ndcg_at": {str(k): round(float(agg.get(nDCG    @ k, 0.0)), 4) for k in KS},
        "mrr_at":  {str(k): round(float(agg.get(RR      @ k, 0.0)), 4) for k in KS},
        "p_at":    {str(k): round(float(agg.get(P       @ k, 0.0)), 4) for k in KS},
        "r_at":    {str(k): round(float(agg.get(Recall  @ k, 0.0)), 4) for k in KS},
        "map_at":  {str(k): round(float(agg.get(AP      @ k, 0.0)), 4) for k in KS},
    }


def aggregate_metrics(full_run, qrels):
    """full_run: list of (qid, docid, score)."""
    qrel_objs = [Qrel(qid, d, int(s))
                 for qid, docs in qrels.items()
                 for d, s in docs.items()]
    run_objs  = [ScoredDoc(qid, docid, float(score)) for qid, docid, score in full_run]
    measures  = [m @ k for k in KS for m in (nDCG, RR, P, Recall, AP)]
    agg = ir_measures.calc_aggregate(measures, qrel_objs, run_objs)
    return {
        "ndcg_at": {k: round(float(agg.get(nDCG    @ k, 0.0)), 4) for k in KS},
        "mrr_at":  {k: round(float(agg.get(RR      @ k, 0.0)), 4) for k in KS},
        "p_at":    {k: round(float(agg.get(P       @ k, 0.0)), 4) for k in KS},
        "r_at":    {k: round(float(agg.get(Recall  @ k, 0.0)), 4) for k in KS},
        "map_at":  {k: round(float(agg.get(AP      @ k, 0.0)), 4) for k in KS},
    }


def main():
    print("Loading qrels …")
    qrels = load_qrels(QRELS_P)
    print(f"  {len(qrels)} queries")

    print("Loading query types …")
    qtypes = load_qtypes(QUERIES_P)

    print(f"Reading {JSONL.name} …")
    entries = []
    with JSONL.open() as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"  {len(entries)} queries")

    t0 = time.time()
    per_query: dict[str, dict] = {}
    full_run: list[tuple[str, str, float]] = []

    for entry in entries:
        qid       = entry["qid"]
        ranked    = entry["permutation"]          # reranked order
        top_docids = ranked[:SAVE_TOPN]
        n = len(ranked)

        metrics = per_query_metrics(qid, ranked, qrels)
        for i, docid in enumerate(ranked):
            full_run.append((qid, docid, float(n - i)))

        pq: dict = {"top_docids": top_docids}
        if metrics is not None:
            pq["metrics"] = metrics
        if qid in qtypes:
            pq["qtype"] = qtypes[qid]
        per_query[qid] = pq

    print("Computing aggregate metrics …")
    agg = aggregate_metrics(full_run, qrels)

    ended_at = time.time()
    elapsed  = round(ended_at - t0, 1)
    ts       = datetime.fromtimestamp(ended_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id   = f"{MODEL}_{ts}"

    payload = {
        "model":      MODEL,
        "run_id":     run_id,
        "started_at": t0,
        "ended_at":   ended_at,
        "elapsed_s":  elapsed,
        "config": {
            "save_topn":        SAVE_TOPN,
            "bm25_retrieve_k":  50,
            "n_questions":      len(per_query),
            "sampled_qids":     None,
            "seed":             None,
            "source":           str(JSONL.name),
            "window":           20,
            "step":             10,
        },
        "comment": "DeepSeek-Chat sliding window (w=20/s=10) on BioASQ Golden",
        "aggregate":  agg,
        "per_query":  per_query,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{ts}.json"
    with out_file.open("w") as f:
        json.dump(payload, f)

    print(f"\nSaved → {out_file}")
    print(f"run_id: {run_id}")
    print(f"Queries in run: {len(per_query)}")
    print(f"\nAggregate metrics:")
    for mk, vals in agg.items():
        print(f"  {mk}: {vals}")


if __name__ == "__main__":
    main()
