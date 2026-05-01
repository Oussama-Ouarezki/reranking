"""
Compare BM25 vs DeepSeek sliding-window reranker on nDCG@10 and nDCG@20.

Reads:
  data/bioasq/processed/qrels.tsv
  data/bioasq/processed/queries.jsonl        (must have 'type' field)
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl

Outputs:
  data/bioasq/reranked/per_query_results.txt  — one row per query (all types mixed)
  data/bioasq/reranked/per_type_results.txt   — per-query rows grouped by type + per-type averages

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/reranked/compare_bm25_vs_deepseek.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

BASE        = Path(__file__).resolve().parents[3]
QRELS       = BASE / "data/bioasq/processed/qrels.tsv"
QUERIES     = BASE / "data/bioasq/processed/queries.jsonl"
BM25_FILE   = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
DS_FILE     = BASE / "data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl"
OUT_ALL     = BASE / "data/bioasq/reranked/per_query_results.txt"
OUT_TYPE    = BASE / "data/bioasq/reranked/per_type_results.txt"


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            if int(score) > 0:
                qrels[qid].add(did)
    return dict(qrels)


def load_query_types(path: Path) -> dict[str, str]:
    types: dict[str, str] = {}
    with path.open() as f:
        for line in f:
            q = json.loads(line)
            if "type" in q:
                types[q["_id"]] = q["type"]
    return types


def load_bm25(path: Path, top_n: int = 50) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            results[r["qid"]] = [h["docid"] for h in r["top100"][:top_n]]
    return results


def load_reranked(path: Path) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            ranked = r.get("permutation") or r.get("reranked") or []
            results[r["qid"]] = ranked
    return results


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(ranked[:k])
        if d in gold
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


def format_row(label: str, b10: float, d10: float, b20: float, d20: float, n_gold: float) -> str:
    delta10 = d10 - b10
    delta20 = d20 - b20
    gold_str = f"{n_gold:.1f}" if isinstance(n_gold, float) and not n_gold.is_integer() else f"{int(n_gold)}"
    return (
        f"{label:<15}  "
        f"{b10:>7.1f}%  {d10:>6.1f}%  "
        f"{'+' if delta10 >= 0 else ''}{delta10:>5.1f}%  "
        f"{b20:>7.1f}%  {d20:>6.1f}%  "
        f"{'+' if delta20 >= 0 else ''}{delta20:>5.1f}%  "
        f"{gold_str:>6}"
    )


HEADER = (
    f"{'qid':<15}  "
    f"{'BM25@10':>8}  {'DS@10':>7}  {'Δ@10':>7}  "
    f"{'BM25@20':>8}  {'DS@20':>7}  {'Δ@20':>7}  "
    f"{'#gold':>6}"
)
SEP = "─" * len(HEADER)


def build_table(rows: list[tuple], n_total: int) -> str:
    """rows: list of (qid, b10, ds10, b20, ds20, n_gold)"""
    lines = [SEP, HEADER, SEP]

    sums = defaultdict(float)
    for label, b10, ds10, b20, ds20, n_gold in rows:
        lines.append(format_row(label, b10, ds10, b20, ds20, n_gold))
        sums["b10"] += b10
        sums["ds10"] += ds10
        sums["b20"] += b20
        sums["ds20"] += ds20
        sums["gold"] += n_gold

    n = n_total
    avgs = {k: v / n for k, v in sums.items()}
    lines.append(SEP)
    lines.append(format_row(f"AVERAGE ({n}q)", avgs["b10"], avgs["ds10"], avgs["b20"], avgs["ds20"], avgs["gold"]))
    lines.append(SEP)
    return "\n".join(lines) + "\n"


def main() -> None:
    qrels      = load_qrels(QRELS)
    qtypes     = load_query_types(QUERIES)
    bm25       = load_bm25(BM25_FILE)
    reranked   = load_reranked(DS_FILE)

    ks = [10, 20]

    # Compute per-query scores
    per_query: list[tuple] = []           # (qid, b10, ds10, b20, ds20)
    by_type: dict[str, list[tuple]] = defaultdict(list)

    for qid in reranked:
        gold = qrels.get(qid, set())
        if not gold:
            continue
        bm25_ranked = bm25.get(qid, [])
        ds_ranked   = reranked.get(qid, [])

        scores = {}
        for k in ks:
            scores[f"bm25_{k}"] = ndcg_at_k(bm25_ranked, gold, k) * 100
            scores[f"ds_{k}"]   = ndcg_at_k(ds_ranked,   gold, k) * 100

        row = (qid, scores["bm25_10"], scores["ds_10"], scores["bm25_20"], scores["ds_20"], len(gold))
        per_query.append(row)
        qtype = qtypes.get(qid, "unknown")
        by_type[qtype].append(row)

    # --- Output 1: all queries ---
    all_table = build_table(per_query, len(per_query))
    print(all_table)
    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    OUT_ALL.write_text(all_table)
    print(f"Saved → {OUT_ALL}")

    # --- Output 2: per-type breakdown ---
    type_lines: list[str] = []

    for qtype in sorted(by_type.keys()):
        rows = by_type[qtype]
        type_lines.append(f"\n{'═' * len(HEADER)}")
        type_lines.append(f"TYPE: {qtype.upper()}  ({len(rows)} queries)")
        type_lines.append(SEP)
        type_lines.append(HEADER)
        type_lines.append(SEP)

        ts = defaultdict(float)
        for qid, b10, ds10, b20, ds20, n_gold in rows:
            type_lines.append(format_row(qid, b10, ds10, b20, ds20, n_gold))
            ts["b10"] += b10
            ts["ds10"] += ds10
            ts["b20"] += b20
            ts["ds20"] += ds20
            ts["gold"] += n_gold

        n = len(rows)
        type_lines.append(SEP)
        type_lines.append(
            format_row(f"AVG {qtype} ({n}q)", ts["b10"] / n, ts["ds10"] / n, ts["b20"] / n, ts["ds20"] / n, ts["gold"] / n)
        )
        type_lines.append(SEP)

    # Grand average across all types
    n_all = len(per_query)
    grand_b10   = sum(r[1] for r in per_query) / n_all
    grand_ds10  = sum(r[2] for r in per_query) / n_all
    grand_b20   = sum(r[3] for r in per_query) / n_all
    grand_ds20  = sum(r[4] for r in per_query) / n_all
    grand_gold  = sum(r[5] for r in per_query) / n_all

    type_lines.append(f"\n{'═' * len(HEADER)}")
    type_lines.append(f"OVERALL AVERAGE  ({n_all} queries)")
    type_lines.append(SEP)
    type_lines.append(format_row(f"AVERAGE ({n_all}q)", grand_b10, grand_ds10, grand_b20, grand_ds20, grand_gold))
    type_lines.append(SEP)

    type_output = "\n".join(type_lines) + "\n"
    print(type_output)
    OUT_TYPE.write_text(type_output)
    print(f"Saved → {OUT_TYPE}")


if __name__ == "__main__":
    main()
