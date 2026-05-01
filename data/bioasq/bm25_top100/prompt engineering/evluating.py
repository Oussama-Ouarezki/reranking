"""
Evaluate BM25 (top-50) vs ALL DeepSeek reranked files on BioASQ.

Metrics: nDCG@K, P@K, R@K, MRR@K, MAP@K  for K in [1,5,10,20,50]
         Per-query nDCG@10 and nDCG@20, grouped by query type.

Reads:
  data/bioasq/processed/qrels.tsv
  data/bioasq/processed/queries.jsonl          (must have 'type' field)
  data/bioasq/bm25_top100/bm25_top100_ids.jsonl
  data/bioasq/bm25_top100/prompt engineering/*.jsonl   (all reranked files)

Writes (one set per reranked file, inside results/):
  data/bioasq/bm25_top100/prompt engineering/results/<name>_average_metrics.txt
  data/bioasq/bm25_top100/prompt engineering/results/<name>_per_query_metrics.txt
  data/bioasq/bm25_top100/prompt engineering/images/<name>_evaluation.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        "data/bioasq/bm25_top100/prompt engineering/evaluate_full.py"
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE        = Path(__file__).resolve().parents[4]
QRELS       = BASE / "data/bioasq/processed/qrels.tsv"
QUERIES     = BASE / "data/bioasq/processed/queries.jsonl"
BM25_FILE   = BASE / "data/bioasq/bm25_top100/bm25_top100_ids.jsonl"
RERANK_DIR  = Path(__file__).resolve().parent   # same folder as this script
OUT_DIR     = RERANK_DIR / "results"
IMG_DIR     = RERANK_DIR / "images"

TOP_N     = 50
NDCG_KS   = [1, 5, 10, 20, 50]
PREC_KS   = [1, 5, 10, 20, 50]
RECALL_KS = [1, 5, 10, 20, 50]
MRR_KS    = [1, 5, 10, 20, 50]
MAP_KS    = [1, 5, 10, 20, 50]


# ── Metrics ───────────────────────────────────────────────────────────────────

def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in gold) / k


def map_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    hits, score = 0, 0.0
    for i, d in enumerate(ranked[:k]):
        if d in gold:
            hits += 1
            score += hits / (i + 1)
    return score / len(gold) if gold else 0.0


def mrr_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return sum(1 for d in ranked[:k] if d in gold) / len(gold) if gold else 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, d in enumerate(ranked[:k])
        if d in gold
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal > 0 else 0.0


# ── Loaders ───────────────────────────────────────────────────────────────────

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


def load_bm25(path: Path, top_n: int = TOP_N) -> dict[str, list[str]]:
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


def find_reranked_files(directory: Path) -> list[Path]:
    """Return all .jsonl files in the directory, sorted alphabetically."""
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {directory}")
    return files


# ── Aggregate Evaluation ──────────────────────────────────────────────────────

def evaluate(results: dict[str, list[str]], qrels: dict[str, set[str]],
             qids: list[str]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    n = 0
    for qid in qids:
        gold = qrels.get(qid, set())
        if not gold:
            continue
        ranked = results.get(qid, [])
        n += 1
        for k in NDCG_KS:
            sums[f"nDCG@{k}"] += ndcg_at_k(ranked, gold, k)
        for k in PREC_KS:
            sums[f"P@{k}"] += precision_at_k(ranked, gold, k)
        for k in RECALL_KS:
            sums[f"R@{k}"] += recall_at_k(ranked, gold, k)
        for k in MRR_KS:
            sums[f"MRR@{k}"] += mrr_at_k(ranked, gold, k)
        for k in MAP_KS:
            sums[f"MAP@{k}"] += map_at_k(ranked, gold, k)
    return {m: 100 * v / n for m, v in sums.items()}


# ── Per-Query Text ────────────────────────────────────────────────────────────

def format_row(label: str, b10: float, d10: float, b20: float, d20: float,
               n_gold: int | float) -> str:
    delta10 = d10 - b10
    delta20 = d20 - b20
    gold_str = (f"{int(n_gold)}" if isinstance(n_gold, float) and n_gold.is_integer()
                else f"{n_gold:.1f}")
    return (
        f"{label:<20}  "
        f"{b10:>7.1f}%  {d10:>6.1f}%  "
        f"{'+' if delta10 >= 0 else ''}{delta10:>5.1f}%  "
        f"{b20:>7.1f}%  {d20:>6.1f}%  "
        f"{'+' if delta20 >= 0 else ''}{delta20:>5.1f}%  "
        f"{gold_str:>6}"
    )


PQHEADER = (
    f"{'qid':<20}  "
    f"{'BM25@10':>8}  {'DS@10':>7}  {'Δ@10':>7}  "
    f"{'BM25@20':>8}  {'DS@20':>7}  {'Δ@20':>7}  "
    f"{'#gold':>6}"
)
PQSEP = "─" * len(PQHEADER)


def build_per_query_text(per_query: list[tuple],
                         by_type: dict[str, list[tuple]],
                         ds_name: str) -> str:
    lines: list[str] = [f"Per-Query nDCG@10 / nDCG@20 Results: {ds_name}\n"]

    # All queries
    lines += ["═" * len(PQHEADER),
              f"ALL QUERIES  ({len(per_query)} queries)",
              PQSEP, PQHEADER, PQSEP]
    sums = defaultdict(float)
    for row in per_query:
        qid, b10, ds10, b20, ds20, n_gold = row
        lines.append(format_row(qid, b10, ds10, b20, ds20, n_gold))
        sums["b10"] += b10; sums["ds10"] += ds10
        sums["b20"] += b20; sums["ds20"] += ds20
        sums["gold"] += n_gold
    n = len(per_query)
    lines += [PQSEP,
              format_row(f"AVERAGE ({n}q)",
                         sums["b10"] / n, sums["ds10"] / n,
                         sums["b20"] / n, sums["ds20"] / n,
                         sums["gold"] / n),
              PQSEP]

    # Per type
    for qtype in sorted(by_type.keys()):
        rows = by_type[qtype]
        lines += ["", "═" * len(PQHEADER),
                  f"TYPE: {qtype.upper()}  ({len(rows)} queries)",
                  PQSEP, PQHEADER, PQSEP]
        ts = defaultdict(float)
        for qid, b10, ds10, b20, ds20, n_gold in rows:
            lines.append(format_row(qid, b10, ds10, b20, ds20, n_gold))
            ts["b10"] += b10; ts["ds10"] += ds10
            ts["b20"] += b20; ts["ds20"] += ds20
            ts["gold"] += n_gold
        nt = len(rows)
        lines += [PQSEP,
                  format_row(f"AVG {qtype} ({nt}q)",
                             ts["b10"] / nt, ts["ds10"] / nt,
                             ts["b20"] / nt, ts["ds20"] / nt,
                             ts["gold"] / nt),
                  PQSEP]

    return "\n".join(lines) + "\n"


# ── Average Metrics Text ──────────────────────────────────────────────────────

def build_average_metrics_text(bm25: dict[str, float], ds: dict[str, float],
                               n_queries: int, ds_name: str) -> str:
    groups = [
        ("nDCG",      [f"nDCG@{k}" for k in NDCG_KS]),
        ("Precision", [f"P@{k}"    for k in PREC_KS]),
        ("Recall",    [f"R@{k}"    for k in RECALL_KS]),
        ("MRR",       [f"MRR@{k}"  for k in MRR_KS]),
        ("MAP",       [f"MAP@{k}"  for k in MAP_KS]),
    ]

    title  = f"BM25 vs {ds_name}\nBioASQ  |  {n_queries} queries  |  top-{TOP_N}\n"
    header = f"  {'Metric':<12}  {'BM25':>8}  {'Reranked':>10}  {'Δ':>8}"
    sep    = "  " + "─" * 46

    lines = [title, header, sep]
    for group_name, metrics in groups:
        lines.append(f"\n  ── {group_name} ──")
        for m in metrics:
            delta = ds[m] - bm25[m]
            sign  = "+" if delta >= 0 else ""
            lines.append(
                f"  {m:<12}  {bm25[m]:>7.2f}%  {ds[m]:>9.2f}%  {sign}{delta:>6.2f}%"
            )

    lines.append(sep)
    return "\n".join(lines) + "\n"


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_panel(ax: plt.Axes, metrics: list[str], bm25: dict[str, float],
                ds: dict[str, float], title: str, ds_label: str,
                show_legend: bool) -> None:
    x = np.arange(len(metrics))
    w = 0.35
    bm25_vals = [bm25[m] for m in metrics]
    ds_vals   = [ds[m]   for m in metrics]

    bars_b = ax.bar(x - w / 2, bm25_vals, w, label=f"BM25 top-{TOP_N}",
                    color="#4C72B0", alpha=0.88)
    bars_d = ax.bar(x + w / 2, ds_vals,   w, label=ds_label,
                    color="#55A868", alpha=0.88)

    for bars, color, vals in [(bars_b, "#4C72B0", bm25_vals),
                               (bars_d, "#55A868", ds_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.4,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, max(max(bm25_vals), max(ds_vals)) * 1.3 + 5)
    ax.set_title(title, fontsize=11, pad=8)
    if show_legend:
        ax.legend(fontsize=9)


def plot(bm25: dict[str, float], ds: dict[str, float],
         n_queries: int, ds_name: str, out_path: Path) -> None:
    panels = [
        ([f"nDCG@{k}" for k in NDCG_KS], "nDCG"),
        ([f"P@{k}"    for k in PREC_KS],  "Precision"),
        ([f"R@{k}"    for k in RECALL_KS], "Recall"),
        ([f"MRR@{k}"  for k in MRR_KS],   "MRR"),
        ([f"MAP@{k}"  for k in MAP_KS],    "MAP"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    fig.suptitle(
        f"BM25 vs {ds_name} — BioASQ  ({n_queries} queries, top-{TOP_N})",
        fontsize=13, y=1.01,
    )

    for ax, (metrics, title), show_legend in zip(
        axes, panels, [True, False, False, False, False]
    ):
        _draw_panel(ax, metrics, bm25, ds, title, ds_name, show_legend)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot saved → {out_path}")


# ── Per-file runner ───────────────────────────────────────────────────────────

def run_one(ds_path: Path, bm25_results: dict[str, list[str]],
            qrels: dict[str, set[str]], qtypes: dict[str, str]) -> None:
    ds_name = ds_path.stem
    print(f"\n{'═' * 60}")
    print(f"  Evaluating: {ds_name}")
    print(f"{'═' * 60}")

    ds_results = load_reranked(ds_path)
    qids = [qid for qid in ds_results if qrels.get(qid)]
    if not qids:
        print("  ⚠  No queries with gold labels found — skipping.")
        return
    print(f"  Queries with gold labels: {len(qids)}")

    # Aggregate metrics
    bm25_scores = evaluate(bm25_results, qrels, qids)
    ds_scores   = evaluate(ds_results,   qrels, qids)

    avg_text = build_average_metrics_text(bm25_scores, ds_scores, len(qids), ds_name)
    print(avg_text)
    avg_out = OUT_DIR / f"{ds_name}_average_metrics.txt"
    avg_out.write_text(avg_text, encoding="utf-8")
    print(f"  Saved → {avg_out}")

    # Per-query metrics
    per_query: list[tuple] = []
    by_type:   dict[str, list[tuple]] = defaultdict(list)

    for qid in qids:
        gold        = qrels[qid]
        bm25_ranked = bm25_results.get(qid, [])
        ds_ranked   = ds_results.get(qid, [])

        b10  = ndcg_at_k(bm25_ranked, gold, 10) * 100
        ds10 = ndcg_at_k(ds_ranked,   gold, 10) * 100
        b20  = ndcg_at_k(bm25_ranked, gold, 20) * 100
        ds20 = ndcg_at_k(ds_ranked,   gold, 20) * 100

        row = (qid, b10, ds10, b20, ds20, len(gold))
        per_query.append(row)
        by_type[qtypes.get(qid, "unknown")].append(row)

    pq_text = build_per_query_text(per_query, by_type, ds_name)
    pq_out  = OUT_DIR / f"{ds_name}_per_query_metrics.txt"
    pq_out.write_text(pq_text, encoding="utf-8")
    print(f"  Saved → {pq_out}")

    # Plot
    img_out = IMG_DIR / f"{ds_name}_evaluation.png"
    plot(bm25_scores, ds_scores, len(qids), ds_name, img_out)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading shared data …")
    qrels        = load_qrels(QRELS)
    qtypes       = load_query_types(QUERIES)
    bm25_results = load_bm25(BM25_FILE)

    reranked_files = find_reranked_files(RERANK_DIR)
    print(f"\nFound {len(reranked_files)} reranked file(s):")
    for f in reranked_files:
        print(f"  {f.name}")

    for ds_path in reranked_files:
        run_one(ds_path, bm25_results, qrels, qtypes)

    print(f"\n{'═' * 60}")
    print(f"All done.  Results written to: {OUT_DIR}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()