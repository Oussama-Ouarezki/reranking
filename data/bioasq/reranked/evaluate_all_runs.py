"""
Evaluate ALL reranked lists found under data/bioasq/ against gold qrels.

Auto-discovers two file formats:
  A) {qid, bm25_order, permutation}  → evaluates the permutation field
  B) {qid, query, top100}            → evaluates top100 (BM25 baseline)

Skips files with fewer than MIN_QUERIES entries (partial experiments).
Prints a table sorted by nDCG@10 and saves a comparison bar chart.

Usage:
    python data/bioasq/reranked/evaluate_all_runs.py
    python data/bioasq/reranked/evaluate_all_runs.py --min_queries 500
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import ir_measures
from ir_measures import nDCG, RR, Recall, P, ScoredDoc, Qrel

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

ROOT       = Path(__file__).resolve().parents[3]
QRELS_FILE = ROOT / "data/bioasq/processed/qrels.tsv"
SEARCH_DIRS = [
    ROOT / "data/bioasq/reranked",
    ROOT / "data/bioasq/bm25_top100",
    ROOT / "data/bioasq/bm25_top100/prompt engineering",
]
OUT_IMG = Path(__file__).parent / "images" / "all_runs_comparison.png"
OUT_IMG.parent.mkdir(parents=True, exist_ok=True)
MIN_QUERIES = 100

METRICS = [nDCG @ 5, nDCG @ 10, nDCG @ 20, RR, Recall @ 10, Recall @ 20, P @ 5, P @ 10]
LABELS  = {
    str(nDCG @ 5):    "nDCG@5",
    str(nDCG @ 10):   "nDCG@10",
    str(nDCG @ 20):   "nDCG@20",
    str(RR):          "MRR",
    str(Recall @ 10): "Recall@10",
    str(Recall @ 20): "Recall@20",
    str(P @ 5):       "P@5",
    str(P @ 10):      "P@10",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_qrels(path: Path) -> list[Qrel]:
    qrels = []
    with path.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qrels.append(Qrel(parts[0], parts[1], int(parts[2])))
    return qrels


def short_name(path: Path) -> str:
    """Derive a readable label from a file path."""
    name = path.stem
    # Remove long common prefixes to keep labels short
    for prefix in ("deepseek_sliding_reranked_", "deepseek_sliding_", "deepseek_reranked_",
                   "deepseek_", "bm25_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.replace("_", " ").replace(" copy", "").strip()
    # Prepend folder hint for disambiguation
    folder = path.parent.name
    if folder not in ("reranked", "bm25_top100"):
        folder = "pe"  # prompt engineering
    return f"[{folder}] {name}"


def detect_format(first_line: dict) -> str | None:
    """Return 'permutation', 'bm25_ids', or None if unrecognised."""
    keys = set(first_line.keys())
    if {"qid", "bm25_order", "permutation"} <= keys:
        return "permutation"
    if {"qid", "top100"} <= keys and isinstance(first_line.get("top100"), list):
        if first_line["top100"] and isinstance(first_line["top100"][0], dict):
            return "bm25_ids"
    return None


def build_run_permutation(path: Path) -> tuple[list[ScoredDoc], int]:
    scored = []
    n_queries = 0
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            perm = d["permutation"]
            n = len(perm)
            for rank, did in enumerate(perm):
                scored.append(ScoredDoc(d["qid"], did, float(n - rank)))
            n_queries += 1
    return scored, n_queries


def build_run_bm25_ids(path: Path) -> tuple[list[ScoredDoc], int]:
    scored = []
    n_queries = 0
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            docs = d["top100"]
            n = len(docs)
            for rank, doc in enumerate(docs):
                scored.append(ScoredDoc(d["qid"], doc["docid"], float(n - rank)))
            n_queries += 1
    return scored, n_queries


def calc_metrics(scored: list[ScoredDoc], qrels: list[Qrel]) -> dict[str, float]:
    return {str(m): round(v, 4)
            for m, v in ir_measures.calc_aggregate(METRICS, qrels, scored).items()}


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_runs(min_queries: int, qrel_qids: set) -> list[tuple[str, list[ScoredDoc], int]]:
    runs = []
    seen_paths = set()

    for search_dir in SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for path in sorted(search_dir.glob("*.jsonl")):
            real = path.resolve()
            if real in seen_paths:
                continue
            seen_paths.add(real)

            try:
                with path.open() as f:
                    first = json.loads(f.readline())
                fmt = detect_format(first)
                if fmt is None:
                    continue

                n_lines = sum(1 for _ in open(path))
                if n_lines < min_queries:
                    print(f"  skip  {path.name:<55} ({n_lines} queries < {min_queries})")
                    continue

                if fmt == "permutation":
                    scored, n_q = build_run_permutation(path)
                else:
                    scored, n_q = build_run_bm25_ids(path)

                # Check overlap with qrels
                run_qids   = {s.query_id for s in scored}
                overlap    = len(run_qids & qrel_qids)
                overlap_pct = overlap / len(run_qids) * 100
                if overlap == 0:
                    print(f"  skip  {path.name:<55} (0 qids match qrels — wrong split?)")
                    continue
                if overlap_pct < 50:
                    print(f"  warn  {path.name:<55} ({overlap_pct:.0f}% qid overlap — partial)")

                label = short_name(path)
                runs.append((label, scored, n_q))
                print(f"  load  {path.name:<55} ({n_q:,} queries, {overlap_pct:.0f}% qrel overlap, fmt={fmt})")

            except Exception as e:
                print(f"  error {path.name}: {e}")

    return runs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(min_queries: int):
    print("Loading qrels …")
    qrels = load_qrels(QRELS_FILE)
    print(f"  {len(qrels):,} qrel entries\n")

    qrel_qids = {q.query_id for q in qrels}
    print("Discovering runs …")
    runs = discover_runs(min_queries, qrel_qids)
    print(f"\n  {len(runs)} runs found\n")

    if not runs:
        print("No runs found — check SEARCH_DIRS or lower --min_queries.")
        return

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("Evaluating …")
    results = []
    for label, scored, n_q in runs:
        scores = calc_metrics(scored, qrels)
        results.append((label, n_q, scores))
        print(f"  {label:<45}  nDCG@10={scores.get(str(nDCG@10), 0):.4f}")

    # ── Sort by nDCG@10 ───────────────────────────────────────────────────────
    key_ndcg10 = str(nDCG @ 10)
    results.sort(key=lambda x: x[2].get(key_ndcg10, 0), reverse=True)

    # ── Print table ───────────────────────────────────────────────────────────
    col_w  = 46
    metric_keys = list(LABELS.keys())
    header_vals = list(LABELS.values())

    sep = "─" * (col_w + 7 + len(metric_keys) * 10)
    print(f"\n{sep}")
    header = f"  {'Run':<{col_w}} {'Queries':>7}  " + "  ".join(f"{h:>8}" for h in header_vals)
    print(header)
    print(f"  {'─'*col_w} {'───────':>7}  " + "  ".join(f"{'────────':>8}" for _ in header_vals))

    for label, n_q, scores in results:
        vals = "  ".join(f"{scores.get(k, float('nan')):>8.4f}" for k in metric_keys)
        print(f"  {label:<{col_w}} {n_q:>7,}  {vals}")

    print(sep)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_metrics = ["nDCG@5", "nDCG@10", "nDCG@20", "MRR", "Recall@10", "P@5"]
    plot_keys    = [k for k, v in LABELS.items() if v in plot_metrics]

    # Limit to top 15 runs by nDCG@10 to keep chart readable
    top_results = results[:15]
    labels_plot = [r[0] for r in top_results]
    n_metrics   = len(plot_metrics)
    n_runs      = len(top_results)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3 * n_metrics, max(4, n_runs * 0.35 + 2)))

    colors = plt.cm.tab20.colors  # up to 20 distinct colours

    for ax, m_key, m_label in zip(axes, plot_keys, plot_metrics):
        vals = [r[2].get(m_key, 0) for r in top_results]
        bars = ax.barh(range(n_runs), vals, color=[colors[i % 20] for i in range(n_runs)])
        ax.set_yticks(range(n_runs))
        ax.set_yticklabels(labels_plot, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel(m_label, fontsize=9)
        ax.set_title(m_label, fontsize=10)
        # Annotate values
        for bar, v in zip(bars, vals):
            ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=6.5)

    plt.suptitle("BioASQ Reranker Comparison (all runs)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_IMG, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {OUT_IMG}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min_queries", type=int, default=MIN_QUERIES)
    args = p.parse_args()
    main(args.min_queries)
