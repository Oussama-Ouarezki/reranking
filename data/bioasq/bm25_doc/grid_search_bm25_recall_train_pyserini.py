"""
BM25 grid search over k1 × b on BioASQ training set — full corpus via Pyserini/Lucene.

Metric: Mean Recall@K = average over queries of (relevant docs retrieved in top-K / total relevant docs for that query).

Index   : data/bm25_indexing_full/corpus_full/lucene_index   (257 907 docs)
Queries : data/bioasq/processed/queries.jsonl                (5 389 queries)
Qrels   : data/bioasq/processed/qrels.tsv

Outputs:
  grid_search_recall_train_pyserini.tsv
  grid_heatmap_recall_at{50,100}_train_full.png
  grid_line_recall_at{50,100}_train_full.png

Usage:
    python data/bioasq/bm25_doc/grid_search_bm25_recall_train_pyserini.py
    python data/bioasq/bm25_doc/grid_search_bm25_recall_train_pyserini.py --threads 8
"""

import os

# Must be set before any pyserini import
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyserini.search.lucene import LuceneSearcher

# ── paths ─────────────────────────────────────────────────────────────────────
BASE      = Path("/home/oussama/Desktop/reranking_project")
INDEX_DIR = BASE / "data" / "bm25_indexing_full" / "corpus_full" / "lucene_index"
DATA_DIR  = BASE / "data" / "bioasq" / "processed"
OUT_DIR   = BASE / "data" / "bioasq" / "bm25_doc" / "images"
TSV_PATH  = BASE / "data" / "bioasq" / "bm25_doc" / "grid_search_recall_train_pyserini.tsv"

ANSERINI_DEF = {"k1": 0.9, "b": 0.4}

K1_VALUES = [0.5, 0.7, 0.9, 1.1, 1.2, 1.5, 1.7, 2.0]
B_VALUES  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]

CUTOFFS = (50, 100)
TOP_K   = max(CUTOFFS)


# ── data loaders ──────────────────────────────────────────────────────────────
def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = q["text"]
    return queries


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as f:
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
    return qrels


# ── metric ────────────────────────────────────────────────────────────────────
def mean_recall_at(results: dict[str, list[str]],
                   qrels: dict[str, set[str]],
                   k: int) -> float:
    recall_sum, n = 0.0, 0
    for qid, ranked in results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        hits = sum(1 for did in ranked[:k] if did in gold)
        recall_sum += hits / len(gold)
    return 100 * recall_sum / n


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_heatmap(matrix: pd.DataFrame, cutoff: int, out_path: Path) -> None:
    plt.style.use("ggplot")
    sns.set_theme(style="whitegrid")
    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt=".1f", linewidths=0.5,
                cmap="YlOrRd", ax=ax,
                cbar_kws={"label": f"Mean Recall@{cutoff} (%)"})

    best_loc = np.unravel_index(matrix.values.argmax(), matrix.shape)
    ax.add_patch(plt.Rectangle(
        (best_loc[1], best_loc[0]), 1, 1,
        fill=False, edgecolor="#2c3e50", lw=3, label="Best"
    ))

    if ANSERINI_DEF["k1"] in matrix.index and ANSERINI_DEF["b"] in matrix.columns:
        r = list(matrix.index).index(ANSERINI_DEF["k1"])
        c = list(matrix.columns).index(ANSERINI_DEF["b"])
        ax.add_patch(plt.Rectangle(
            (c, r), 1, 1, fill=False, edgecolor="#27ae60", lw=2,
            linestyle="--",
            label=f'Anserini (k1={ANSERINI_DEF["k1"]}, b={ANSERINI_DEF["b"]})'
        ))

    best_k1  = matrix.index[best_loc[0]]
    best_b   = matrix.columns[best_loc[1]]
    best_val = matrix.values.max()
    ax.set_title(
        f"BM25 Grid Search — Mean Recall@{cutoff} [Training set, full corpus]\n"
        f"Best: k1={best_k1}, b={best_b} → {best_val:.1f}%",
        fontsize=13, pad=12
    )
    ax.set_xlabel("b", fontsize=11)
    ax.set_ylabel("k1", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


def plot_line(df: pd.DataFrame, cutoff: int, out_path: Path) -> None:
    plt.style.use("ggplot")
    sns.set_theme(style="whitegrid")
    plt.style.use("ggplot")

    col = f"mean_recall@{cutoff}"
    fig, ax = plt.subplots(figsize=(9, 5))

    for k1_val in sorted(df["k1"].unique()):
        sub = df[df["k1"] == k1_val].sort_values("b")
        ax.plot(sub["b"], sub[col], marker="o", label=f"k1={k1_val}")

    ax.axvline(ANSERINI_DEF["b"], color="#27ae60", linestyle=":", lw=1.5,
               label=f'Anserini b={ANSERINI_DEF["b"]}')
    ax.set_title(
        f"BM25 Grid Search — Mean Recall@{cutoff} "
        f"[Training set, full corpus 257 907 docs]",
        fontsize=12,
    )
    ax.set_xlabel("b", fontsize=10)
    ax.set_ylabel(f"Mean Recall@{cutoff} (%)", fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8,
                        help="Threads for Lucene batch_search")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Lucene index from {INDEX_DIR} …")
    t0 = time.time()
    searcher = LuceneSearcher(str(INDEX_DIR))
    print(f"  Searcher ready in {time.time() - t0:.2f}s")

    print("\nLoading queries …")
    queries = load_queries(DATA_DIR / "queries.jsonl")
    print(f"  {len(queries):,} queries")

    print("Loading qrels …")
    qrels = load_qrels(DATA_DIR / "qrels.tsv")
    covered = sum(1 for qid in queries if qrels.get(qid))
    print(f"  {covered:,} queries with ≥1 relevant doc  |  "
          f"{sum(len(v) for v in qrels.values()):,} pairs")

    qids_list   = list(queries.keys())
    qtexts_list = [queries[qid] for qid in qids_list]

    total = len(K1_VALUES) * len(B_VALUES)
    print(f"\nGrid search: {len(K1_VALUES)} k1 × {len(B_VALUES)} b "
          f"= {total} combos  (top-k={TOP_K}, threads={args.threads})\n")

    rows = []
    t_global = time.time()

    for i, k1 in enumerate(K1_VALUES):
        for j, b in enumerate(B_VALUES):
            combo = i * len(B_VALUES) + j + 1
            t0 = time.time()

            searcher.set_bm25(k1=k1, b=b)
            batch_hits = searcher.batch_search(
                queries=qtexts_list,
                qids=qids_list,
                k=TOP_K,
                threads=args.threads,
            )
            results = {qid: [hit.docid for hit in hits]
                       for qid, hits in batch_hits.items()}

            row = {"k1": k1, "b": b}
            metrics_str = []
            for k in CUTOFFS:
                r = mean_recall_at(results, qrels, k)
                row[f"mean_recall@{k}"] = r
                metrics_str.append(f"R@{k}={r:.2f}%")

            rows.append(row)
            print(f"  [{combo:>3}/{total}]  k1={k1:.1f}  b={b:.2f}  "
                  f"{'  '.join(metrics_str)}  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(TSV_PATH, sep="\t", index=False, float_format="%.4f")
    print(f"\n  Results → {TSV_PATH}")
    print(f"  Total time: {time.time()-t_global:.0f}s")

    print("\nGenerating plots …")
    for cutoff in CUTOFFS:
        col = f"mean_recall@{cutoff}"
        best = df.loc[df[col].idxmax()]
        ref  = df[(df["k1"] == ANSERINI_DEF["k1"]) &
                  (df["b"]  == ANSERINI_DEF["b"])].iloc[0]
        print(f"\n  Recall@{cutoff}:")
        print(f"    Best              : {best[col]:.2f}%  "
              f"(k1={best['k1']}, b={best['b']})")
        print(f"    Anserini defaults : {ref[col]:.2f}%  "
              f"(k1={ANSERINI_DEF['k1']}, b={ANSERINI_DEF['b']})")

        recall_m = df.pivot(index="k1", columns="b", values=col)
        plot_heatmap(recall_m, cutoff,
                     OUT_DIR / f"grid_heatmap_recall_at{cutoff}_train_full.png")
        plot_line(df, cutoff,
                  OUT_DIR / f"grid_line_recall_at{cutoff}_train_full.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
