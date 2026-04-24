"""
BM25 retrieval over BioASQ training set — Pyserini/Lucene edition — dual-bar chart.

For each cutoff K in (5, 10, 20, 50, 100) two bars are drawn side by side:
  • Mean Recall@K    — fraction of relevant docs retrieved (out of all gold docs)
  • Mean Precision@K — fraction of retrieved docs that are relevant (hits / K)

Reads from  : data/bioasq/processed/   (corpus + queries + qrels)
Index       : data/bm25_indexing/lucene_index/  (built automatically on first run)
Writes to   : data/bioasq/bm25_doc/images/recall_precision_at_k_train_pyserini.png

Usage:
    python data/bioasq/bm25_doc/retrieve_bm25_doc_train_pyserini.py
    python data/bioasq/bm25_doc/retrieve_bm25_doc_train_pyserini.py --k1 0.7 --b 0.9 --threads 8
"""

import os

# Must be set before any pyserini import
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyserini.search.lucene import LuceneSearcher

# ── paths ─────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).resolve().parents[3]
DATA_DIR  = BASE / 'data' / 'bioasq' / 'processed'
INDEX_DIR = BASE / 'data' / 'bm25_indexing' / 'lucene_index'
OUT_DIR   = BASE / 'data' / 'bioasq' / 'bm25_doc' / 'images'

CUTOFFS = (5, 10, 20, 50, 100)


# ── index builder (runs once if index absent) ─────────────────────────────────
def build_index_if_needed(corpus_path: Path, index_dir: Path) -> None:
    if index_dir.exists() and any(index_dir.iterdir()):
        print(f'  Index already exists at {index_dir}')
        return

    print('  Building Pyserini index …')
    tmp_input = index_dir.parent / '_pyserini_input'
    tmp_input.mkdir(parents=True, exist_ok=True)

    with corpus_path.open(encoding='utf-8') as src, \
         (tmp_input / 'docs.jsonl').open('w', encoding='utf-8') as dst:
        for line in src:
            doc = json.loads(line)
            contents = (doc.get('title', '') + ' ' + doc.get('text', '')).strip()
            dst.write(json.dumps({'id': doc['_id'], 'contents': contents},
                                 ensure_ascii=False) + '\n')

    result = subprocess.run([
        sys.executable, '-m', 'pyserini.index.lucene',
        '--collection',  'JsonCollection',
        '--input',       str(tmp_input),
        '--index',       str(index_dir),
        '--generator',   'DefaultLuceneDocumentGenerator',
        '--threads',     '4',
        '--storePositions', '--storeDocvectors', '--storeRaw',
    ], env=os.environ.copy(), capture_output=True, text=True)

    shutil.rmtree(tmp_input)

    if result.returncode != 0:
        raise RuntimeError(f'Pyserini indexing failed:\n{result.stderr}')
    print(f'  Index built → {index_dir}')


# ── data loaders ──────────────────────────────────────────────────────────────
def load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']
    return queries


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            if int(score) > 0:
                qrels[qid].add(did)
    return qrels


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(
    results: dict[str, list[str]],
    qrels: dict[str, set[str]],
) -> tuple[dict[int, float], dict[int, float], int]:
    recall_sum    = {k: 0.0 for k in CUTOFFS}
    precision_sum = {k: 0.0 for k in CUTOFFS}
    n = 0
    for qid, ranked in results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        for k in CUTOFFS:
            hits = sum(1 for did in ranked[:k] if did in gold)
            recall_sum[k]    += hits / len(gold)
            precision_sum[k] += hits / k
    mean_recall    = {k: 100 * recall_sum[k]    / n for k in CUTOFFS}
    mean_precision = {k: 100 * precision_sum[k] / n for k in CUTOFFS}
    return mean_recall, mean_precision, n


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_dual(
    mean_recall: dict[int, float],
    mean_precision: dict[int, float],
    out_path: Path,
    k1: float,
    b: float,
) -> None:
    sns.set_theme(style='darkgrid')
    plt.style.use('ggplot')

    cutoffs   = list(CUTOFFS)
    x         = np.arange(len(cutoffs))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    recall_vals    = [mean_recall[k]    for k in cutoffs]
    precision_vals = [mean_precision[k] for k in cutoffs]

    bars_r = ax.bar(x - bar_width / 2, recall_vals,    bar_width,
                    label='Mean Recall@K',    color='#4C72B0', alpha=0.88)
    bars_p = ax.bar(x + bar_width / 2, precision_vals, bar_width,
                    label='Mean Precision@K', color='#DD8452', alpha=0.88)

    for bar, val in zip(bars_r, recall_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#4C72B0')

    for bar, val in zip(bars_p, precision_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#DD8452')

    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in cutoffs], fontsize=11)
    ax.set_xlabel('Top-K cutoff', fontsize=12)
    ax.set_ylabel('Mean %', fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(
        f'BM25 (Pyserini/Lucene) — Mean Recall@K vs Mean Precision@K\n'
        f'(k1={k1}, b={b})  [BioASQ training set]',
        fontsize=13, pad=12,
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--k1',      type=float, default=0.7)
    parser.add_argument('--b',       type=float, default=0.9)
    parser.add_argument('--top-k',   type=int,   default=100)
    parser.add_argument('--threads', type=int,   default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Checking Lucene index …')
    build_index_if_needed(DATA_DIR / 'corpus.jsonl', INDEX_DIR)

    print(f'\nLoading searcher  (k1={args.k1}, b={args.b}) …')
    t0 = time.time()
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=args.k1, b=args.b)
    print(f'  Ready in {time.time() - t0:.2f}s')

    print('\nLoading queries …')
    queries = load_queries(DATA_DIR / 'queries.jsonl')
    print(f'  {len(queries):,} queries')

    print('Loading qrels …')
    qrels = load_qrels(DATA_DIR / 'qrels.tsv')
    print(f'  {sum(len(v) for v in qrels.values()):,} relevant pairs')

    print(f'\nRetrieving top-{args.top_k} for {len(queries):,} queries '
          f'(threads={args.threads}) …')
    t0 = time.time()
    qids_list   = list(queries.keys())
    qtexts_list = [queries[qid] for qid in qids_list]

    batch_hits = searcher.batch_search(
        queries=qtexts_list,
        qids=qids_list,
        k=args.top_k,
        threads=args.threads,
    )
    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.2f}s  ({elapsed / len(queries) * 1000:.2f} ms/query)')

    results: dict[str, list[str]] = {
        qid: [hit.docid for hit in hits]
        for qid, hits in batch_hits.items()
    }

    mean_recall, mean_precision, n_queries = compute_metrics(results, qrels)

    print(f'\n{"─"*55}')
    print(f'  BM25/Pyserini  k1={args.k1}  b={args.b}   ({n_queries} queries)')
    print(f'{"─"*55}')
    print(f'  {"K":<6}  {"Recall@K":>10}  {"Precision@K":>12}')
    print(f'  {"─"*6}  {"─"*10}  {"─"*12}')
    for k in CUTOFFS:
        print(f'  {k:<6}  {mean_recall[k]:>9.2f}%  {mean_precision[k]:>11.2f}%')
    print(f'{"─"*55}')

    print('\nGenerating plot …')
    plot_dual(
        mean_recall, mean_precision,
        OUT_DIR / 'recall_precision_at_k_train_pyserini.png',
        args.k1, args.b,
    )
    print('\nDone.')


if __name__ == '__main__':
    main()
