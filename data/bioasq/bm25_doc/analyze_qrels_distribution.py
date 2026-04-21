"""
Distribution of relevant documents per query in BioASQ document-level qrels.

Reads from  : data/bioasq/processed/qrels.tsv
Writes to   : data/bioasq/bm25_doc/

Outputs:
  qrels_dist_histogram.png   — how many queries have 1, 2, 3, ... relevant docs
  qrels_dist_cumulative.png  — cumulative % of queries by number of relevant docs

Usage:
    python data/bioasq/bm25_doc/analyze_qrels_distribution.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE    = Path('/home/oussama/Desktop/reranking_project')
DOC_DIR = BASE / 'data' / 'bioasq' / 'raw' / 'Task13BGoldenEnriched'
OUT_DIR = BASE / 'data' / 'bioasq' / 'bm25_doc'


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            qid, did, _ = line.strip().split('\t')
            qrels[qid].add(did)
    return qrels


def load_queries(path):
    qids = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            qids.append(json.loads(line)['_id'])
    return qids


def plot_histogram(counts: list[int], out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    max_count = max(counts)
    bins = list(range(0, max_count + 2))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(counts, bins=bins, discrete=True,
                 color='steelblue', alpha=0.85, ax=ax)

    mean_val = np.mean(counts)
    ax.axvline(mean_val, color='#e74c3c', linewidth=2,
               linestyle='--', label=f'Mean = {mean_val:.1f}')
    ax.axvline(np.median(counts), color='#f39c12', linewidth=2,
               linestyle=':', label=f'Median = {np.median(counts):.0f}')

    ax.set_title('Distribution of Relevant Documents per Query — BioASQ',
                 fontsize=14, pad=12)
    ax.set_xlabel('Number of relevant documents', fontsize=12)
    ax.set_ylabel('Number of queries', fontsize=12)
    ax.set_xticks(range(0, max_count + 1, max(1, max_count // 20)))
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def plot_cumulative(counts: list[int], out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    sorted_counts = np.sort(counts)
    cumulative    = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_counts, cumulative, color='steelblue', linewidth=2)

    for pct, color, ls in [(50, '#f39c12', ':'), (75, '#27ae60', '--'), (90, '#e74c3c', '-.')]:
        val = np.percentile(counts, pct)
        ax.axvline(val, color=color, linewidth=1.5, linestyle=ls,
                   label=f'P{pct} = {val:.0f} docs')
        ax.axhline(pct, color=color, linewidth=0.8, linestyle=ls, alpha=0.4)

    ax.set_title('Cumulative Distribution — Relevant Docs per Query',
                 fontsize=14, pad=12)
    ax.set_xlabel('Number of relevant documents', fontsize=12)
    ax.set_ylabel('% of queries', fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading queries and qrels...')
    qids  = load_queries(DOC_DIR / 'queries.jsonl')
    qrels = load_qrels(DOC_DIR / 'qrels.tsv')

    # count relevant docs per query (0 if query has no qrels entry)
    counts = [len(qrels.get(qid, set())) for qid in qids]
    counts_nonzero = [c for c in counts if c > 0]

    freq = Counter(counts)

    print(f'\n{"─"*45}')
    print(f'  Total queries        : {len(qids):,}')
    print(f'  Queries with 0 docs  : {freq[0]:,}')
    print(f'  Queries with ≥1 doc  : {len(counts_nonzero):,}')
    print(f'{"─"*45}')
    print(f'  {"# docs":<10} {"# queries":<12} {"% queries"}')
    print(f'  {"─"*35}')
    for n_docs in sorted(k for k in freq if k > 0):
        pct = 100 * freq[n_docs] / len(qids)
        print(f'  {n_docs:<10} {freq[n_docs]:<12,} {pct:.1f}%')
    print(f'{"─"*45}')
    print(f'  Mean docs/query  : {np.mean(counts_nonzero):.2f}')
    print(f'  Median           : {np.median(counts_nonzero):.1f}')
    print(f'  Max              : {max(counts_nonzero)}')
    print(f'{"─"*45}')

    print('\nGenerating plots...')
    plot_histogram(counts_nonzero,  OUT_DIR / 'qrels_dist_histogram.png')
    plot_cumulative(counts_nonzero, OUT_DIR / 'qrels_dist_cumulative.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
