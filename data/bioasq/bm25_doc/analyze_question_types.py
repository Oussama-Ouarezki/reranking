"""
BioASQ question type analysis — relevant document distribution per type.

Reads from  : data/bioasq/raw/training13b.json
              data/bioasq/processed/corpus.jsonl  (to filter to known PMIDs)
Writes to   : data/bioasq/bm25_doc/

Outputs:
  qtype_counts.png              — bar chart: number of queries per type
  qtype_docs_histogram.png      — per-type histogram of relevant docs per query
  qtype_docs_boxplot.png        — side-by-side boxplot across types

Usage:
    python data/bioasq/bm25_doc/analyze_question_types.py
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

BASE     = Path('/home/oussama/Desktop/reranking_project')
RAW_FILE = BASE / 'data' / 'bioasq' / 'raw' / 'training13b.json'
DOC_DIR  = BASE / 'data' / 'bioasq' / 'processed'
OUT_DIR  = BASE / 'data' / 'bioasq' / 'bm25_doc'

TYPE_COLORS = {
    'yesno':   '#3498db',
    'factoid': '#e74c3c',
    'list':    '#2ecc71',
    'summary': '#f39c12',
}


def extract_pmid(url: str) -> str | None:
    m = re.search(r'pubmed/(\d+)', url or '')
    return m.group(1) if m else None


def load_corpus_pmids(path) -> set:
    pmids = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            pmids.add(json.loads(line)['_id'])
    return pmids


def load_data(raw_file, valid_pmids: set):
    """Returns list of dicts: {qid, type, n_docs}"""
    with open(raw_file, encoding='utf-8') as f:
        data = json.load(f)

    records = []
    for q in data['questions']:
        pmids = {extract_pmid(u) for u in q.get('documents', [])}
        pmids = {p for p in pmids if p and p in valid_pmids}
        records.append({
            'qid':    q['id'],
            'type':   q.get('type', 'unknown'),
            'n_docs': len(pmids),
        })
    return records


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_type_counts(records, out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    type_counts = Counter(r['type'] for r in records)
    types  = list(type_counts.keys())
    counts = [type_counts[t] for t in types]
    colors = [TYPE_COLORS.get(t, '#95a5a6') for t in types]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(types, counts, color=colors, alpha=0.85, edgecolor='white')

    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Number of Queries per Question Type — BioASQ', fontsize=14, pad=12)
    ax.set_xlabel('Question type', fontsize=12)
    ax.set_ylabel('Number of queries', fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def plot_docs_histogram(records, out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    types = sorted({r['type'] for r in records})
    fig, axes = plt.subplots(1, len(types), figsize=(5 * len(types), 5), sharey=False)

    for ax, qtype in zip(axes, types):
        counts = [r['n_docs'] for r in records if r['type'] == qtype and r['n_docs'] > 0]
        color  = TYPE_COLORS.get(qtype, '#95a5a6')

        max_c = max(counts) if counts else 1
        bins  = list(range(0, max_c + 2))
        sns.histplot(counts, bins=bins, discrete=True, color=color, alpha=0.85, ax=ax)

        mean_val = np.mean(counts)
        ax.axvline(mean_val, color='#2c3e50', linewidth=2, linestyle='--',
                   label=f'Mean={mean_val:.1f}')

        ax.set_title(f'{qtype.capitalize()}\n(n={len(counts):,})', fontsize=12)
        ax.set_xlabel('# relevant docs', fontsize=10)
        ax.set_ylabel('# queries', fontsize=10)
        ax.legend(fontsize=9)

    plt.suptitle('Relevant Documents per Query — by Question Type',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved → {out_path}')


def plot_docs_boxplot(records, out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    df = pd.DataFrame([r for r in records if r['n_docs'] > 0])

    fig, ax = plt.subplots(figsize=(10, 6))
    types  = sorted(df['type'].unique())
    colors = [TYPE_COLORS.get(t, '#95a5a6') for t in types]

    sns.boxplot(data=df, x='type', y='n_docs', order=types,
                palette={t: TYPE_COLORS.get(t, '#95a5a6') for t in types},
                ax=ax, width=0.5, flierprops=dict(marker='.', alpha=0.4, markersize=4))

    ax.set_title('Distribution of Relevant Documents per Query Type — BioASQ',
                 fontsize=14, pad=12)
    ax.set_xlabel('Question type', fontsize=12)
    ax.set_ylabel('Number of relevant documents', fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading corpus PMIDs...')
    valid_pmids = load_corpus_pmids(DOC_DIR / 'corpus.jsonl')
    print(f'  {len(valid_pmids):,} known PMIDs')

    print('Loading BioASQ training data...')
    records = load_data(RAW_FILE, valid_pmids)
    print(f'  {len(records):,} questions loaded')

    # ── summary table ─────────────────────────────────────────────────────────
    by_type = defaultdict(list)
    for r in records:
        by_type[r['type']].append(r['n_docs'])

    print(f'\n{"─"*60}')
    print(f'  {"Type":<12} {"Queries":>8} {"0-doc":>7} {"Mean":>7} {"Median":>8} {"Max":>6}')
    print(f'  {"─"*56}')
    for qtype in sorted(by_type):
        counts     = by_type[qtype]
        nonzero    = [c for c in counts if c > 0]
        zero_count = counts.count(0)
        mean   = np.mean(nonzero) if nonzero else 0
        median = np.median(nonzero) if nonzero else 0
        mx     = max(nonzero) if nonzero else 0
        print(f'  {qtype:<12} {len(counts):>8,} {zero_count:>7,} {mean:>7.1f} {median:>8.1f} {mx:>6}')
    print(f'{"─"*60}')

    print('\nGenerating plots...')
    plot_type_counts(records,    OUT_DIR / 'qtype_counts.png')
    plot_docs_histogram(records, OUT_DIR / 'qtype_docs_histogram.png')
    plot_docs_boxplot(records,   OUT_DIR / 'qtype_docs_boxplot.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
