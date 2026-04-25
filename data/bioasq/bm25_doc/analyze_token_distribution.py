"""
Token length distribution analysis for BioASQ document-level corpus.

Reads from  : data/bioasq/processed/corpus.jsonl
Writes to   : data/bioasq/bm25_doc/

Outputs:
  token_dist_histogram.png     — histogram of token counts per document
  token_dist_cumulative.png    — cumulative % of docs by token length
  token_dist_boxplot.png       — boxplot with percentile annotations

Usage:
    python data/bioasq/bm25_doc/analyze_token_distribution.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer

BASE      = Path('/home/oussama/Desktop/reranking_project')
DOC_DIR   = BASE / 'data' / 'bioasq' / 'processed'
OUT_DIR   = BASE / 'data' / 'bioasq' / 'bm25_doc'
TOKENIZER = str(BASE / 'checkpoints' / 'LiT5-Distill-base')

Q_COLORS = {
    'Q1': '#2ecc71',
    'Q3': '#9b59b6',
}


def load_token_lengths(path, tokenizer) -> list[int]:
    lengths = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids = tokenizer.encode(doc['text'], add_special_tokens=False)
            lengths.append(len(ids))
    return lengths


def _add_quartile_lines(ax, lengths, orientation='vertical'):
    quartiles = {
        'Q1': np.percentile(lengths, 25),
        'Q3': np.percentile(lengths, 75),
    }
    add_line = ax.axvline if orientation == 'vertical' else ax.axhline
    for label, val in quartiles.items():
        add_line(
            val,
            color=Q_COLORS[label],
            linewidth=1.8,
            linestyle='-.',
            label=f'{label} = {val:.0f}',
            zorder=3,
        )
    return quartiles


def _add_variance_text(ax, lengths, x=0.98, y=0.97):
    variance = np.var(lengths)
    std_dev  = np.std(lengths)
    text = f'Variance : {variance:,.0f}\nStd dev  : {std_dev:.1f}'
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10,
        color='#2c3e50',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#bdc3c7', alpha=0.85),
    )


def plot_histogram(lengths: list[int], out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(lengths, bins=60, color='steelblue', alpha=0.85, ax=ax)

    mean_len = np.mean(lengths)
    ax.axvline(mean_len, color='#e74c3c', linewidth=2,
               linestyle='--', label=f'Mean = {mean_len:.0f}')

    _add_quartile_lines(ax, lengths, orientation='vertical')
    _add_variance_text(ax, lengths)

    ax.set_title('Token Length Distribution — BioASQ Document Corpus', fontsize=14, pad=12)
    ax.set_xlabel('Token count per document', fontsize=12)
    ax.set_ylabel('Number of documents', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def plot_cumulative(lengths: list[int], out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    sorted_lens = np.sort(lengths)
    cumulative  = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_lens, cumulative, color='steelblue', linewidth=2)

    for pct, color, ls in [(90, '#e74c3c', '--'), (95, '#8e44ad', '-.')]:
        val = np.percentile(lengths, pct)
        ax.axvline(val, color=color, linewidth=1.5, linestyle=ls,
                   label=f'P{pct} = {val:.0f} tokens')
        ax.axhline(pct, color=color, linewidth=0.8, linestyle=ls, alpha=0.4)

    quartiles = _add_quartile_lines(ax, lengths, orientation='vertical')
    for label, val in quartiles.items():
        pct_level = {'Q1': 25, 'Q3': 75}[label]
        ax.axhline(pct_level, color=Q_COLORS[label],
                   linewidth=0.8, linestyle='-.', alpha=0.4)

    _add_variance_text(ax, lengths)

    ax.set_title('Cumulative Distribution of Token Lengths — BioASQ Documents',
                 fontsize=14, pad=12)
    ax.set_xlabel('Token count', fontsize=12)
    ax.set_ylabel('% of documents', fontsize=12)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def plot_boxplot(lengths: list[int], out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(lengths, vert=False, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.6),
               medianprops=dict(color='#e74c3c', linewidth=2),
               flierprops=dict(marker='.', alpha=0.3, markersize=3))

    for pct in (90, 95):
        val = np.percentile(lengths, pct)
        ax.axvline(val, color='grey', linewidth=0.8, linestyle='--')
        ax.text(val, 1.38, f'P{pct}\n{val:.0f}', ha='center', va='bottom',
                fontsize=8, color='#2c3e50')

    for label, val in zip(
        ['Q1', 'Q3'],
        [np.percentile(lengths, p) for p in (25, 75)],
    ):
        ax.axvline(val, color=Q_COLORS[label], linewidth=1.8,
                   linestyle='-.', label=f'{label} = {val:.0f}', zorder=4)

    _add_variance_text(ax, lengths, x=0.98, y=0.85)

    ax.set_title('Token Length Boxplot — BioASQ Document Corpus', fontsize=14, pad=12)
    ax.set_xlabel('Token count per document', fontsize=12)
    ax.set_yticks([])
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading tokenizer from {TOKENIZER} …')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    print('Loading corpus and tokenising...')
    lengths = load_token_lengths(DOC_DIR / 'corpus.jsonl', tokenizer)
    print(f'  {len(lengths):,} documents loaded')

    arr = np.array(lengths)
    print(f'\n{"─"*40}')
    print(f'  Min      : {arr.min():,}')
    print(f'  Max      : {arr.max():,}')
    print(f'  Mean     : {arr.mean():.1f}')
    print(f'  Q1 (P25) : {np.percentile(arr, 25):.1f}')
    print(f'  Q3 (P75) : {np.percentile(arr, 75):.1f}')
    print(f'  P90      : {np.percentile(arr, 90):.1f}')
    print(f'  P95      : {np.percentile(arr, 95):.1f}')
    print(f'  Variance : {arr.var():,.1f}')
    print(f'  Std dev  : {arr.std():.1f}')
    print(f'{"─"*40}')

    print('\nGenerating plots...')
    plot_histogram(lengths,  OUT_DIR / 'token_dist_histogram.png')
    plot_cumulative(lengths, OUT_DIR / 'token_dist_cumulative.png')
    plot_boxplot(lengths,    OUT_DIR / 'token_dist_boxplot.png')

    print('\nDone.')


if __name__ == '__main__':
    main()