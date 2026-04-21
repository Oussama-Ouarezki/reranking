"""
Token length distribution analysis for BioASQ test-set document corpus.

Reads from  : data/bioasq/raw/Task13BGoldenEnriched/corpus.jsonl
Writes to   : data/bioasq/bm25_doc/

Outputs:
  token_dist_test_histogram.png     — histogram of token counts per document
  token_dist_test_cumulative.png    — cumulative % of docs by token length
  token_dist_test_boxplot.png       — boxplot with percentile annotations

Usage:
    python data/bioasq/bm25_doc/analyze_token_distribution_test.py
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
DOC_DIR   = BASE / 'data' / 'bioasq' / 'raw' / 'Task13BGoldenEnriched'
OUT_DIR   = BASE / 'data' / 'bioasq' / 'bm25_doc'
TOKENIZER = str(BASE / 'checkpoints' / 'LiT5-Distill-base')


def load_token_lengths(path, tokenizer) -> list[int]:
    lengths = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            text = (doc.get('title', '') + ' ' + doc['text']).strip()
            ids = tokenizer.encode(text, add_special_tokens=False)
            lengths.append(len(ids))
    return lengths


def plot_histogram(lengths: list[int], out_path: Path):
    sns.set_theme(style='darkgrid')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(lengths, bins=60, color='steelblue', alpha=0.85, ax=ax)

    mean_len = np.mean(lengths)
    ax.axvline(mean_len, color='#e74c3c', linewidth=2,
               linestyle='--', label=f'Mean = {mean_len:.0f}')
    ax.axvline(np.median(lengths), color='#f39c12', linewidth=2,
               linestyle=':', label=f'Median = {np.median(lengths):.0f}')

    ax.set_title('Token Length Distribution — BioASQ Test Corpus (13B1–13B4)', fontsize=14, pad=12)
    ax.set_xlabel('Token count per document', fontsize=12)
    ax.set_ylabel('Number of documents', fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def plot_cumulative(lengths: list[int], out_path: Path):
    sns.set_theme(style='darkgrid')
    plt.style.use('ggplot')

    sorted_lens = np.sort(lengths)
    cumulative  = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens) * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sorted_lens, cumulative, color='steelblue', linewidth=2)

    for pct, color, ls in [(50, '#f39c12', ':'), (90, '#e74c3c', '--'), (95, '#8e44ad', '-.')]:
        val = np.percentile(lengths, pct)
        ax.axvline(val, color=color, linewidth=1.5, linestyle=ls,
                   label=f'P{pct} = {val:.0f} tokens')
        ax.axhline(pct, color=color, linewidth=0.8, linestyle=ls, alpha=0.4)

    ax.set_title('Cumulative Distribution of Token Lengths — BioASQ Test Corpus (13B1–13B4)',
                 fontsize=14, pad=12)
    ax.set_xlabel('Token count', fontsize=12)
    ax.set_ylabel('% of documents', fontsize=12)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def plot_boxplot(lengths: list[int], out_path: Path):
    sns.set_theme(style='darkgrid')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(lengths, vert=False, patch_artist=True,
               boxprops=dict(facecolor='steelblue', alpha=0.6),
               medianprops=dict(color='#e74c3c', linewidth=2),
               flierprops=dict(marker='.', alpha=0.3, markersize=3))

    for pct in (25, 50, 75, 90, 95):
        val = np.percentile(lengths, pct)
        ax.axvline(val, color='grey', linewidth=0.8, linestyle='--')
        ax.text(val, 1.38, f'P{pct}\n{val:.0f}', ha='center', va='bottom',
                fontsize=8, color='#2c3e50')

    ax.set_title('Token Length Boxplot — BioASQ Test Corpus (13B1–13B4)', fontsize=14, pad=12)
    ax.set_xlabel('Token count per document', fontsize=12)
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved → {out_path}')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Loading tokenizer from {TOKENIZER} …')
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    corpus_path = DOC_DIR / 'corpus.jsonl'
    print(f'Loading test corpus from {corpus_path} …')
    lengths = load_token_lengths(corpus_path, tokenizer)
    print(f'  {len(lengths):,} documents loaded')

    arr = np.array(lengths)
    print(f'\n{"─"*40}')
    print(f'  Min     : {arr.min():,}')
    print(f'  Max     : {arr.max():,}')
    print(f'  Mean    : {arr.mean():.1f}')
    print(f'  Median  : {np.median(arr):.1f}')
    print(f'  P75     : {np.percentile(arr, 75):.1f}')
    print(f'  P90     : {np.percentile(arr, 90):.1f}')
    print(f'  P95     : {np.percentile(arr, 95):.1f}')
    print(f'  Std dev : {arr.std():.1f}')
    print(f'{"─"*40}')

    print('\nGenerating plots...')
    plot_histogram(lengths,  OUT_DIR / 'token_dist_test_histogram.png')
    plot_cumulative(lengths, OUT_DIR / 'token_dist_test_cumulative.png')
    plot_boxplot(lengths,    OUT_DIR / 'token_dist_test_boxplot.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
