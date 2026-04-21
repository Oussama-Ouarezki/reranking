"""
BM25 retrieval over BioASQ snippet corpus (Anserini defaults: k1=0.9, b=0.4).

Reads from  : data/bioasq/processed/snippets/
Writes to   : data/bioasq/bm25/

Outputs:
  run_bm25_top100.txt          — TREC-format run file (top 100 per query)
  recall_at_k.png              — % queries with ≥1 gold snippet at each cutoff
  per_query_recall_dist.png    — histogram of per-query Recall@20

Usage:
    python data/bioasq/bm25/retrieve_bm25.py
    python data/bioasq/bm25/retrieve_bm25.py --k1 0.9 --b 0.4 --top-k 100
"""

import json
import re
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from rank_bm25 import BM25Okapi

# ── paths ─────────────────────────────────────────────────────────────────────

BASE       = Path('/home/oussama/Desktop/reranking_project')
SNIPPET_DIR = BASE / 'data' / 'bioasq' / 'processed' / 'snippets'
OUT_DIR    = BASE / 'data' / 'bioasq' / 'bm25'


# ── tokeniser (mirrors Anserini WhitespaceAnalyzer-like behaviour) ────────────

def tokenize(text: str) -> list[str]:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


# ── data loaders ──────────────────────────────────────────────────────────────

def load_corpus(path):
    corpus_ids, corpus_texts = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus_ids.append(doc['_id'])
            corpus_texts.append(doc['text'])
    return corpus_ids, corpus_texts


def load_queries(path):
    queries = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']
    return queries


def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            qid, sid, _ = line.strip().split('\t')
            qrels[qid].add(sid)
    return qrels


# ── metrics ───────────────────────────────────────────────────────────────────

def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold snippets found in top-k retrieved."""
    if not gold:
        return 0.0
    hits = sum(1 for sid in retrieved[:k] if sid in gold)
    return hits / len(gold)


def has_hit_at_k(retrieved: list[str], gold: set[str], k: int) -> bool:
    return any(sid in gold for sid in retrieved[:k])


def compute_metrics(results, qrels, cutoffs=(5, 10, 20, 50, 100)):
    """
    results : {qid: [sid, ...]} ranked list
    Returns : per_query_recall_20, recall_at_k_pct dict
    """
    per_query_recall20 = []
    hits_at = {k: 0 for k in cutoffs}
    n = 0

    for qid, ranked in results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        per_query_recall20.append(recall_at_k(ranked, gold, 20))
        for k in cutoffs:
            if has_hit_at_k(ranked, gold, k):
                hits_at[k] += 1

    pct_at_k = {k: 100 * hits_at[k] / n for k in cutoffs}
    return per_query_recall20, pct_at_k, n


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_recall_at_k(pct_at_k, out_path, k1, b):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    cutoffs = list(pct_at_k.keys())
    values  = list(pct_at_k.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(x=cutoffs, y=values, hue=cutoffs,
                       palette='Blues_d', legend=False, ax=ax)

    # highlight @20
    idx20 = cutoffs.index(20) if 20 in cutoffs else None
    if idx20 is not None:
        bars.patches[idx20].set_facecolor('#e74c3c')

    for patch, val in zip(bars.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    ax.set_title(
        f'BM25 — % Queries with ≥1 Gold Snippet Retrieved\n'
        f'(k1={k1}, b={b}  |  highlighted = @20)',
        fontsize=13, pad=12
    )
    ax.set_xlabel('Top-K cutoff', fontsize=11)
    ax.set_ylabel('% Queries with hit', fontsize=11)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {out_path}')


def plot_per_query_recall_dist(per_query_recall20, out_path, k1, b):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    mean_r = np.mean(per_query_recall20) * 100
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.histplot(
        [r * 100 for r in per_query_recall20],
        bins=30, color='steelblue', alpha=0.85, ax=ax
    )
    ax.axvline(mean_r, color='#e74c3c', linewidth=2,
               linestyle='--', label=f'Mean = {mean_r:.1f}%')

    ax.set_title(
        f'Distribution of Per-Query Recall@20 — BM25 (k1={k1}, b={b})',
        fontsize=13, pad=12
    )
    ax.set_xlabel('Recall@20 per query (%)', fontsize=11)
    ax.set_ylabel('Number of queries',       fontsize=11)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {out_path}')


# ── run-file writer ───────────────────────────────────────────────────────────

def write_run_file(results, path, tag='bm25'):
    with open(path, 'w') as f:
        for qid, ranked in results.items():
            for rank, (sid, score) in enumerate(ranked, start=1):
                f.write(f'{qid} Q0 {sid} {rank} {score:.6f} {tag}\n')
    print(f'  Run file saved → {path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k1',    type=float, default=0.9)
    parser.add_argument('--b',     type=float, default=0.4)
    parser.add_argument('--top-k', type=int,   default=100)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    print('Loading corpus...')
    corpus_ids, corpus_texts = load_corpus(SNIPPET_DIR / 'corpus.jsonl')
    print(f'  {len(corpus_ids):,} snippets')

    print('Loading queries...')
    queries = load_queries(SNIPPET_DIR / 'queries.jsonl')
    print(f'  {len(queries):,} queries')

    print('Loading qrels...')
    qrels = load_qrels(SNIPPET_DIR / 'qrels.tsv')

    # ── build BM25 index ──────────────────────────────────────────────────────
    print(f'\nBuilding BM25 index  (k1={args.k1}, b={args.b})...')
    t0 = time.time()
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=args.k1, b=args.b)
    print(f'  Index built in {time.time() - t0:.1f}s')

    # ── retrieve ──────────────────────────────────────────────────────────────
    print(f'\nRetrieving top-{args.top_k} for {len(queries):,} queries...')
    results = {}   # qid → [(sid, score), ...]
    t0 = time.time()

    for i, (qid, qtext) in enumerate(queries.items(), start=1):
        tok_q  = tokenize(qtext)
        scores = bm25.get_scores(tok_q)
        top_idx = np.argsort(scores)[::-1][: args.top_k]
        results[qid] = [(corpus_ids[j], float(scores[j])) for j in top_idx]

        if i % 500 == 0 or i == len(queries):
            print(f'  [{i}/{len(queries)}]  elapsed: {time.time() - t0:.1f}s')

    # ── metrics ───────────────────────────────────────────────────────────────
    ranked_only = {qid: [sid for sid, _ in hits] for qid, hits in results.items()}
    per_query_r20, pct_at_k, n_queries = compute_metrics(ranked_only, qrels)

    print(f'\n{"─"*45}')
    print(f'  BM25  k1={args.k1}  b={args.b}   ({n_queries} queries)')
    print(f'{"─"*45}')
    for k, pct in pct_at_k.items():
        marker = ' ◀' if k == 20 else ''
        print(f'  Recall@{k:<4} {pct:6.2f}%{marker}')
    print(f'  Mean Recall@20 (per-query): {np.mean(per_query_r20)*100:.2f}%')
    print(f'{"─"*45}')

    # ── save run file ─────────────────────────────────────────────────────────
    write_run_file(results, OUT_DIR / 'run_bm25_top100.txt')

    # ── plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating plots...')
    plot_recall_at_k(pct_at_k,       OUT_DIR / 'recall_at_k.png',            args.k1, args.b)
    plot_per_query_recall_dist(per_query_r20, OUT_DIR / 'per_query_recall_dist.png', args.k1, args.b)

    print('\nDone.')


if __name__ == '__main__':
    main()
