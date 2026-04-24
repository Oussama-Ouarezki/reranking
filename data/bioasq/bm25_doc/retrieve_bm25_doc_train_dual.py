"""
BM25 retrieval over BioASQ training set — dual-bar chart.

For each cutoff K in (5, 10, 20, 50, 100) two bars are drawn side by side:
  • Mean Recall@K    — fraction of relevant docs retrieved (out of all gold docs)
  • Mean Precision@K — fraction of retrieved docs that are relevant (hits / K)

Reads from  : data/bioasq/processed/
Writes to   : data/bioasq/bm25_doc/images/recall_precision_at_k_train.png

Usage:
    python data/bioasq/bm25_doc/retrieve_bm25_doc_train_dual.py
    python data/bioasq/bm25_doc/retrieve_bm25_doc_train_dual.py --k1 0.7 --b 0.9
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
BASE     = Path('/home/oussama/Desktop/reranking_project')
DATA_DIR = BASE / 'data' / 'bioasq' / 'processed'
OUT_DIR  = BASE / 'data' / 'bioasq' / 'bm25_doc' / 'images'

CUTOFFS = (5, 10, 20, 50, 100)


# ── tokeniser ─────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list[str]:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


# ── data loaders ──────────────────────────────────────────────────────────────
def load_corpus(path):
    corpus_ids, corpus_texts = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus_ids.append(doc['_id'])
            corpus_texts.append((doc.get('title', '') + ' ' + doc['text']).strip())
    return corpus_ids, corpus_texts


def load_queries(path):
    queries = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']
    return queries


def load_qrels(path):
    qrels: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding='utf-8') as f:
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
def compute_metrics(results, qrels):
    recall_sum    = {k: 0.0 for k in CUTOFFS}
    precision_sum = {k: 0.0 for k in CUTOFFS}
    n = 0
    for qid, ranked in results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        for k in CUTOFFS:
            top_k = ranked[:k]
            hits  = sum(1 for did in top_k if did in gold)
            recall_sum[k]    += hits / len(gold)
            precision_sum[k] += hits / k
    mean_recall    = {k: 100 * recall_sum[k]    / n for k in CUTOFFS}
    mean_precision = {k: 100 * precision_sum[k] / n for k in CUTOFFS}
    return mean_recall, mean_precision, n


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_dual(mean_recall, mean_precision, out_path, k1, b):
    sns.set_theme(style='darkgrid')
    plt.style.use('ggplot')

    cutoffs   = list(CUTOFFS)
    n_groups  = len(cutoffs)
    x         = np.arange(n_groups)
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
        f'BM25 — Mean Recall@K vs Mean Precision@K\n'
        f'(k1={k1}, b={b})  [BioASQ training set]',
        fontsize=13, pad=12
    )
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k1',    type=float, default=0.7)
    parser.add_argument('--b',     type=float, default=0.9)
    parser.add_argument('--top-k', type=int,   default=100)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading corpus …')
    corpus_ids, corpus_texts = load_corpus(DATA_DIR / 'corpus.jsonl')
    print(f'  {len(corpus_ids):,} documents')

    print('Loading queries …')
    queries = load_queries(DATA_DIR / 'queries.jsonl')
    print(f'  {len(queries):,} queries')

    print('Loading qrels …')
    qrels = load_qrels(DATA_DIR / 'qrels.tsv')
    print(f'  {sum(len(v) for v in qrels.values()):,} relevant pairs')

    print(f'\nBuilding BM25 index  (k1={args.k1}, b={args.b}) …')
    t0 = time.time()
    tokenized_corpus = [tokenize(t) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus, k1=args.k1, b=args.b)
    print(f'  Index built in {time.time() - t0:.1f}s')

    print(f'\nRetrieving top-{args.top_k} for {len(queries):,} queries …')
    results = {}
    t0 = time.time()
    for i, (qid, qtext) in enumerate(queries.items(), start=1):
        scores  = bm25.get_scores(tokenize(qtext))
        top_idx = np.argsort(scores)[::-1][: args.top_k]
        results[qid] = [corpus_ids[j] for j in top_idx]
        if i % 500 == 0 or i == len(queries):
            print(f'  [{i}/{len(queries)}]  elapsed: {time.time() - t0:.1f}s')

    mean_recall, mean_precision, n_queries = compute_metrics(results, qrels)

    print(f'\n{"─"*55}')
    print(f'  BM25  k1={args.k1}  b={args.b}   ({n_queries} queries)')
    print(f'{"─"*55}')
    print(f'  {"K":<6}  {"Recall@K":>10}  {"Precision@K":>12}')
    print(f'  {"─"*6}  {"─"*10}  {"─"*12}')
    for k in CUTOFFS:
        print(f'  {k:<6}  {mean_recall[k]:>9.2f}%  {mean_precision[k]:>11.2f}%')
    print(f'{"─"*55}')

    print('\nGenerating plot …')
    plot_dual(mean_recall, mean_precision,
              OUT_DIR / 'recall_precision_at_k_train.png', args.k1, args.b)
    print('\nDone.')


if __name__ == '__main__':
    main()
