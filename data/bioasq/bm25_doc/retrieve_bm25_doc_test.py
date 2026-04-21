"""
BM25 retrieval over BioASQ Task13BGoldenEnriched test set.

Reads from  : data/bioasq/raw/Task13BGoldenEnriched/   (corpus + 13B1–13B4 batches)
Writes to   : data/bioasq/bm25_doc/images/

Outputs:
  recall_at_k_test.png         — % queries with ≥1 relevant document at each cutoff

Usage:
    python data/bioasq/bm25_doc/retrieve_bm25_doc_test.py
    python data/bioasq/bm25_doc/retrieve_bm25_doc_test.py --k1 0.7 --b 0.9
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
DATA_DIR = BASE / 'data' / 'bioasq' / 'raw' / 'Task13BGoldenEnriched'
OUT_DIR  = BASE / 'data' / 'bioasq' / 'bm25_doc' / 'images'
BATCHES  = ['13B1', '13B2', '13B3', '13B4']


# ── tokeniser (same as training script) ───────────────────────────────────────
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


def load_test_queries_and_qrels():
    """Merge all 4 test batches into a single queries dict and qrels dict."""
    queries: dict[str, str] = {}
    qrels: dict[str, set[str]] = defaultdict(set)

    for batch in BATCHES:
        batch_dir = DATA_DIR / batch
        with open(batch_dir / 'queries.jsonl', encoding='utf-8') as f:
            for line in f:
                q = json.loads(line)
                queries[q['_id']] = q['text']

        with open(batch_dir / 'qrels.tsv', encoding='utf-8') as f:
            next(f)  # skip header
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

    return queries, qrels


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(results, qrels, cutoffs=(5, 10, 20, 50, 100)):
    """Mean Recall@K: fraction of relevant documents retrieved in top-K, averaged over queries."""
    recall_sum = {k: 0.0 for k in cutoffs}
    n = 0
    for qid, ranked in results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        for k in cutoffs:
            hits = sum(1 for did in ranked[:k] if did in gold)
            recall_sum[k] += hits / len(gold)
    mean_recall = {k: 100 * recall_sum[k] / n for k in cutoffs}
    return mean_recall, n


# ── plotting (same style as retrieve_bm25_doc.py) ─────────────────────────────
def plot_recall_at_k(mean_recall, out_path, k1, b):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    cutoffs = list(mean_recall.keys())
    values  = list(mean_recall.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(x=cutoffs, y=values, hue=cutoffs,
                       palette='Blues_d', legend=False, ax=ax)

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
        f'BM25 Document Level — Mean Recall@K (fraction of relevant docs retrieved)\n'
        f'(k1={k1}, b={b}  |  highlighted = @20)  [Test set: 13B1–13B4]',
        fontsize=13, pad=12
    )
    ax.set_xlabel('Top-K cutoff', fontsize=11)
    ax.set_ylabel('Mean Recall@K (%)', fontsize=11)
    ax.set_ylim(0, 105)

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

    print('Loading test queries and qrels (13B1–13B4) …')
    queries, qrels = load_test_queries_and_qrels()
    print(f'  {len(queries):,} queries')
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
        if i % 100 == 0 or i == len(queries):
            print(f'  [{i}/{len(queries)}]  elapsed: {time.time() - t0:.1f}s')

    pct_at_k, n_queries = compute_metrics(results, qrels)

    print(f'\n{"─"*45}')
    print(f'  BM25 (test set)  k1={args.k1}  b={args.b}   ({n_queries} queries)')
    print(f'{"─"*45}')
    for k, pct in pct_at_k.items():
        marker = ' ◀' if k == 20 else ''
        print(f'  Mean Recall@{k:<4} {pct:6.2f}%{marker}')
    print(f'{"─"*45}')

    print('\nGenerating plot …')
    plot_recall_at_k(pct_at_k, OUT_DIR / 'recall_at_k_test.png', args.k1, args.b)
    print('\nDone.')


if __name__ == '__main__':
    main()
