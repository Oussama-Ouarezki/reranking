"""
BM25 grid search over k1 × b on BioASQ training set.

Metric: Mean Recall@K = average over queries of (relevant docs retrieved in top-K / total relevant docs for that query).

Reads from  : data/bioasq/processed/   (corpus, queries, qrels)
Writes to   : data/bioasq/bm25_doc/images/

Outputs:
  grid_search_recall_train.tsv
  grid_heatmap_recall_at<K>_train.png
  grid_line_recall_at<K>_train.png

Usage:
    python data/bioasq/bm25_doc/grid_search_bm25_recall_train.py
    python data/bioasq/bm25_doc/grid_search_bm25_recall_train.py --cutoff 100
"""

import json
import re
import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── paths ─────────────────────────────────────────────────────────────────────
BASE    = Path('/home/oussama/Desktop/reranking_project')
DOC_DIR = BASE / 'data' / 'bioasq' / 'processed'
OUT_DIR = BASE / 'data' / 'bioasq' / 'bm25_doc' / 'images'

ANSERINI_DEF = {'k1': 0.9, 'b': 0.4}

K1_VALUES = [0.5, 0.7, 0.9, 1.1, 1.2, 1.5, 1.7, 2.0]
B_VALUES  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]


# ── tokeniser ─────────────────────────────────────────────────────────────────
def tokenize(text):
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


# ── data loaders ──────────────────────────────────────────────────────────────
def load_corpus(path):
    ids, texts = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc['_id'])
            texts.append(doc['text'])
    return ids, texts


def load_queries(path):
    qids, qtexts = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qids.append(q['_id'])
            qtexts.append(q['text'])
    return qids, qtexts


def load_qrels(path):
    qrels = defaultdict(set)
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


# ── sparse BM25 index ─────────────────────────────────────────────────────────
class SparseBM25Index:
    def __init__(self, corpus_texts, query_texts):
        print('  Building vocabulary …')
        t0 = time.time()

        tok_corpus  = [tokenize(t) for t in corpus_texts]
        tok_queries = [tokenize(t) for t in query_texts]

        vocab = {term: i for i, term in enumerate(
            sorted({t for doc in tok_corpus for t in doc})
        )}
        V = len(vocab)
        D = len(tok_corpus)
        Q = len(tok_queries)
        print(f'  Vocab: {V:,} terms | Docs: {D:,} | Queries: {Q:,}')

        rows, cols, data = [], [], []
        dl = np.zeros(D, dtype=np.float32)
        df = np.zeros(V, dtype=np.int32)

        for d_idx, tokens in enumerate(tok_corpus):
            dl[d_idx] = len(tokens)
            counts = defaultdict(int)
            for t in tokens:
                if t in vocab:
                    counts[vocab[t]] += 1
            for t_idx, cnt in counts.items():
                rows.append(d_idx)
                cols.append(t_idx)
                data.append(cnt)
                df[t_idx] += 1

        self.tf_matrix = sp.csr_matrix(
            (np.array(data, dtype=np.float32), (rows, cols)), shape=(D, V)
        )
        self.dl    = dl
        self.avgdl = float(dl.mean())
        self.N     = D
        self.idf   = np.log((D - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

        q_rows, q_cols, q_data = [], [], []
        for q_idx, tokens in enumerate(tok_queries):
            counts = defaultdict(int)
            for t in tokens:
                if t in vocab:
                    counts[vocab[t]] += 1
            for t_idx, cnt in counts.items():
                q_rows.append(q_idx)
                q_cols.append(t_idx)
                q_data.append(cnt)

        self.q_matrix = sp.csr_matrix(
            (np.array(q_data, dtype=np.float32), (q_rows, q_cols)), shape=(Q, V)
        )
        print(f'  Index built in {time.time() - t0:.1f}s  (avgdl={self.avgdl:.1f} tokens)')

    def build_doc_weights(self, k1, b):
        norm = (1 - b + b * self.dl / self.avgdl).astype(np.float32)
        ri, ci = self.tf_matrix.nonzero()
        tf = self.tf_matrix.data
        w  = tf * (k1 + 1) / (tf + k1 * norm[ri]) * self.idf[ci]
        return sp.csr_matrix((w, (ri, ci)), shape=self.tf_matrix.shape)

    def score_batched(self, k1, b, cutoff, batch_size=256):
        W   = self.build_doc_weights(k1, b).T.tocsr()
        Q   = self.q_matrix.shape[0]
        top = np.empty((Q, cutoff), dtype=np.int32)
        for s in range(0, Q, batch_size):
            batch  = self.q_matrix[s : s + batch_size]
            scores = (batch @ W).toarray().astype(np.float32)
            top[s : s + len(scores)] = np.argpartition(
                scores, -cutoff, axis=1
            )[:, -cutoff:]
        return top


# ── metric ────────────────────────────────────────────────────────────────────
def mean_recall_at_k(top_indices, qids, corpus_ids, qrels):
    """
    For each query: hits / len(gold), averaged over all queries with ≥1 relevant doc.
    hits = relevant docs retrieved in top-K
    len(gold) = total relevant docs for that query
    """
    did_to_idx = {did: i for i, did in enumerate(corpus_ids)}
    recall_sum = 0.0
    n = 0

    for q_i, qid in enumerate(qids):
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        retrieved = set(top_indices[q_i].tolist())
        hits = sum(1 for did in gold if did in did_to_idx and did_to_idx[did] in retrieved)
        recall_sum += hits / len(gold)

    return 100 * recall_sum / n


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_heatmap(matrix, cutoff, out_path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.1f', linewidths=0.5,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': f'Mean Recall@{cutoff} (%)'})

    best_loc = np.unravel_index(matrix.values.argmax(), matrix.shape)
    ax.add_patch(plt.Rectangle(
        (best_loc[1], best_loc[0]), 1, 1,
        fill=False, edgecolor='#2c3e50', lw=3, label='Best'
    ))

    if ANSERINI_DEF['k1'] in matrix.index and ANSERINI_DEF['b'] in matrix.columns:
        r = list(matrix.index).index(ANSERINI_DEF['k1'])
        c = list(matrix.columns).index(ANSERINI_DEF['b'])
        ax.add_patch(plt.Rectangle(
            (c, r), 1, 1, fill=False, edgecolor='#27ae60', lw=2,
            linestyle='--', label=f'Anserini (k1={ANSERINI_DEF["k1"]}, b={ANSERINI_DEF["b"]})'
        ))

    best_k1  = matrix.index[best_loc[0]]
    best_b   = matrix.columns[best_loc[1]]
    best_val = matrix.values.max()
    ax.set_title(
        f'BM25 Grid Search — Mean Recall@{cutoff} [Training set]\n'
        f'Best: k1={best_k1}, b={best_b} → {best_val:.1f}%',
        fontsize=13, pad=12
    )
    ax.set_xlabel('b', fontsize=11)
    ax.set_ylabel('k1', fontsize=11)
    ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {out_path}')


def plot_line(df, cutoff, out_path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    col = f'mean_recall@{cutoff}'
    fig, ax = plt.subplots(figsize=(9, 5))

    for k1_val in sorted(df['k1'].unique()):
        sub = df[df['k1'] == k1_val].sort_values('b')
        ax.plot(sub['b'], sub[col], marker='o', label=f'k1={k1_val}')

    ax.axvline(ANSERINI_DEF['b'], color='#27ae60', linestyle=':', lw=1.5,
               label=f'Anserini b={ANSERINI_DEF["b"]}')
    ax.set_title(f'BM25 Grid Search — Mean Recall@{cutoff} [Training set]', fontsize=12)
    ax.set_xlabel('b', fontsize=10)
    ax.set_ylabel(f'Mean Recall@{cutoff} (%)', fontsize=10)
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=int, default=100)
    args = parser.parse_args()
    cutoff = args.cutoff

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading corpus …')
    corpus_ids, corpus_texts = load_corpus(DOC_DIR / 'corpus.jsonl')
    print(f'  {len(corpus_ids):,} documents')

    print('Loading queries …')
    qids, qtexts = load_queries(DOC_DIR / 'queries.jsonl')
    print(f'  {len(qids):,} queries')

    print('Loading qrels …')
    qrels = load_qrels(DOC_DIR / 'qrels.tsv')
    covered = sum(1 for qid in qids if qrels.get(qid))
    print(f'  {covered:,} queries with ≥1 relevant doc  |  '
          f'{sum(len(v) for v in qrels.values()):,} pairs')

    print('\nBuilding sparse BM25 index (one-time) …')
    index = SparseBM25Index(corpus_texts, qtexts)

    total = len(K1_VALUES) * len(B_VALUES)
    print(f'\nGrid search: {len(K1_VALUES)} k1 × {len(B_VALUES)} b = {total} combos  '
          f'(cutoff @{cutoff})\n')

    col = f'mean_recall@{cutoff}'
    rows = []
    t_global = time.time()

    for i, k1 in enumerate(K1_VALUES):
        for j, b in enumerate(B_VALUES):
            combo = i * len(B_VALUES) + j + 1
            t0 = time.time()
            top_idx = index.score_batched(k1, b, cutoff)
            recall  = mean_recall_at_k(top_idx, qids, corpus_ids, qrels)
            rows.append({'k1': k1, 'b': b, col: recall})
            print(f'  [{combo:>3}/{total}]  k1={k1:.1f}  b={b:.2f}  '
                  f'mean_recall@{cutoff}={recall:.2f}%  ({time.time()-t0:.1f}s)')

    df   = pd.DataFrame(rows)
    best = df.loc[df[col].idxmax()]
    ref  = df[(df['k1'] == ANSERINI_DEF['k1']) & (df['b'] == ANSERINI_DEF['b'])].iloc[0]

    print(f'\n{"="*55}')
    print(f'  Total time       : {time.time()-t_global:.0f}s')
    print(f'  {"Setting":<30} Mean Recall@{cutoff}')
    print(f'  {"─"*50}')
    print(f'  {"Best":<30} {best[col]:.2f}%  (k1={best["k1"]}, b={best["b"]})')
    print(f'  {"Anserini defaults":<30} {ref[col]:.2f}%  '
          f'(k1={ANSERINI_DEF["k1"]}, b={ANSERINI_DEF["b"]})')
    print(f'{"="*55}')

    tsv_path = BASE / 'data' / 'bioasq' / 'bm25_doc' / 'grid_search_recall_train.tsv'
    df.to_csv(tsv_path, sep='\t', index=False, float_format='%.4f')
    print(f'\n  Results → {tsv_path}')

    print('\nGenerating plots …')
    recall_m = df.pivot(index='k1', columns='b', values=col)
    plot_heatmap(recall_m, cutoff, OUT_DIR / f'grid_heatmap_recall_at{cutoff}_train.png')
    plot_line(df, cutoff, OUT_DIR / f'grid_line_recall_at{cutoff}_train.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
