"""
BM25 grid search over k1 × b to maximise Recall@20 on BioASQ snippets.

Uses scipy sparse matrices to score ALL queries in one matrix multiply per
(k1, b) combo — ~15-20× faster than per-query Python loops.

Reads from  : data/bioasq/processed/snippets/
Writes to   : data/bioasq/bm25/

Outputs:
  grid_search_results.tsv
  grid_heatmap_hit_at<N>.png
  grid_heatmap_mean_recall_at<N>.png
  grid_line_at<N>.png

Usage:
    python data/bioasq/bm25/grid_search_bm25.py
    python data/bioasq/bm25/grid_search_bm25.py --cutoff 10
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

BASE        = Path('/home/oussama/Desktop/reranking_project')
SNIPPET_DIR = BASE / 'data' / 'bioasq' / 'processed' / 'snippets'
OUT_DIR     = BASE / 'data' / 'bioasq' / 'bm25'

# ── grid ──────────────────────────────────────────────────────────────────────

K1_VALUES = [0.5, 0.7, 0.9, 1.1, 1.2, 1.5, 1.7, 2.0]
B_VALUES  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]


# ── tokeniser ─────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
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
            qid, sid, _ = line.strip().split('\t')
            qrels[qid].add(sid)
    return qrels


# ── sparse index ──────────────────────────────────────────────────────────────

class SparseBM25Index:
    """
    Builds vocabulary and sparse TF matrix once.
    For each (k1, b), computes BM25 scores for all queries via matrix multiply.
    """

    def __init__(self, corpus_texts: list[str], query_texts: list[str]):
        print('  Building vocabulary...')
        t0 = time.time()

        # tokenise
        tok_corpus  = [tokenize(t) for t in corpus_texts]
        tok_queries = [tokenize(t) for t in query_texts]

        # vocabulary
        vocab = {term: i for i, term in enumerate(
            sorted({t for doc in tok_corpus for t in doc})
        )}
        V = len(vocab)
        D = len(tok_corpus)
        Q = len(tok_queries)
        print(f'  Vocab: {V:,} terms | Docs: {D:,} | Queries: {Q:,}')

        # ── doc TF sparse matrix  (D × V) ────────────────────────────────────
        rows, cols, data = [], [], []
        dl = np.zeros(D, dtype=np.float32)
        df = np.zeros(V, dtype=np.int32)

        for d_idx, tokens in enumerate(tok_corpus):
            dl[d_idx] = len(tokens)
            counts: dict[int, int] = defaultdict(int)
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

        # ── IDF vector  (V,) ─────────────────────────────────────────────────
        self.idf = np.log((D - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

        # ── query TF sparse matrix  (Q × V) ──────────────────────────────────
        q_rows, q_cols, q_data = [], [], []
        for q_idx, tokens in enumerate(tok_queries):
            counts: dict[int, int] = defaultdict(int)
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

        print(f'  Index built in {time.time() - t0:.1f}s')

    def build_doc_weights(self, k1: float, b: float) -> sp.csr_matrix:
        """
        Precompute sparse BM25 doc weight matrix W_idf  (D × V) for given k1/b.
        Same sparsity as TF matrix — memory stays bounded.
        """
        norm = (1 - b + b * self.dl / self.avgdl).astype(np.float32)  # (D,)
        rows_idx, cols_idx = self.tf_matrix.nonzero()
        tf_data   = self.tf_matrix.data
        denom_data = tf_data + k1 * norm[rows_idx]
        w_data     = tf_data * (k1 + 1) / denom_data * self.idf[cols_idx]
        return sp.csr_matrix(
            (w_data, (rows_idx, cols_idx)), shape=self.tf_matrix.shape
        )   # (D × V)

    def score_batched(self, k1: float, b: float,
                      cutoff: int, batch_size: int = 256) -> np.ndarray:
        """
        Returns top-cutoff doc indices per query  (Q × cutoff).
        Processes queries in batches to keep peak memory ~batch_size × D × 4 bytes.
        """
        W = self.build_doc_weights(k1, b)          # (D × V) sparse
        W_t = W.T.tocsr()                           # (V × D) sparse
        Q = self.q_matrix.shape[0]
        top_indices = np.empty((Q, cutoff), dtype=np.int32)

        for start in range(0, Q, batch_size):
            q_batch = self.q_matrix[start : start + batch_size]   # (B × V)
            # (B × D) dense — B=256, D=67k → ~69 MB
            scores_batch = (q_batch @ W_t).toarray().astype(np.float32)
            # argpartition is faster than full sort for top-k
            top_indices[start : start + len(scores_batch)] = np.argpartition(
                scores_batch, -cutoff, axis=1
            )[:, -cutoff:]

        return top_indices   # (Q × cutoff)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(top_indices: np.ndarray, qids: list[str],
                    corpus_ids: list[str], qrels: dict):
    """
    top_indices : (Q × cutoff) int array of retrieved doc indices
    Returns (pct_hit, mean_recall)
    """
    sid_to_idx = {sid: i for i, sid in enumerate(corpus_ids)}
    hits = 0
    recall_sum = 0.0
    n = 0

    for q_i, qid in enumerate(qids):
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        retrieved = set(top_indices[q_i].tolist())
        hit_count = sum(
            1 for sid in gold
            if sid in sid_to_idx and sid_to_idx[sid] in retrieved
        )
        recall_sum += hit_count / len(gold)
        if hit_count > 0:
            hits += 1

    return (100 * hits / n), (100 * recall_sum / n)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(matrix: pd.DataFrame, title: str, label: str, out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.1f', linewidths=0.5,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': label})

    best_loc = np.unravel_index(matrix.values.argmax(), matrix.shape)
    ax.add_patch(plt.Rectangle(
        (best_loc[1], best_loc[0]), 1, 1,
        fill=False, edgecolor='#2c3e50', lw=3
    ))
    best_k1 = matrix.index[best_loc[0]]
    best_b  = matrix.columns[best_loc[1]]
    best_val = matrix.values.max()

    ax.set_title(f'{title}\nBest: k1={best_k1}, b={best_b} → {best_val:.1f}%',
                 fontsize=13, pad=12)
    ax.set_xlabel('b', fontsize=11)
    ax.set_ylabel('k1', fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {out_path}')


def plot_line_comparison(df: pd.DataFrame, cutoff: int, out_path: Path):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in zip(
        axes,
        [f'pct_hit@{cutoff}', f'mean_recall@{cutoff}'],
        [f'% Queries with hit@{cutoff}', f'Mean Recall@{cutoff} (%)']
    ):
        for k1_val in sorted(df['k1'].unique()):
            sub = df[df['k1'] == k1_val].sort_values('b')
            ax.plot(sub['b'], sub[metric], marker='o', label=f'k1={k1_val}')
        ax.set_title(ylabel, fontsize=12)
        ax.set_xlabel('b', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8, ncol=2)

    plt.suptitle(f'BM25 Grid Search — BioASQ Snippets (@{cutoff})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Plot saved → {out_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', type=int, default=20)
    args = parser.parse_args()
    cutoff = args.cutoff

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Loading data...')
    corpus_ids, corpus_texts = load_corpus(SNIPPET_DIR / 'corpus.jsonl')
    qids, qtexts             = load_queries(SNIPPET_DIR / 'queries.jsonl')
    qrels                    = load_qrels(SNIPPET_DIR / 'qrels.tsv')
    print(f'  {len(corpus_ids):,} docs | {len(qids):,} queries')

    print('\nBuilding sparse index (one-time)...')
    index = SparseBM25Index(corpus_texts, qtexts)

    total = len(K1_VALUES) * len(B_VALUES)
    print(f'\nGrid search: {len(K1_VALUES)} k1 × {len(B_VALUES)} b = {total} combos\n')

    rows = []
    t_global = time.time()

    for i, k1 in enumerate(K1_VALUES):
        for j, b in enumerate(B_VALUES):
            combo = i * len(B_VALUES) + j + 1
            t0 = time.time()

            top_idx  = index.score_batched(k1, b, cutoff)              # (Q × cutoff)
            pct_hit, mean_recall = compute_metrics(
                top_idx, qids, corpus_ids, qrels
            )

            rows.append({
                'k1': k1, 'b': b,
                f'pct_hit@{cutoff}': pct_hit,
                f'mean_recall@{cutoff}': mean_recall,
            })
            print(f'  [{combo:>3}/{total}]  k1={k1:.1f}  b={b:.2f}  '
                  f'hit@{cutoff}={pct_hit:.2f}%  '
                  f'recall@{cutoff}={mean_recall:.2f}%  '
                  f'({time.time()-t0:.1f}s)')

    df = pd.DataFrame(rows)
    total_elapsed = time.time() - t_global

    best = df.loc[df[f'pct_hit@{cutoff}'].idxmax()]
    ref  = df[(df['k1'] == 0.9) & (df['b'] == 0.4)].iloc[0]

    print(f'\n{"="*55}')
    print(f'  Total time: {total_elapsed:.0f}s')
    print(f'  Best → k1={best["k1"]}, b={best["b"]}')
    print(f'    % hit@{cutoff}       : {best[f"pct_hit@{cutoff}"]:.2f}%')
    print(f'    Mean Recall@{cutoff}  : {best[f"mean_recall@{cutoff}"]:.2f}%')
    print(f'  Anserini defaults (k1=0.9, b=0.4):')
    print(f'    % hit@{cutoff}       : {ref[f"pct_hit@{cutoff}"]:.2f}%')
    print(f'    Mean Recall@{cutoff}  : {ref[f"mean_recall@{cutoff}"]:.2f}%')
    print(f'{"="*55}')

    tsv_path = OUT_DIR / 'grid_search_results.tsv'
    df.to_csv(tsv_path, sep='\t', index=False, float_format='%.4f')
    print(f'\n  Results → {tsv_path}')

    print('\nGenerating plots...')
    hit_matrix    = df.pivot(index='k1', columns='b', values=f'pct_hit@{cutoff}')
    recall_matrix = df.pivot(index='k1', columns='b', values=f'mean_recall@{cutoff}')

    plot_heatmap(hit_matrix,
                 f'BM25 Grid Search — % Queries with ≥1 Gold Snippet @{cutoff}',
                 f'% hit@{cutoff}',
                 OUT_DIR / f'grid_heatmap_hit_at{cutoff}.png')
    plot_heatmap(recall_matrix,
                 f'BM25 Grid Search — Mean Per-Query Recall@{cutoff}',
                 f'Mean Recall@{cutoff} (%)',
                 OUT_DIR / f'grid_heatmap_mean_recall_at{cutoff}.png')
    plot_line_comparison(df, cutoff, OUT_DIR / f'grid_line_at{cutoff}.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
