"""
BM25 grid search over k1 × b to maximise Recall@20 on BioASQ documents.

Uses scipy sparse matrices to score ALL queries in one matrix multiply per
(k1, b) combo — ~15-20× faster than per-query Python loops.

Reads from  : data/bioasq/processed/   (full PubMed abstracts, ~49k docs)
Writes to   : data/bioasq/bm25_doc/

Outputs:
  grid_search_results.tsv
  grid_heatmap_hit_at<N>.png
  grid_heatmap_mean_recall_at<N>.png
  grid_line_at<N>.png

Usage:
    python data/bioasq/bm25_doc/grid_search_bm25_doc.py
    python data/bioasq/bm25_doc/grid_search_bm25_doc.py --cutoff 10
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
OUT_DIR = BASE / 'data' / 'bioasq' / 'bm25_doc'

ANSERINI_DEF = {'k1': 0.9, 'b': 0.4}

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
        next(f)  # skip header
        for line in f:
            qid, did, _ = line.strip().split('\t')
            qrels[qid].add(did)
    return qrels


# ── sparse BM25 index ─────────────────────────────────────────────────────────

class SparseBM25Index:
    """
    Builds vocabulary and sparse TF matrix once.
    For each (k1, b), computes BM25 scores for all queries via matrix multiply.
    """

    def __init__(self, corpus_texts: list[str], query_texts: list[str]):
        print('  Building vocabulary...')
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

        # doc TF matrix (D × V)
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
        self.idf   = np.log((D - df + 0.5) / (df + 0.5) + 1).astype(np.float32)

        # query TF matrix (Q × V)
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
        print(f'  Index built in {time.time() - t0:.1f}s  '
              f'(avgdl={self.avgdl:.1f} tokens)')

    def build_doc_weights(self, k1: float, b: float) -> sp.csr_matrix:
        norm = (1 - b + b * self.dl / self.avgdl).astype(np.float32)
        ri, ci = self.tf_matrix.nonzero()
        tf = self.tf_matrix.data
        w  = tf * (k1 + 1) / (tf + k1 * norm[ri]) * self.idf[ci]
        return sp.csr_matrix((w, (ri, ci)), shape=self.tf_matrix.shape)

    def score_batched(self, k1: float, b: float,
                      cutoff: int, batch_size: int = 256) -> np.ndarray:
        """Returns top-cutoff doc indices per query  (Q × cutoff)."""
        W   = self.build_doc_weights(k1, b).T.tocsr()   # (V × D)
        Q   = self.q_matrix.shape[0]
        top = np.empty((Q, cutoff), dtype=np.int32)
        for s in range(0, Q, batch_size):
            batch  = self.q_matrix[s : s + batch_size]
            scores = (batch @ W).toarray().astype(np.float32)
            top[s : s + len(scores)] = np.argpartition(
                scores, -cutoff, axis=1
            )[:, -cutoff:]
        return top


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(top_indices: np.ndarray, qids: list[str],
                    corpus_ids: list[str], qrels: dict):
    """
    top_indices : (Q × cutoff) int array of retrieved doc indices
    Returns (pct_hit, mean_recall)
    """
    did_to_idx = {did: i for i, did in enumerate(corpus_ids)}
    hits = recall_sum = n = 0

    for q_i, qid in enumerate(qids):
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        retrieved = set(top_indices[q_i].tolist())
        hit_count = sum(
            1 for did in gold
            if did in did_to_idx and did_to_idx[did] in retrieved
        )
        recall_sum += hit_count / len(gold)
        if hit_count > 0:
            hits += 1

    return (100 * hits / n), (100 * recall_sum / n)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmap(matrix: pd.DataFrame, title: str, label: str, out_path: Path,
                 mark_cells: list[tuple] = None):
    plt.style.use('ggplot')
    sns.set_theme(style='whitegrid')
    plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.1f', linewidths=0.5,
                cmap='YlOrRd', ax=ax, cbar_kws={'label': label})

    best_loc = np.unravel_index(matrix.values.argmax(), matrix.shape)
    ax.add_patch(plt.Rectangle(
        (best_loc[1], best_loc[0]), 1, 1,
        fill=False, edgecolor='#2c3e50', lw=3, label='Best'
    ))

    if mark_cells:
        for label_tag, k1_val, b_val, color in mark_cells:
            if k1_val in matrix.index and b_val in matrix.columns:
                r = list(matrix.index).index(k1_val)
                c = list(matrix.columns).index(b_val)
                ax.add_patch(plt.Rectangle(
                    (c, r), 1, 1, fill=False, edgecolor=color, lw=2,
                    linestyle='--', label=label_tag
                ))

    best_k1  = matrix.index[best_loc[0]]
    best_b   = matrix.columns[best_loc[1]]
    best_val = matrix.values.max()
    ax.set_title(f'{title}\nBest: k1={best_k1}, b={best_b} → {best_val:.1f}%',
                 fontsize=13, pad=12)
    ax.set_xlabel('b', fontsize=11)
    ax.set_ylabel('k1', fontsize=11)
    ax.legend(loc='upper left', fontsize=8, bbox_to_anchor=(1.15, 1))

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
        ax.axvline(ANSERINI_DEF['b'], color='#27ae60', linestyle=':', lw=1.5,
                   label=f'Anserini b={ANSERINI_DEF["b"]}')
        ax.set_title(ylabel, fontsize=12)
        ax.set_xlabel('b', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=7, ncol=2)

    plt.suptitle(f'BM25 Grid Search — BioASQ Document Level (@{cutoff})',
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

    print('Loading document corpus...')
    corpus_ids, corpus_texts = load_corpus(DOC_DIR / 'corpus.jsonl')
    print(f'  {len(corpus_ids):,} documents  |  avglen ≈ '
          f'{sum(len(t.split()) for t in corpus_texts) // len(corpus_texts)} words')

    print('Loading queries...')
    qids, qtexts = load_queries(DOC_DIR / 'queries.jsonl')
    print(f'  {len(qids):,} queries')

    print('Loading document-level qrels...')
    qrels = load_qrels(DOC_DIR / 'qrels.tsv')
    total_pairs = sum(len(v) for v in qrels.values())
    covered = sum(1 for qid in qids if qrels.get(qid))
    print(f'  {covered:,} queries with ≥1 relevant doc  |  {total_pairs:,} pairs')

    print('\nBuilding sparse BM25 index (one-time)...')
    index = SparseBM25Index(corpus_texts, qtexts)

    total = len(K1_VALUES) * len(B_VALUES)
    print(f'\nGrid search: {len(K1_VALUES)} k1 × {len(B_VALUES)} b = {total} combos  '
          f'(cutoff @{cutoff})\n')

    rows = []
    t_global = time.time()

    for i, k1 in enumerate(K1_VALUES):
        for j, b in enumerate(B_VALUES):
            combo = i * len(B_VALUES) + j + 1
            t0 = time.time()
            top_idx = index.score_batched(k1, b, cutoff)
            pct_hit, mean_recall = compute_metrics(top_idx, qids, corpus_ids, qrels)
            rows.append({'k1': k1, 'b': b,
                         f'pct_hit@{cutoff}': pct_hit,
                         f'mean_recall@{cutoff}': mean_recall})
            print(f'  [{combo:>3}/{total}]  k1={k1:.1f}  b={b:.2f}  '
                  f'hit@{cutoff}={pct_hit:.2f}%  '
                  f'recall@{cutoff}={mean_recall:.2f}%  '
                  f'({time.time()-t0:.1f}s)')

    df = pd.DataFrame(rows)

    best = df.loc[df[f'pct_hit@{cutoff}'].idxmax()]
    ref  = df[(df['k1'] == ANSERINI_DEF['k1']) & (df['b'] == ANSERINI_DEF['b'])].iloc[0]

    print(f'\n{"="*60}')
    print(f'  Total time       : {time.time()-t_global:.0f}s')
    print(f'  {"Setting":<30} {"hit@"+str(cutoff):<12} {"recall@"+str(cutoff)}')
    print(f'  {"─"*54}')
    print(f'  {"Best (doc level)":<30} {best[f"pct_hit@{cutoff}"]:<12.2f} {best[f"mean_recall@{cutoff}"]:.2f}%')
    print(f'    k1={best["k1"]}, b={best["b"]}')
    print(f'  {"Anserini defaults":<30} {ref[f"pct_hit@{cutoff}"]:<12.2f} {ref[f"mean_recall@{cutoff}"]:.2f}%')
    print(f'    k1={ANSERINI_DEF["k1"]}, b={ANSERINI_DEF["b"]}')
    print(f'{"="*60}')

    tsv_path = OUT_DIR / 'grid_search_results.tsv'
    df.to_csv(tsv_path, sep='\t', index=False, float_format='%.4f')
    print(f'\n  Results → {tsv_path}')

    print('\nGenerating plots...')
    hit_m    = df.pivot(index='k1', columns='b', values=f'pct_hit@{cutoff}')
    recall_m = df.pivot(index='k1', columns='b', values=f'mean_recall@{cutoff}')

    ref_cells = [
        (f'Anserini (k1={ANSERINI_DEF["k1"]},b={ANSERINI_DEF["b"]})',
         ANSERINI_DEF['k1'], ANSERINI_DEF['b'], '#27ae60'),
    ]

    plot_heatmap(hit_m,
                 f'BM25 Document Level — % Queries with ≥1 Relevant Doc @{cutoff}',
                 f'% hit@{cutoff}',
                 OUT_DIR / f'grid_heatmap_hit_at{cutoff}.png',
                 mark_cells=ref_cells)
    plot_heatmap(recall_m,
                 f'BM25 Document Level — Mean Per-Query Recall@{cutoff}',
                 f'Mean Recall@{cutoff} (%)',
                 OUT_DIR / f'grid_heatmap_mean_recall_at{cutoff}.png',
                 mark_cells=ref_cells)
    plot_line_comparison(df, cutoff, OUT_DIR / f'grid_line_at{cutoff}.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
