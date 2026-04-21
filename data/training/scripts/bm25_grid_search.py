"""
BM25 hyperparameter grid search on the head-truncated BioASQ training corpus.

Grid:
    k1 : [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    b  : [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    → 42 combinations

Speed trick: the inverted index (posting lists + IDF) is built ONCE.
Per query, candidate docs and their TFs are fetched once, then all 42 (k1, b)
combos are scored in a tight numpy inner loop — no BM25Okapi rebuilds.

Saves:
    data/training/images/bm25_grid_recall20.png
    data/training/images/bm25_grid_recall100.png
    data/training/images/bm25_grid_combined.png

Usage:
    python data/training/scripts/bm25_grid_search.py
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────
BASE         = Path('/home/oussama/Desktop/reranking_project')
CORPUS_FILE  = BASE / 'data' / 'training' / 'truncated' / 'corpus.jsonl'
QUERIES_FILE = BASE / 'data' / 'training' / 'truncated' / 'queries.jsonl'
QRELS_FILE   = BASE / 'data' / 'training' / 'truncated' / 'qrels.tsv'
IMG_DIR      = BASE / 'data' / 'training' / 'images'

K1_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
B_VALUES  = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
K_EVAL    = [20, 100]


# ── Fast BM25 (inverted-index, numpy scoring) ─────────────────────────────────

class FastBM25:
    def __init__(self, tokenized_corpus: list[list[str]]):
        N = len(tokenized_corpus)
        self.N = N
        self.dl     = np.array([len(d) for d in tokenized_corpus], dtype=np.float32)
        self.avgdl  = float(self.dl.mean())

        tf_raw: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for doc_idx, tokens in enumerate(tokenized_corpus):
            for tok in tokens:
                tf_raw[tok][doc_idx] += 1

        self.idf: dict[str, float] = {}
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, doc_tfs in tf_raw.items():
            df = len(doc_tfs)
            self.idf[term] = float(np.log((N - df + 0.5) / (df + 0.5) + 1))
            ids = np.fromiter(doc_tfs.keys(), dtype=np.int32, count=df)
            tfs = np.fromiter(doc_tfs.values(), dtype=np.float32, count=df)
            self.postings[term] = (ids, tfs)

    def score_all_params(self,
                         query_tokens: list[str],
                         k1_values: list[float],
                         b_values: list[float]) -> np.ndarray:
        """
        Returns scores[i, j, doc_idx] for k1_values[i], b_values[j].
        Only allocates memory for candidate docs (docs with ≥1 query term).
        """
        # collect per-term posting data for query terms
        term_data: list[tuple[np.ndarray, np.ndarray, float]] = []
        cand_set: set[int] = set()
        for tok in set(query_tokens):
            if tok in self.postings:
                ids, tfs = self.postings[tok]
                idf      = self.idf[tok]
                term_data.append((ids, tfs, idf))
                cand_set.update(ids.tolist())

        out = np.zeros((len(k1_values), len(b_values), self.N), dtype=np.float32)
        if not term_data:
            return out

        for ids, tfs, idf in term_data:
            dl_d = self.dl[ids]
            for i, k1 in enumerate(k1_values):
                tf_num = tfs * (k1 + 1.0)
                for j, b in enumerate(b_values):
                    denom = tfs + k1 * (1.0 - b + b * dl_d / self.avgdl)
                    out[i, j, ids] += idf * tf_num / denom

        return out


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus():
    doc_ids, texts = [], []
    with CORPUS_FILE.open(encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_ids.append(doc['_id'])
            texts.append((doc.get('title', '') + ' ' + doc['text']).strip())
    return doc_ids, texts


def load_queries_and_qrels():
    queries: dict[str, str] = {}
    with QUERIES_FILE.open(encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']

    relevant: dict[str, set[str]] = defaultdict(set)
    with QRELS_FILE.open(encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                qid, doc_id, score = parts
            elif len(parts) == 4:
                qid, _, doc_id, score = parts
            else:
                continue
            if int(score) > 0:
                relevant[qid].add(doc_id)
    return queries, relevant


# ── Grid evaluation ───────────────────────────────────────────────────────────

def run_grid(bm25: FastBM25, doc_ids: list[str],
             queries: dict[str, str],
             relevant: dict[str, set[str]]) -> dict[int, np.ndarray]:
    """
    Returns recall_grid[k] as a (len(K1_VALUES), len(B_VALUES)) array.
    """
    nk1, nb = len(K1_VALUES), len(B_VALUES)
    max_k   = max(K_EVAL)

    # accumulators: sum of recall, count of queries with relevant docs
    recall_sum   = {k: np.zeros((nk1, nb), dtype=np.float64) for k in K_EVAL}
    query_counts = {k: 0 for k in K_EVAL}

    t0    = time.time()
    total = len(queries)

    for q_num, (qid, qtext) in enumerate(queries.items(), 1):
        rel = relevant.get(qid, set())
        if not rel:
            continue

        qtoks  = qtext.lower().split()
        scores = bm25.score_all_params(qtoks, K1_VALUES, B_VALUES)  # (nk1, nb, N)

        # get top-max_k indices for every (k1, b) combo
        # argsort on last axis, take last max_k → shape (nk1, nb, max_k)
        top_idx = np.argpartition(scores, -max_k, axis=2)[:, :, -max_k:]

        for k in K_EVAL:
            # for each (i,j) count how many of the top-k are relevant
            topk = top_idx[:, :, -k:]          # (nk1, nb, k)  — still unsorted but sufficient
            for i in range(nk1):
                for j in range(nb):
                    hits = sum(1 for idx in topk[i, j] if doc_ids[idx] in rel)
                    recall_sum[k][i, j] += hits / len(rel)
            query_counts[k] += 1

        if q_num % 500 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / q_num * (total - q_num)
            print(f'  {q_num:,}/{total:,}  elapsed {elapsed:.0f}s  ETA ~{eta:.0f}s',
                  flush=True)

    return {k: recall_sum[k] / query_counts[k] for k in K_EVAL}


# ── Heatmap plotting ──────────────────────────────────────────────────────────

def plot_heatmap(matrix: np.ndarray, k: int, path: Path, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        matrix,
        xticklabels=[str(b) for b in B_VALUES],
        yticklabels=[str(k1) for k1 in K1_VALUES],
        annot=True, fmt='.4f', cmap='YlGnBu',
        linewidths=0.4, ax=ax,
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel('b', fontsize=12)
    ax.set_ylabel('k1', fontsize=12)
    ax.set_title(f'BM25 Mean Recall@{k} — Grid Search\nBioASQ Head-Truncated Training Set',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved → {path}')


def plot_combined(grids: dict[int, np.ndarray], path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cmaps = ['YlGnBu', 'YlOrRd']
    for ax, (k, cmap) in zip(axes, zip(K_EVAL, cmaps)):
        sns.heatmap(
            grids[k],
            xticklabels=[str(b) for b in B_VALUES],
            yticklabels=[str(k1) for k1 in K1_VALUES],
            annot=True, fmt='.4f', cmap=cmap,
            linewidths=0.4, ax=ax,
        )
        ax.set_xlabel('b', fontsize=11)
        ax.set_ylabel('k1', fontsize=11)
        ax.set_title(f'Recall@{k}', fontsize=12)

    fig.suptitle('BM25 Grid Search — BioASQ Head-Truncated Training Set', fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Saved → {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

print('Loading corpus …')
doc_ids, texts = load_corpus()
print(f'  {len(doc_ids):,} documents')

print('Building inverted index …')
t0 = time.time()
bm25 = FastBM25([t.lower().split() for t in texts])
print(f'  Done in {time.time()-t0:.1f}s  |  vocab={len(bm25.postings):,} terms')

print('Loading queries + qrels …')
queries, relevant = load_queries_and_qrels()
print(f'  {len(queries):,} queries  |  {sum(len(v) for v in relevant.values()):,} pairs')

print(f'\nRunning grid search  ({len(K1_VALUES)} k1 × {len(B_VALUES)} b = '
      f'{len(K1_VALUES)*len(B_VALUES)} combos) …')
t0    = time.time()
grids = run_grid(bm25, doc_ids, queries, relevant)
print(f'Grid search done in {time.time()-t0:.1f}s')

# ── Results summary ───────────────────────────────────────────────────────────
for k in K_EVAL:
    best_i, best_j = np.unravel_index(np.argmax(grids[k]), grids[k].shape)
    print(f'\nRecall@{k}  best: k1={K1_VALUES[best_i]}  b={B_VALUES[best_j]}'
          f'  →  {grids[k][best_i, best_j]:.4f}')
    print(f'  (baseline k1=0.7 b=0.9 not in grid — nearest k1=0.75 b=0.9: '
          f'{grids[k][K1_VALUES.index(0.75)][B_VALUES.index(0.9)]:.4f})')

# ── Plots ─────────────────────────────────────────────────────────────────────
IMG_DIR.mkdir(parents=True, exist_ok=True)
print('\nSaving plots …')
for k in K_EVAL:
    plot_heatmap(grids[k], k, IMG_DIR / f'bm25_grid_recall{k}.png')
plot_combined(grids, IMG_DIR / 'bm25_grid_combined.png')
