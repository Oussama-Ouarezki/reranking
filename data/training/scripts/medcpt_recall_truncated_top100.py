"""
MedCPT dense retrieval recall@5/10/20/50/100 on the TRUNCATED corpus.

Reuses the cached FAISS index and corpus embeddings from dense_retrival/.
Only query encoding is done from scratch.

Saves:
    data/training/dense_retrival/recall_scores_top100.tsv
    data/training/images/medcpt_recall_truncated_top100.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/training/scripts/medcpt_recall_truncated_top100.py
"""

import csv
import json
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────
BASE         = Path('/home/oussama/Desktop/reranking_project')
QUERIES_PATH = BASE / 'data' / 'training' / 'truncated' / 'queries.jsonl'
QRELS_PATH   = BASE / 'data' / 'training' / 'truncated' / 'qrels.tsv'
CACHE_DIR    = BASE / 'data' / 'training' / 'dense_retrival'
OUT_TSV      = CACHE_DIR / 'recall_scores_top100.tsv'
OUT_PNG      = BASE / 'data' / 'training' / 'images' / 'medcpt_recall_truncated_top100.png'

QUERY_MODEL    = 'ncbi/MedCPT-Query-Encoder'
QUERY_BATCH    = 64
MAX_LENGTH     = 512
TOP_K_RETRIEVE = 100
RECALL_CUTOFFS = [5, 10, 20, 50, 100]
EMBED_DIM      = 768
BAR_COLOR      = '#2196F3'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_queries():
    qids, qtexts = [], []
    with QUERIES_PATH.open(encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qids.append(q['_id'])
            qtexts.append(q['text'])
    print(f'  Queries : {len(qids):,}')
    return qids, qtexts


def load_qrels(valid_ids: set[str]) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with QRELS_PATH.open(encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            qid = row['query-id']
            did = row['corpus-id']
            if did not in valid_ids or int(row['score']) < 1:
                continue
            qrels.setdefault(qid, set()).add(did)
    print(f'  Qrels   : {sum(len(v) for v in qrels.values()):,} pairs  '
          f'({len(qrels):,} queries)')
    return qrels


# ── Query encoding ────────────────────────────────────────────────────────────

def encode_queries(qtexts, device) -> np.ndarray:
    print(f'Loading {QUERY_MODEL} …')
    tok   = AutoTokenizer.from_pretrained(QUERY_MODEL)
    model = AutoModel.from_pretrained(QUERY_MODEL).to(device).eval()
    n     = len(qtexts)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, n, QUERY_BATCH), desc='Encoding queries', unit='batch'):
            end = min(start + QUERY_BATCH, n)
            enc = tok(qtexts[start:end], max_length=MAX_LENGTH,
                      padding=True, truncation=True, return_tensors='pt')
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[start:end] = cls / norm
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embs


# ── Recall ────────────────────────────────────────────────────────────────────

def compute_and_save_recall(retrieved, corpus_ids, qids, qrels):
    results: dict[str, list[float]] = {}
    for i, qid in enumerate(qids):
        relevant = qrels.get(qid, set())
        if not relevant:
            results[qid] = [0.0] * len(RECALL_CUTOFFS)
            continue
        row = []
        for k in RECALL_CUTOFFS:
            top_k = {corpus_ids[idx] for idx in retrieved[i, :k]}
            row.append(len(top_k & relevant) / len(relevant))
        results[qid] = row

    means = [np.mean([v[i] for v in results.values()])
             for i in range(len(RECALL_CUTOFFS))]

    with OUT_TSV.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['qid'] + [f'recall@{k}' for k in RECALL_CUTOFFS])
        for qid, scores in results.items():
            w.writerow([qid] + [f'{s:.4f}' for s in scores])
        w.writerow(['mean'] + [f'{m:.4f}' for m in means])
    print(f'  Saved → {OUT_TSV}')
    return means


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_histogram(means):
    x    = range(len(RECALL_CUTOFFS))
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(x, means, color=BAR_COLOR, alpha=0.85, width=0.55,
                  label='MedCPT Dense — Truncated Corpus')

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.25', fc=BAR_COLOR, ec='none', alpha=0.85),
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f'@{k}' for k in RECALL_CUTOFFS], fontsize=12)
    ax.set_xlabel('K  (number of retrieved documents)', fontsize=12)
    ax.set_ylabel('Mean Recall@K', fontsize=12)
    ax.set_title(
        'MedCPT Dense Retrieval — Recall@K\nTruncated BioASQ Training Corpus (280 tokens)',
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f'  Plot saved → {OUT_PNG}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('\n── Loading cached index ──────────────────────────────')
    index      = faiss.read_index(str(CACHE_DIR / 'corpus.index'))
    corpus_ids = json.loads((CACHE_DIR / 'corpus_ids.json').read_text())
    print(f'  {index.ntotal:,} vectors loaded')

    print('\n── Loading queries + qrels ───────────────────────────')
    qids, qtexts = load_queries()
    qrels        = load_qrels(set(corpus_ids))

    print('\n── Encoding queries ──────────────────────────────────')
    query_embs = encode_queries(qtexts, device)

    print(f'\n── Retrieving top-{TOP_K_RETRIEVE} ───────────────────────────────')
    _, retrieved = index.search(query_embs, TOP_K_RETRIEVE)
    print(f'  Retrieved top-{TOP_K_RETRIEVE} for {len(qids):,} queries')

    print('\n── Computing recall ──────────────────────────────────')
    means = compute_and_save_recall(retrieved, corpus_ids, qids, qrels)

    print(f'\n  {"K":<8}  {"Mean Recall":>12}')
    print('  ' + '─' * 22)
    for k, m in zip(RECALL_CUTOFFS, means):
        print(f'  @{k:<7}  {m:>12.4f}')

    print('\n── Plotting ──────────────────────────────────────────')
    plot_histogram(means)


if __name__ == '__main__':
    main()
