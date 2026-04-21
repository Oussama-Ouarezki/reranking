"""
MedCPT dense retrieval recall on the FULL (non-truncated) BioASQ corpus.

Encodes data/bioasq/processed/corpus.jsonl with MedCPT-Article-Encoder and
queries from data/training/truncated/queries.jsonl with MedCPT-Query-Encoder.
Embeddings, FAISS index, and recall scores are cached so re-runs are instant.

Recall cutoffs : [5, 10, 20, 50, 100]
Saves:
    data/training/dense_retrival_full/corpus_embeddings.npy
    data/training/dense_retrival_full/corpus.index
    data/training/dense_retrival_full/corpus_ids.json
    data/training/dense_retrival_full/recall_scores.tsv
    data/training/images/medcpt_recall_full_corpus.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/training/scripts/medcpt_recall_full_corpus.py
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
CORPUS_PATH  = BASE / 'data' / 'bioasq' / 'processed' / 'corpus.jsonl'
QUERIES_PATH = BASE / 'data' / 'training' / 'truncated' / 'queries.jsonl'
QRELS_PATH   = BASE / 'data' / 'training' / 'truncated' / 'qrels.tsv'
OUT_DIR      = BASE / 'data' / 'training' / 'dense_retrival_full'
OUT_PNG      = BASE / 'data' / 'training' / 'images' / 'medcpt_recall_full_corpus.png'

ARTICLE_MODEL  = 'ncbi/MedCPT-Article-Encoder'
QUERY_MODEL    = 'ncbi/MedCPT-Query-Encoder'
ARTICLE_BATCH  = 32
QUERY_BATCH    = 64
MAX_LENGTH     = 512
TOP_K_RETRIEVE = 100
RECALL_CUTOFFS = [5, 10, 20, 50, 100]
EMBED_DIM      = 768

BAR_COLOR      = '#673AB7'   # purple — distinct from BM25 blue


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus():
    ids, titles, texts = [], [], []
    with CORPUS_PATH.open(encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc['_id'])
            titles.append(doc.get('title', ''))
            texts.append(doc['text'])
    print(f'  Corpus  : {len(ids):,} documents')
    return ids, titles, texts


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


# ── Encoding ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = f'  ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''
    print(f'  Device  : {device}{tag}')
    return device


def encode_articles(titles, texts, device) -> np.ndarray:
    print(f'Loading {ARTICLE_MODEL} …')
    tok   = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model = AutoModel.from_pretrained(ARTICLE_MODEL).to(device).eval()
    n     = len(titles)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, n, ARTICLE_BATCH), desc='Encoding corpus', unit='batch'):
            end = min(start + ARTICLE_BATCH, n)
            enc = tok(titles[start:end], texts[start:end],
                      max_length=MAX_LENGTH, padding=True,
                      truncation=True, return_tensors='pt')
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[start:end] = cls / norm
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embs


def encode_queries(qtexts, device) -> np.ndarray:
    print(f'Loading {QUERY_MODEL} …')
    tok   = AutoTokenizer.from_pretrained(QUERY_MODEL)
    model = AutoModel.from_pretrained(QUERY_MODEL).to(device).eval()
    n     = len(qtexts)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in tqdm(range(0, n, QUERY_BATCH), desc='Encoding queries', unit='batch'):
            end = min(start + QUERY_BATCH, n)
            enc = tok(qtexts[start:end],
                      max_length=MAX_LENGTH, padding=True,
                      truncation=True, return_tensors='pt')
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[start:end] = cls / norm
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embs


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_or_load_index(embs, index_path, ids_path, corpus_ids):
    if index_path.exists() and ids_path.exists():
        print('Loading cached FAISS index …')
        index = faiss.read_index(str(index_path))
        print(f'  {index.ntotal:,} vectors')
        return index
    print('Building FAISS IndexFlatIP …')
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embs)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(corpus_ids), encoding='utf-8')
    print(f'  {index.ntotal:,} vectors indexed  →  {index_path}')
    return index


# ── Recall ────────────────────────────────────────────────────────────────────

def compute_recall(retrieved, corpus_ids, qids, qrels):
    results = {}
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
    return results


def save_recall(recall, out_path):
    header = ['qid'] + [f'recall@{k}' for k in RECALL_CUTOFFS]
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(header)
        for qid, scores in recall.items():
            w.writerow([qid] + [f'{s:.4f}' for s in scores])
        means = [np.mean([v[i] for v in recall.values()])
                 for i in range(len(RECALL_CUTOFFS))]
        w.writerow(['mean'] + [f'{m:.4f}' for m in means])
    print(f'  Recall scores → {out_path}')
    return means


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_histogram(means, out_path):
    x    = range(len(RECALL_CUTOFFS))
    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(x, means, color=BAR_COLOR, alpha=0.85, width=0.55,
                  label='MedCPT Dense — Full Corpus')

    # highlighted value labels
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
        'MedCPT Dense Retrieval — Recall@K\nFull (Non-Truncated) BioASQ Training Corpus',
        fontsize=13,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f'  Plot saved → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_path   = OUT_DIR / 'corpus_embeddings.npy'
    index_path = OUT_DIR / 'corpus.index'
    ids_path   = OUT_DIR / 'corpus_ids.json'
    recall_path= OUT_DIR / 'recall_scores.tsv'

    print('── Loading data ──────────────────────────────────────')
    corpus_ids, titles, texts = load_corpus()
    qids, qtexts              = load_queries()
    qrels                     = load_qrels(set(corpus_ids))

    device = get_device()

    print('\n── Encoding corpus ───────────────────────────────────')
    if emb_path.exists():
        print(f'  Found cached embeddings → {emb_path}  (skipping)')
        corpus_embs = np.load(str(emb_path))
    else:
        corpus_embs = encode_articles(titles, texts, device)
        np.save(str(emb_path), corpus_embs)
        print(f'  Embeddings saved → {emb_path}')

    print('\n── Building FAISS index ──────────────────────────────')
    index = build_or_load_index(corpus_embs, index_path, ids_path, corpus_ids)

    print('\n── Encoding queries ──────────────────────────────────')
    query_embs = encode_queries(qtexts, device)

    print(f'\n── Retrieving top-{TOP_K_RETRIEVE} ────────────────────────────────')
    _, retrieved = index.search(query_embs, TOP_K_RETRIEVE)
    print(f'  Retrieved top-{TOP_K_RETRIEVE} for {len(qids):,} queries')

    print('\n── Computing recall ──────────────────────────────────')
    recall = compute_recall(retrieved, corpus_ids, qids, qrels)
    means  = save_recall(recall, recall_path)

    print(f'\n  {"K":<8}  {"Mean Recall":>12}')
    print('  ' + '─' * 22)
    for k, m in zip(RECALL_CUTOFFS, means):
        print(f'  @{k:<7}  {m:>12.4f}')

    print('\n── Plotting ──────────────────────────────────────────')
    plot_histogram(means, OUT_PNG)

    print('\n── Done ──────────────────────────────────────────────')


if __name__ == '__main__':
    main()
