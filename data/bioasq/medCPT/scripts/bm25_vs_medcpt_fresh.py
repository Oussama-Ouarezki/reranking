"""
BM25 vs MedCPT dense retrieval — full corpus, fresh indexes.

All data read from data/bioasq/processed/.
Both indexes built from scratch (no cache reuse).

Artefacts saved to data/bioasq/medCPT/fresh/:
    corpus_embeddings.npy
    corpus.index
    corpus_ids.json
    recall_medcpt.tsv
    recall_bm25.tsv
    overlap_scores.tsv
    images/recall_comparison.png
    images/overlap.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/medCPT/scripts/bm25_vs_medcpt_fresh.py
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────
BASE         = Path('/home/oussama/Desktop/reranking_project')
PROCESSED    = BASE / 'data' / 'bioasq' / 'processed'
CORPUS_PATH  = PROCESSED / 'corpus.jsonl'
QUERIES_PATH = PROCESSED / 'queries.jsonl'
QRELS_PATH   = PROCESSED / 'qrels.tsv'

OUT_DIR      = BASE / 'data' / 'bioasq' / 'medCPT' / 'fresh'
IMG_DIR      = OUT_DIR / 'images'

ARTICLE_MODEL  = 'ncbi/MedCPT-Article-Encoder'
QUERY_MODEL    = 'ncbi/MedCPT-Query-Encoder'
ARTICLE_BATCH  = 32
QUERY_BATCH    = 64
MAX_LENGTH     = 512
TOP_K          = 100
RECALL_CUTOFFS = [5, 10, 20, 50, 100]
EMBED_DIM      = 768

BM25_K1 = 0.7
BM25_B  = 0.75


def tokenize(text: str) -> list[str]:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()

COLORS = {'BM25': '#2196F3', 'MedCPT': '#673AB7', 'Overlap': '#4CAF50'}


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


def load_qrels(valid_ids: set) -> dict:
    qrels: dict[str, set[str]] = {}
    with QRELS_PATH.open(encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            qid, did = row['query-id'], row['corpus-id']
            if did not in valid_ids or int(row['score']) < 1:
                continue
            qrels.setdefault(qid, set()).add(did)
    print(f'  Qrels   : {sum(len(v) for v in qrels.values()):,} pairs '
          f'({len(qrels):,} queries)')
    return qrels


# ── MedCPT encoding ───────────────────────────────────────────────────────────

def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag = f'  ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''
    print(f'  Device  : {device}{tag}')
    return device


def encode_articles(titles, texts, device):
    print(f'  Loading {ARTICLE_MODEL} …')
    tok   = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model = AutoModel.from_pretrained(ARTICLE_MODEL).to(device).eval()
    n     = len(titles)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, n, ARTICLE_BATCH), desc='  Encoding corpus', unit='batch'):
            e   = min(s + ARTICLE_BATCH, n)
            enc = tok(titles[s:e], texts[s:e], max_length=MAX_LENGTH,
                      padding=True, truncation=True, return_tensors='pt')
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[s:e] = cls / norm
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embs


def encode_queries(qtexts, device):
    print(f'  Loading {QUERY_MODEL} …')
    tok   = AutoTokenizer.from_pretrained(QUERY_MODEL)
    model = AutoModel.from_pretrained(QUERY_MODEL).to(device).eval()
    n     = len(qtexts)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, n, QUERY_BATCH), desc='  Encoding queries', unit='batch'):
            e   = min(s + QUERY_BATCH, n)
            enc = tok(qtexts[s:e], max_length=MAX_LENGTH,
                      padding=True, truncation=True, return_tensors='pt')
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[s:e] = cls / norm
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embs


# ── FAISS (always rebuilt) ────────────────────────────────────────────────────

def build_index(corpus_embs, corpus_ids):
    print('  Building FAISS IndexFlatIP …')
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(corpus_embs)
    index_path = OUT_DIR / 'corpus.index'
    ids_path   = OUT_DIR / 'corpus_ids.json'
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(corpus_ids), encoding='utf-8')
    print(f'  {index.ntotal:,} vectors indexed  →  {index_path}')
    return index


# ── Retrieval ─────────────────────────────────────────────────────────────────

def medcpt_retrieve(index, query_embs, corpus_ids):
    _, retrieved = index.search(query_embs, TOP_K)
    return {qids[i]: [corpus_ids[idx] for idx in retrieved[i]]
            for i in range(len(qids))}


def bm25_retrieve(corpus_texts, queries_dict, corpus_ids):
    print(f'  Building BM25 index (k1={BM25_K1}, b={BM25_B}) …', flush=True)
    bm25 = BM25Okapi([tokenize(t) for t in corpus_texts], k1=BM25_K1, b=BM25_B)
    results = {}
    for qid, qtext in tqdm(queries_dict.items(), desc='  BM25 retrieval'):
        scores   = bm25.get_scores(tokenize(qtext))
        top_idxs = np.argsort(scores)[::-1][:TOP_K]
        results[qid] = [corpus_ids[i] for i in top_idxs]
    return results


# ── Recall + overlap ──────────────────────────────────────────────────────────

def compute_recall(top_dict, qids, qrels):
    per_query: dict[str, list[float]] = {}
    acc: dict[int, list[float]] = defaultdict(list)
    for qid in qids:
        rel = qrels.get(qid, set())
        if not rel:
            continue
        top = top_dict[qid]
        row = []
        for k in RECALL_CUTOFFS:
            hits = len(set(top[:k]) & rel)
            row.append(hits / len(rel))
            acc[k].append(hits / len(rel))
        per_query[qid] = row
    means = {k: float(np.mean(v)) for k, v in acc.items()}
    return means, per_query


def compute_overlap(bm25_top, medcpt_top, qids):
    vals = []
    for qid in qids:
        if qid in bm25_top and qid in medcpt_top:
            vals.append(len(set(bm25_top[qid]) & set(medcpt_top[qid])) / TOP_K)
    mean_ov = float(np.mean(vals))
    print(f'  Mean overlap (BM25 ∩ MedCPT) / {TOP_K} = {mean_ov:.4f}  ({mean_ov*100:.1f}%)')

    with (OUT_DIR / 'overlap_scores.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['qid', 'overlap'])
        for qid, v in zip(qids, vals):
            w.writerow([qid, f'{v:.4f}'])
        w.writerow(['mean', f'{mean_ov:.4f}'])
    return mean_ov


def save_tsv(per_query, means, path):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['qid'] + [f'recall@{k}' for k in RECALL_CUTOFFS])
        for qid, scores in per_query.items():
            w.writerow([qid] + [f'{s:.4f}' for s in scores])
        w.writerow(['mean'] + [f'{means[k]:.4f}' for k in RECALL_CUTOFFS])
    print(f'  Saved → {path}')


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_recall(bm25_means, medcpt_means):
    bar_width = 0.35
    x = np.arange(len(RECALL_CUTOFFS))
    fig, ax = plt.subplots(figsize=(11, 6))

    bars_b = ax.bar(x - bar_width/2, [bm25_means[k] for k in RECALL_CUTOFFS],
                    width=bar_width, color=COLORS['BM25'], alpha=0.85, label='BM25')
    bars_m = ax.bar(x + bar_width/2, [medcpt_means[k] for k in RECALL_CUTOFFS],
                    width=bar_width, color=COLORS['MedCPT'], alpha=0.85, label='MedCPT')

    for bars, color in [(bars_b, COLORS['BM25']), (bars_m, COLORS['MedCPT'])]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{bar.get_height():.3f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=0.85))

    ax.set_xticks(x)
    ax.set_xticklabels([f'@{k}' for k in RECALL_CUTOFFS], fontsize=12)
    ax.set_ylabel('Mean Recall@K', fontsize=12)
    ax.set_title('BM25 vs MedCPT — Recall@K\nFull BioASQ Corpus (Fresh Indexes)', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)

    out = IMG_DIR / 'recall_comparison.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved → {out}')


def plot_overlap(mean_ov):
    fig, ax = plt.subplots(figsize=(5, 5))
    bar = ax.bar([0], [mean_ov], color=COLORS['Overlap'], alpha=0.85, width=0.4)
    ax.text(bar[0].get_x() + bar[0].get_width() / 2, mean_ov + 0.01,
            f'{mean_ov:.4f}\n({mean_ov*100:.1f}%)',
            ha='center', va='bottom', fontsize=13, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc=COLORS['Overlap'], ec='none', alpha=0.85))
    ax.set_xticks([0])
    ax.set_xticklabels(['BM25 ∩ MedCPT'], fontsize=12)
    ax.set_ylabel('Mean overlap@100', fontsize=12)
    ax.set_title('Top-100 Overlap\nBM25 vs MedCPT — Full BioASQ Corpus', fontsize=13)
    ax.set_ylim(0, 1.05)
    out = IMG_DIR / 'overlap.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved → {out}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    print('── Loading data ──────────────────────────────────────')
    corpus_ids, titles, texts = load_corpus()
    global qids
    qids, qtexts = load_queries()
    qrels        = load_qrels(set(corpus_ids))
    queries_dict = dict(zip(qids, qtexts))
    corpus_texts = [(t + ' ' + tx).strip() for t, tx in zip(titles, texts)]

    device = get_device()

    print('\n── Encoding corpus (MedCPT) ──────────────────────────')
    corpus_embs = encode_articles(titles, texts, device)
    emb_path    = OUT_DIR / 'corpus_embeddings.npy'
    np.save(str(emb_path), corpus_embs)
    print(f'  Embeddings saved → {emb_path}')

    print('\n── Building FAISS index ──────────────────────────────')
    index = build_index(corpus_embs, corpus_ids)

    print('\n── Encoding queries (MedCPT) ─────────────────────────')
    query_embs = encode_queries(qtexts, device)

    print(f'\n── MedCPT retrieval (top-{TOP_K}) ────────────────────')
    medcpt_top = medcpt_retrieve(index, query_embs, corpus_ids)

    print(f'\n── BM25 retrieval (top-{TOP_K}) ──────────────────────')
    bm25_top = bm25_retrieve(corpus_texts, queries_dict, corpus_ids)

    print('\n── Computing recall ──────────────────────────────────')
    medcpt_means, medcpt_pq = compute_recall(medcpt_top, qids, qrels)
    bm25_means,   bm25_pq   = compute_recall(bm25_top,   qids, qrels)
    save_tsv(medcpt_pq, medcpt_means, OUT_DIR / 'recall_medcpt.tsv')
    save_tsv(bm25_pq,   bm25_means,   OUT_DIR / 'recall_bm25.tsv')

    print('\n── Computing overlap ─────────────────────────────────')
    mean_ov = compute_overlap(bm25_top, medcpt_top, qids)

    print(f'\n  {"K":<8}  {"BM25":>10}  {"MedCPT":>10}  {"Δ":>8}')
    print('  ' + '─' * 42)
    for k in RECALL_CUTOFFS:
        d    = medcpt_means[k] - bm25_means[k]
        sign = '+' if d >= 0 else ''
        print(f'  @{k:<7}  {bm25_means[k]:>10.4f}  {medcpt_means[k]:>10.4f}  {sign}{d:>7.4f}')
    print(f'\n  Mean overlap@{TOP_K}: {mean_ov:.4f}  ({mean_ov*100:.1f}%)')

    print('\n── Saving plots ──────────────────────────────────────')
    plot_recall(bm25_means, medcpt_means)
    plot_overlap(mean_ov)
    print('\n── Done ──────────────────────────────────────────────')


if __name__ == '__main__':
    main()
