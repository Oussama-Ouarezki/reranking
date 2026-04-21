"""
BM25 vs MedCPT dense retrieval on the full BioASQ corpus.

Both retrieve top-100 per query.  Computes:
  - Recall@5/10/20/50/100 for each method
  - Mean overlap: len(bm25_top100 & medcpt_top100) / 100

MedCPT artefacts (embeddings, index, corpus_ids) are stored in
data/bioasq/medCPT/.  If the full-corpus embeddings were already built at
data/training/dense_retrival_full/ they are reused (copied) automatically.

Saves:
    data/bioasq/medCPT/corpus_embeddings.npy
    data/bioasq/medCPT/corpus.index
    data/bioasq/medCPT/corpus_ids.json
    data/bioasq/medCPT/recall_bm25.tsv
    data/bioasq/medCPT/recall_medcpt.tsv
    data/bioasq/medCPT/overlap_scores.tsv
    data/bioasq/medCPT/images/recall_comparison.png
    data/bioasq/medCPT/images/overlap.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/medCPT/scripts/bm25_vs_medcpt_full_corpus.py
"""

import csv
import json
import shutil
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
CORPUS_PATH  = BASE / 'data' / 'bioasq' / 'processed' / 'corpus.jsonl'
QUERIES_PATH = BASE / 'data' / 'training' / 'truncated' / 'queries.jsonl'
QRELS_PATH   = BASE / 'data' / 'training' / 'truncated' / 'qrels.tsv'
OUT_DIR      = BASE / 'data' / 'bioasq' / 'medCPT'
IMG_DIR      = OUT_DIR / 'images'

# cached embeddings from earlier full-corpus run (reused if present)
EXISTING_EMB   = BASE / 'data' / 'training' / 'dense_retrival_full' / 'corpus_embeddings.npy'
EXISTING_INDEX = BASE / 'data' / 'training' / 'dense_retrival_full' / 'corpus.index'
EXISTING_IDS   = BASE / 'data' / 'training' / 'dense_retrival_full' / 'corpus_ids.json'

QUERY_MODEL    = 'ncbi/MedCPT-Query-Encoder'
ARTICLE_MODEL  = 'ncbi/MedCPT-Article-Encoder'
QUERY_BATCH    = 64
ARTICLE_BATCH  = 32
MAX_LENGTH     = 512
TOP_K          = 100
RECALL_CUTOFFS = [5, 10, 20, 50, 100]
EMBED_DIM      = 768

# BM25 best params from grid search
BM25_K1 = 0.7
BM25_B  = 0.75

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
    tok   = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model = AutoModel.from_pretrained(ARTICLE_MODEL).to(device).eval()
    n     = len(titles)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, n, ARTICLE_BATCH), desc='Encoding corpus', unit='batch'):
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
    tok   = AutoTokenizer.from_pretrained(QUERY_MODEL)
    model = AutoModel.from_pretrained(QUERY_MODEL).to(device).eval()
    n     = len(qtexts)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, n, QUERY_BATCH), desc='Encoding queries', unit='batch'):
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


# ── FAISS ─────────────────────────────────────────────────────────────────────

def build_or_load_index(corpus_embs, corpus_ids):
    index_path = OUT_DIR / 'corpus.index'
    ids_path   = OUT_DIR / 'corpus_ids.json'
    if index_path.exists():
        print('  Loading cached FAISS index …')
        return faiss.read_index(str(index_path))
    print('  Building FAISS IndexFlatIP …')
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(corpus_embs)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(corpus_ids), encoding='utf-8')
    print(f'  {index.ntotal:,} vectors indexed')
    return index


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def medcpt_retrieve(index, query_embs, k=TOP_K):
    _, retrieved = index.search(query_embs, k)
    return retrieved   # (n_queries, k)


def bm25_retrieve(corpus_texts, queries, doc_ids, k=TOP_K):
    print('  Building BM25 index …', flush=True)
    bm25  = BM25Okapi([t.lower().split() for t in corpus_texts],
                      k1=BM25_K1, b=BM25_B)
    id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    results   = {}   # qid -> list of doc_ids (top-k)
    for qid, qtext in tqdm(queries.items(), desc='BM25 retrieval'):
        scores   = bm25.get_scores(qtext.lower().split())
        top_idxs = np.argpartition(scores, -k)[-k:]
        top_idxs = top_idxs[np.argsort(scores[top_idxs])[::-1]]
        results[qid] = [doc_ids[i] for i in top_idxs]
    return results


# ── Recall + overlap ──────────────────────────────────────────────────────────

def compute_recall(retrieved_ids: list[list[str]],
                   qids: list[str],
                   qrels: dict) -> tuple[dict[int, float], dict[str, list[float]]]:
    per_query: dict[str, list[float]] = {}
    recall_lists: dict[int, list[float]] = defaultdict(list)
    for i, qid in enumerate(qids):
        rel = qrels.get(qid, set())
        if not rel:
            continue
        top = retrieved_ids[i]
        row = []
        for k in RECALL_CUTOFFS:
            hits = len(set(top[:k]) & rel)
            row.append(hits / len(rel))
            recall_lists[k].append(hits / len(rel))
        per_query[qid] = row
    means = {k: float(np.mean(v)) for k, v in recall_lists.items()}
    return means, per_query


def compute_overlap(bm25_top: dict[str, list[str]],
                    medcpt_top: dict[str, list[str]]) -> float:
    overlaps = []
    for qid in bm25_top:
        if qid not in medcpt_top:
            continue
        b_set = set(bm25_top[qid])
        m_set = set(medcpt_top[qid])
        overlaps.append(len(b_set & m_set) / TOP_K)
    mean_overlap = float(np.mean(overlaps))
    print(f'  Mean overlap (BM25 ∩ MedCPT) / {TOP_K} = {mean_overlap:.4f}  '
          f'({mean_overlap*100:.1f}%)')
    return mean_overlap


def save_tsv(per_query, out_path, means):
    with out_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['qid'] + [f'recall@{k}' for k in RECALL_CUTOFFS])
        for qid, scores in per_query.items():
            w.writerow([qid] + [f'{s:.4f}' for s in scores])
        w.writerow(['mean'] + [f'{means[k]:.4f}' for k in RECALL_CUTOFFS])
    print(f'  Saved → {out_path}')


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_recall_comparison(bm25_means, medcpt_means):
    n_k       = len(RECALL_CUTOFFS)
    bar_width = 0.35
    x         = np.arange(n_k)

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
    ax.set_title('BM25 vs MedCPT Dense — Recall@K\nFull BioASQ Corpus', fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / 'recall_comparison.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'  Saved → {out}')


def plot_overlap(mean_overlap):
    fig, ax = plt.subplots(figsize=(5, 5))
    bar = ax.bar([0], [mean_overlap], color=COLORS['Overlap'], alpha=0.85,
                 width=0.4, label='Mean overlap')
    ax.text(bar[0].get_x() + bar[0].get_width() / 2, mean_overlap + 0.01,
            f'{mean_overlap:.4f}\n({mean_overlap*100:.1f}%)',
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

    print('── Loading corpus & queries ──────────────────────────')
    corpus_ids, titles, texts = load_corpus()
    qids, qtexts              = load_queries()
    qrels                     = load_qrels(set(corpus_ids))
    queries_dict              = dict(zip(qids, qtexts))
    corpus_texts_full         = [(t + ' ' + tx).strip() for t, tx in zip(titles, texts)]

    device = get_device()

    # ── MedCPT corpus embeddings (reuse from dense_retrival_full if available) ──
    emb_path = OUT_DIR / 'corpus_embeddings.npy'
    print('\n── MedCPT corpus embeddings ──────────────────────────')
    if emb_path.exists():
        print(f'  Loading cached embeddings → {emb_path}')
        corpus_embs = np.load(str(emb_path))
    elif EXISTING_EMB.exists():
        print(f'  Copying from {EXISTING_EMB} …')
        shutil.copy(EXISTING_EMB, emb_path)
        shutil.copy(EXISTING_IDS, OUT_DIR / 'corpus_ids.json')
        corpus_embs = np.load(str(emb_path))
        print('  Done.')
    else:
        print('  Encoding full corpus with MedCPT-Article-Encoder …')
        corpus_embs = encode_articles(titles, texts, device)
        np.save(str(emb_path), corpus_embs)
        print(f'  Embeddings saved → {emb_path}')

    # ── FAISS index ────────────────────────────────────────────────────────────
    print('\n── FAISS index ───────────────────────────────────────')
    index = build_or_load_index(corpus_embs, corpus_ids)

    # ── MedCPT query encoding + retrieval ─────────────────────────────────────
    print('\n── MedCPT query encoding ─────────────────────────────')
    query_embs = encode_queries(qtexts, device)
    print(f'\n── MedCPT retrieval (top-{TOP_K}) ────────────────────')
    dense_retrieved = medcpt_retrieve(index, query_embs)
    medcpt_top = {qids[i]: [corpus_ids[idx] for idx in dense_retrieved[i]]
                  for i in range(len(qids))}

    # ── BM25 retrieval ─────────────────────────────────────────────────────────
    print(f'\n── BM25 retrieval (top-{TOP_K}, k1={BM25_K1}, b={BM25_B}) ────')
    bm25_top = bm25_retrieve(corpus_texts_full, queries_dict, corpus_ids)

    # ── Recall ─────────────────────────────────────────────────────────────────
    print('\n── Computing recall ──────────────────────────────────')
    medcpt_means, medcpt_pq = compute_recall(
        [medcpt_top[qid] for qid in qids], qids, qrels)
    bm25_means, bm25_pq = compute_recall(
        [bm25_top[qid] for qid in qids], qids, qrels)

    save_tsv(medcpt_pq, OUT_DIR / 'recall_medcpt.tsv', medcpt_means)
    save_tsv(bm25_pq,   OUT_DIR / 'recall_bm25.tsv',   bm25_means)

    # ── Overlap ────────────────────────────────────────────────────────────────
    print('\n── Computing overlap ─────────────────────────────────')
    mean_overlap = compute_overlap(bm25_top, medcpt_top)
    with (OUT_DIR / 'overlap_scores.tsv').open('w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['qid', 'overlap'])
        for qid in bm25_top:
            if qid in medcpt_top:
                ov = len(set(bm25_top[qid]) & set(medcpt_top[qid])) / TOP_K
                w.writerow([qid, f'{ov:.4f}'])
        w.writerow(['mean', f'{mean_overlap:.4f}'])

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f'\n  {"K":<8}  {"BM25":>10}  {"MedCPT":>10}  {"Δ":>8}')
    print('  ' + '─' * 42)
    for k in RECALL_CUTOFFS:
        delta = medcpt_means[k] - bm25_means[k]
        sign  = '+' if delta >= 0 else ''
        print(f'  @{k:<7}  {bm25_means[k]:>10.4f}  {medcpt_means[k]:>10.4f}  '
              f'{sign}{delta:>7.4f}')
    print(f'\n  Mean overlap@{TOP_K}: {mean_overlap:.4f}  ({mean_overlap*100:.1f}%)')

    # ── Plots ──────────────────────────────────────────────────────────────────
    print('\n── Saving plots ──────────────────────────────────────')
    plot_recall_comparison(bm25_means, medcpt_means)
    plot_overlap(mean_overlap)
    print('\n── Done ──────────────────────────────────────────────')


if __name__ == '__main__':
    main()
