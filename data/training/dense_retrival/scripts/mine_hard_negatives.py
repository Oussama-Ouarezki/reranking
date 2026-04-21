"""
MedCPT hard negative mining for BioASQ training triples.

Encodes the truncated corpus and queries with MedCPT, builds a FAISS index,
retrieves the top-100 most semantically similar documents per query, removes
known positives, and saves the top-20 hardest negatives as training triples.

Reads from  : data/training/truncated/
Writes to   : data/training/dense_retrival/
  hard_negatives_triples.tsv   — query TAB positive TAB negative (text)
  hard_negatives_ids.tsv       — qid TAB pos_id TAB neg_id
  recall_scores.tsv            — recall@1/5/10/20 per query + mean row
  corpus.index                 — FAISS IndexFlatIP (reused on resume)
  corpus_ids.json              — FAISS position → corpus _id
  corpus_embeddings.npy        — (49528, 768) float32 (reused on resume)

Installation (one-time):
    pip install faiss-cpu

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/training/dense_retrival/scripts/mine_hard_negatives.py
"""

import csv
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ── paths ─────────────────────────────────────────────────────────────────────
BASE        = Path('/home/oussama/Desktop/reranking_project')
CORPUS_PATH = BASE / 'data' / 'training' / 'truncated' / 'corpus.jsonl'
QUERIES_PATH= BASE / 'data' / 'training' / 'truncated' / 'queries.jsonl'
QRELS_PATH  = BASE / 'data' / 'training' / 'truncated' / 'qrels.tsv'
OUT_DIR     = BASE / 'data' / 'training' / 'dense_retrival'

# ── models ────────────────────────────────────────────────────────────────────
ARTICLE_MODEL   = 'ncbi/MedCPT-Article-Encoder'
QUERY_MODEL     = 'ncbi/MedCPT-Query-Encoder'

# ── hyperparameters ───────────────────────────────────────────────────────────
ARTICLE_BATCH   = 32
QUERY_BATCH     = 64
MAX_LENGTH      = 512
TOP_K_RETRIEVE  = 100
TOP_K_NEGATIVES = 20
RECALL_CUTOFFS  = [1, 5, 10, 20]
EMBED_DIM       = 768


# ── data loading ──────────────────────────────────────────────────────────────

def load_corpus(path: Path) -> tuple[list[str], list[str], list[str]]:
    ids, titles, texts = [], [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc['_id'])
            titles.append(doc.get('title', ''))
            texts.append(doc['text'])
    print(f'  Corpus  : {len(ids):,} documents')
    return ids, titles, texts


def load_queries(path: Path) -> tuple[list[str], list[str]]:
    qids, qtexts = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qids.append(q['_id'])
            qtexts.append(q['text'])
    print(f'  Queries : {len(qids):,}')
    return qids, qtexts


def load_qrels(path: Path, valid_ids: set[str]) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    skipped = 0
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            qid = row['query-id']
            did = row['corpus-id']
            if did not in valid_ids:
                skipped += 1
                continue
            if int(row['score']) < 1:
                continue
            qrels.setdefault(qid, set()).add(did)
    print(f'  Qrels   : {sum(len(v) for v in qrels.values()):,} pairs '
          f'({len(qrels):,} queries, {skipped} corpus-id misses skipped)')
    return qrels


# ── encoding ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device  : {device}'
          + (f'  ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''))
    return device


def encode_articles(titles: list[str], texts: list[str],
                    device: torch.device) -> np.ndarray:
    print(f'Loading article encoder ({ARTICLE_MODEL}) …')
    tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model     = AutoModel.from_pretrained(ARTICLE_MODEL).to(device).eval()

    n    = len(titles)
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, n, ARTICLE_BATCH),
                          desc='Encoding corpus', unit='batch'):
            end   = min(start + ARTICLE_BATCH, n)
            t_bat = titles[start:end]
            d_bat = texts[start:end]
            enc   = tokenizer(t_bat, d_bat,
                              max_length=MAX_LENGTH,
                              padding=True, truncation=True,
                              return_tensors='pt')
            enc   = {k: v.to(device) for k, v in enc.items()}
            out   = model(**enc)
            cls   = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            norm  = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[start:end] = cls / norm

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embs


def encode_queries(qtexts: list[str], device: torch.device) -> np.ndarray:
    print(f'Loading query encoder ({QUERY_MODEL}) …')
    tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL)
    model     = AutoModel.from_pretrained(QUERY_MODEL).to(device).eval()

    n    = len(qtexts)
    embs = np.zeros((n, EMBED_DIM), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, n, QUERY_BATCH),
                          desc='Encoding queries', unit='batch'):
            end  = min(start + QUERY_BATCH, n)
            enc  = tokenizer(qtexts[start:end],
                             max_length=MAX_LENGTH,
                             padding=True, truncation=True,
                             return_tensors='pt')
            enc  = {k: v.to(device) for k, v in enc.items()}
            out  = model(**enc)
            cls  = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[start:end] = cls / norm

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return embs


# ── faiss ─────────────────────────────────────────────────────────────────────

def build_or_load_index(embs: np.ndarray,
                        index_path: Path,
                        ids_path: Path,
                        corpus_ids: list[str]):
    if index_path.exists() and ids_path.exists():
        print('Loading existing FAISS index …')
        index = faiss.read_index(str(index_path))
        print(f'  {index.ntotal:,} vectors loaded')
        return index

    print('Building FAISS IndexFlatIP …')
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embs)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(corpus_ids), encoding='utf-8')
    print(f'  {index.ntotal:,} vectors indexed  →  {index_path}')
    return index


# ── recall ────────────────────────────────────────────────────────────────────

def compute_recall(retrieved: np.ndarray,
                   corpus_ids: list[str],
                   qids: list[str],
                   qrels: dict[str, set[str]]) -> dict[str, list[float]]:
    results: dict[str, list[float]] = {}
    for i, qid in enumerate(qids):
        relevant = qrels.get(qid, set())
        if not relevant:
            results[qid] = [0.0] * len(RECALL_CUTOFFS)
            continue
        row = []
        for k in RECALL_CUTOFFS:
            top_k_ids = {corpus_ids[idx] for idx in retrieved[i, :k]}
            row.append(len(top_k_ids & relevant) / len(relevant))
        results[qid] = row
    return results


def write_recall_scores(recall: dict[str, list[float]], out_path: Path) -> None:
    header = ['qid'] + [f'recall@{k}' for k in RECALL_CUTOFFS]
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(header)
        for qid, scores in recall.items():
            w.writerow([qid] + [f'{s:.4f}' for s in scores])
        means = [np.mean([v[i] for v in recall.values()])
                 for i in range(len(RECALL_CUTOFFS))]
        w.writerow(['mean'] + [f'{m:.4f}' for m in means])
    print(f'  Recall scores → {out_path}')
    for k, m in zip(RECALL_CUTOFFS, means):
        print(f'    Mean Recall@{k:<3} : {m*100:.2f}%')


# ── hard negative mining ──────────────────────────────────────────────────────

def mine_hard_negatives(retrieved: np.ndarray,
                        corpus_ids: list[str],
                        qids: list[str],
                        qtexts: list[str],
                        qrels: dict[str, set[str]],
                        id_to_doc: dict[str, dict]) -> list[dict]:
    triples   = []
    few_warns = 0

    for i, qid in enumerate(qids):
        positives = qrels.get(qid, set())
        if not positives:
            continue

        pos_id = sorted(positives)[0]
        if pos_id not in id_to_doc:
            continue

        hard_negs = []
        for idx in retrieved[i]:
            did = corpus_ids[int(idx)]
            if did in positives:
                continue
            if did not in id_to_doc:
                continue
            hard_negs.append(did)
            if len(hard_negs) == TOP_K_NEGATIVES:
                break

        if len(hard_negs) < TOP_K_NEGATIVES:
            few_warns += 1
            print(f'  WARNING: query {qid} yielded only {len(hard_negs)}/{TOP_K_NEGATIVES} hard negatives')

        for neg_id in hard_negs:
            triples.append({
                'qid'       : qid,
                'query_text': qtexts[i],
                'pos_id'    : pos_id,
                'pos_text'  : id_to_doc[pos_id]['text'],
                'neg_id'    : neg_id,
                'neg_text'  : id_to_doc[neg_id]['text'],
            })

    if few_warns:
        print(f'  {few_warns} queries had fewer than {TOP_K_NEGATIVES} hard negatives')
    return triples


def write_triples(triples: list[dict],
                  text_path: Path,
                  ids_path: Path) -> None:
    with open(text_path, 'w', encoding='utf-8', newline='') as ft, \
         open(ids_path,  'w', encoding='utf-8', newline='') as fi:
        wt = csv.writer(ft, delimiter='\t')
        wi = csv.writer(fi, delimiter='\t')
        wt.writerow(['query', 'positive', 'negative'])
        wi.writerow(['qid', 'pos_id', 'neg_id'])
        for t in triples:
            wt.writerow([t['query_text'], t['pos_text'], t['neg_text']])
            wi.writerow([t['qid'], t['pos_id'], t['neg_id']])
    print(f'  Triples (text) → {text_path}  ({len(triples):,} rows)')
    print(f'  Triples (ids)  → {ids_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'scripts').mkdir(exist_ok=True)

    emb_path   = OUT_DIR / 'corpus_embeddings.npy'
    index_path = OUT_DIR / 'corpus.index'
    ids_path   = OUT_DIR / 'corpus_ids.json'

    print('── Loading data ──────────────────────────────────────')
    corpus_ids, titles, texts = load_corpus(CORPUS_PATH)
    qids, qtexts              = load_queries(QUERIES_PATH)
    valid_ids                 = set(corpus_ids)
    qrels                     = load_qrels(QRELS_PATH, valid_ids)
    id_to_doc = {cid: {'title': t, 'text': tx}
                 for cid, t, tx in zip(corpus_ids, titles, texts)}

    device = get_device()

    print('\n── Encoding corpus ───────────────────────────────────')
    if emb_path.exists() and index_path.exists():
        print(f'  Found cached embeddings → {emb_path}  (skipping article encoding)')
        corpus_embs = np.load(str(emb_path))
    else:
        corpus_embs = encode_articles(titles, texts, device)
        np.save(str(emb_path), corpus_embs)
        print(f'  Embeddings saved → {emb_path}')

    print('\n── Building FAISS index ──────────────────────────────')
    index = build_or_load_index(corpus_embs, index_path, ids_path, corpus_ids)

    print('\n── Encoding queries ──────────────────────────────────')
    query_embs = encode_queries(qtexts, device)

    print('\n── Retrieving top-100 ────────────────────────────────')
    _, retrieved = index.search(query_embs, TOP_K_RETRIEVE)
    print(f'  Retrieved top-{TOP_K_RETRIEVE} for {len(qids):,} queries')

    print('\n── Computing recall ──────────────────────────────────')
    recall = compute_recall(retrieved, corpus_ids, qids, qrels)
    write_recall_scores(recall, OUT_DIR / 'recall_scores.tsv')

    print('\n── Mining hard negatives ─────────────────────────────')
    triples = mine_hard_negatives(retrieved, corpus_ids, qids, qtexts,
                                  qrels, id_to_doc)
    write_triples(triples,
                  OUT_DIR / 'hard_negatives_triples.tsv',
                  OUT_DIR / 'hard_negatives_ids.tsv')

    print(f'\n── Done ──────────────────────────────────────────────')
    print(f'  Total triples : {len(triples):,}')
    print(f'  Output dir    : {OUT_DIR}')


if __name__ == '__main__':
    main()
