"""
Step 1b — False Negative Filtering with MedCPT Cross-Encoder

Pipeline per query:
  1. BM25 top-100 candidates        (data/bm25_indexing/)
  2. MedCPT bi-encoder top-100      (data/bioasq/medCPT/fresh/)
  3. Union → remove gold docs
  4. Score gold docs + candidates with MedCPT cross-encoder
  5. gold_score = min(cross-encoder scores over gold docs)
  6. Filter: 0.1 * gold_score < neg_score < 0.95 * gold_score  (TopK-PercPos)
  7. Sort survivors desc by score, keep top-20
  8. pos_id = gold doc with highest cross-encoder score

Saves to data/bioasq/hard_negatives/:
    hard_negatives_ids.tsv    — qid TAB pos_id TAB neg_id  (≤20 rows per query)
    filtering_stats.tsv       — per-query stats
    progress.json             — checkpoint for resume

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/hard_negatives/mine_hard_negatives.py
"""

import csv
import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import (AutoModel, AutoModelForSequenceClassification,
                          AutoTokenizer)

# ── Config ────────────────────────────────────────────────────────────────────
BASE         = Path('/home/oussama/Desktop/reranking_project')
PROCESSED    = BASE / 'data' / 'bioasq' / 'processed'
BM25_DIR     = BASE / 'data' / 'bm25_indexing'
DENSE_DIR    = BASE / 'data' / 'bioasq' / 'medCPT' / 'fresh'
OUT_DIR      = BASE / 'data' / 'bioasq' / 'hard_negatives'

QUERY_ENCODER  = 'ncbi/MedCPT-Query-Encoder'
CROSS_ENCODER  = 'ncbi/MedCPT-Cross-Encoder'

TOP_K          = 100
MAX_NEGATIVES  = 20
UPPER_RATIO    = 0.95   # TopK-PercPos
LOWER_RATIO    = 0.10   # RocketQA lower bound (relative)
CE_BATCH       = 32     # cross-encoder batch size
QUERY_BATCH    = 64
MAX_LENGTH     = 512
EMBED_DIM      = 768


def tokenize_bm25(text: str) -> list[str]:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_corpus() -> tuple[list[str], dict[str, str]]:
    ids, id2text = [], {}
    with (PROCESSED / 'corpus.jsonl').open(encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc['_id'])
            id2text[doc['_id']] = (doc.get('title', '') + ' ' + doc['text']).strip()
    print(f'  Corpus  : {len(ids):,} documents')
    return ids, id2text


def load_queries() -> tuple[list[str], list[str]]:
    qids, qtexts = [], []
    with (PROCESSED / 'queries.jsonl').open(encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qids.append(q['_id'])
            qtexts.append(q['text'])
    print(f'  Queries : {len(qids):,}')
    return qids, qtexts


def load_qrels(valid_ids: set) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with (PROCESSED / 'qrels.tsv').open(encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            qid, did = row['query-id'], row['corpus-id']
            if did not in valid_ids or int(row['score']) < 1:
                continue
            qrels.setdefault(qid, set()).add(did)
    print(f'  Qrels   : {sum(len(v) for v in qrels.values()):,} pairs '
          f'({len(qrels):,} queries)')
    return qrels


# ── Phase 1: BM25 retrieval (all queries) ────────────────────────────────────

def bm25_retrieve_all(qids, qtexts, bm25_corpus_ids) -> dict[str, list[str]]:
    print('Loading BM25 index …')
    bm25 = pickle.load((BM25_DIR / 'bm25_index.pkl').open('rb'))
    results = {}
    for qid, qtext in tqdm(zip(qids, qtexts), total=len(qids),
                           desc='BM25 retrieval'):
        scores   = bm25.get_scores(tokenize_bm25(qtext))
        top_idxs = np.argsort(scores)[::-1][:TOP_K]
        results[qid] = [bm25_corpus_ids[i] for i in top_idxs]
    del bm25
    return results


# ── Phase 2: MedCPT bi-encoder retrieval (all queries) ───────────────────────

def dense_retrieve_all(qids, qtexts, dense_corpus_ids, device) -> dict[str, list[str]]:
    print(f'Loading FAISS index from {DENSE_DIR} …')
    index = faiss.read_index(str(DENSE_DIR / 'corpus.index'))
    print(f'  {index.ntotal:,} vectors')

    print(f'Encoding queries with {QUERY_ENCODER} …')
    tok   = AutoTokenizer.from_pretrained(QUERY_ENCODER)
    model = AutoModel.from_pretrained(QUERY_ENCODER).to(device).eval()
    n     = len(qtexts)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, n, QUERY_BATCH), desc='Encoding queries'):
            e   = min(s + QUERY_BATCH, n)
            enc = tok(qtexts[s:e], max_length=MAX_LENGTH, padding=True,
                      truncation=True, return_tensors='pt')
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[s:e] = cls / norm
    del model

    _, retrieved = index.search(embs, TOP_K)
    results = {qids[i]: [dense_corpus_ids[idx] for idx in retrieved[i]]
               for i in range(n)}
    return results


# ── Phase 3: cross-encoder scoring + filtering ───────────────────────────────

def score_pairs(ce_tok, ce_model, query: str,
                doc_texts: list[str], device) -> np.ndarray:
    scores = np.zeros(len(doc_texts), dtype=np.float32)
    for s in range(0, len(doc_texts), CE_BATCH):
        e     = min(s + CE_BATCH, len(doc_texts))
        batch = doc_texts[s:e]
        enc   = ce_tok([query] * len(batch), batch,
                       max_length=MAX_LENGTH, padding=True,
                       truncation=True, return_tensors='pt')
        enc   = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = ce_model(**enc).logits.squeeze(-1)
        scores[s:e] = torch.sigmoid(logits).cpu().float().numpy()
    return scores


def filter_negatives(qid, qtext, gold_ids, candidates,
                     id2text, ce_tok, ce_model, device):
    # score gold docs
    gold_list   = [d for d in gold_ids if d in id2text]
    if not gold_list:
        return None, [], 0.0, 0.0, 0.0

    gold_texts  = [id2text[d] for d in gold_list]
    gold_scores = score_pairs(ce_tok, ce_model, qtext, gold_texts, device)
    pos_id      = gold_list[int(np.argmax(gold_scores))]
    gold_min    = float(gold_scores.min())

    upper = UPPER_RATIO * gold_min
    lower = LOWER_RATIO * gold_min

    # score candidate docs
    cand_list  = [d for d in candidates if d in id2text]
    cand_texts = [id2text[d] for d in cand_list]
    if not cand_list:
        return pos_id, [], gold_min, upper, lower

    cand_scores = score_pairs(ce_tok, ce_model, qtext, cand_texts, device)

    # filter
    survivors = [(did, float(sc)) for did, sc in zip(cand_list, cand_scores)
                 if lower < sc < upper]

    # sort desc, top-20
    survivors.sort(key=lambda x: x[1], reverse=True)
    survivors = survivors[:MAX_NEGATIVES]

    return pos_id, survivors, gold_min, upper, lower


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag    = f'  ({torch.cuda.get_device_name(0)})' if device.type == 'cuda' else ''
    print(f'Device: {device}{tag}')

    # ── Load data ─────────────────────────────────────────────────────────────
    print('\n── Loading data ──────────────────────────────────────')
    corpus_ids, id2text     = load_corpus()
    qids, qtexts            = load_queries()
    qrels                   = load_qrels(set(corpus_ids))
    bm25_corpus_ids         = json.loads((BM25_DIR / 'corpus_ids.json').read_text())
    dense_corpus_ids        = json.loads((DENSE_DIR / 'corpus_ids.json').read_text())

    # ── Phase 1 & 2: retrieval ────────────────────────────────────────────────
    print('\n── Phase 1: BM25 retrieval ───────────────────────────')
    bm25_top = bm25_retrieve_all(qids, qtexts, bm25_corpus_ids)

    print('\n── Phase 2: MedCPT bi-encoder retrieval ──────────────')
    dense_top = dense_retrieve_all(qids, qtexts, dense_corpus_ids, device)

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # ── Phase 3: cross-encoder filtering ─────────────────────────────────────
    print(f'\n── Phase 3: cross-encoder filtering ({CROSS_ENCODER}) ──')
    ce_tok   = AutoTokenizer.from_pretrained(CROSS_ENCODER)
    ce_model = AutoModelForSequenceClassification.from_pretrained(
        CROSS_ENCODER).to(device).eval()

    # resume support
    progress_path = OUT_DIR / 'progress.json'
    done_qids: set[str] = set()
    if progress_path.exists():
        done_qids = set(json.loads(progress_path.read_text()))
        print(f'  Resuming — {len(done_qids):,} queries already done')

    ids_path   = OUT_DIR / 'hard_negatives_ids.tsv'
    stats_path = OUT_DIR / 'filtering_stats.tsv'

    ids_mode   = 'a' if done_qids else 'w'
    stats_mode = 'a' if done_qids else 'w'

    with ids_path.open(ids_mode, encoding='utf-8', newline='') as fids, \
         stats_path.open(stats_mode, encoding='utf-8', newline='') as fstats:

        ids_writer   = csv.writer(fids,   delimiter='\t')
        stats_writer = csv.writer(fstats, delimiter='\t')

        if not done_qids:
            ids_writer.writerow(['qid', 'pos_id', 'neg_id'])
            stats_writer.writerow(['qid', 'n_candidates', 'n_survivors',
                                   'gold_min_score', 'upper_thresh',
                                   'lower_thresh'])

        for qid, qtext in tqdm(zip(qids, qtexts), total=len(qids),
                               desc='Cross-encoder filtering'):
            if qid in done_qids:
                continue

            gold_ids = qrels.get(qid, set())
            if not gold_ids:
                done_qids.add(qid)
                continue

            # union of BM25 + dense candidates, remove gold
            candidates = list(
                (set(bm25_top.get(qid, [])) | set(dense_top.get(qid, [])))
                - gold_ids
            )

            pos_id, survivors, gold_min, upper, lower = filter_negatives(
                qid, qtext, gold_ids, candidates,
                id2text, ce_tok, ce_model, device
            )

            if pos_id is None:
                done_qids.add(qid)
                continue

            # write negatives
            for neg_id, _ in survivors:
                ids_writer.writerow([qid, pos_id, neg_id])
            fids.flush()

            # write stats
            stats_writer.writerow([
                qid, len(candidates), len(survivors),
                f'{gold_min:.4f}', f'{upper:.4f}', f'{lower:.4f}'
            ])
            fstats.flush()

            done_qids.add(qid)
            # checkpoint every 100 queries
            if len(done_qids) % 100 == 0:
                progress_path.write_text(json.dumps(list(done_qids)))

    progress_path.write_text(json.dumps(list(done_qids)))

    # ── Summary ───────────────────────────────────────────────────────────────
    with ids_path.open(encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # minus header
    print(f'\n── Done ──────────────────────────────────────────────')
    print(f'  Queries processed : {len(done_qids):,}')
    print(f'  Total triples     : {total_rows:,}')
    print(f'  Output            : {ids_path}')


if __name__ == '__main__':
    main()
