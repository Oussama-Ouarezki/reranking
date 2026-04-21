"""
Build and serialize a BM25 index over data/bioasq/processed/corpus.jsonl.

Parameters: k1=0.7, b=0.9
Tokenizer : punctuation-stripped lowercase (same as retrieve_bm25_doc_train.py)

Saves to data/bm25_indexing/:
    bm25_index.pkl       — serialized BM25Okapi object
    corpus_ids.json      — ordered list of corpus document IDs
    index_meta.json      — k1, b, vocab size, doc count, build timestamp

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python data/bm25_indexing/build_bm25_index.py
"""

import json
import pickle
import re
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────
BASE        = Path('/home/oussama/Desktop/reranking_project')
CORPUS_PATH = BASE / 'data' / 'bioasq' / 'processed' / 'corpus.jsonl'
OUT_DIR     = BASE / 'data' / 'bm25_indexing'

K1 = 0.7
B  = 0.9


def tokenize(text: str) -> list[str]:
    return re.sub(r'[^\w\s]', ' ', text.lower()).split()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load corpus ───────────────────────────────────────────────────────────
    print('Loading corpus …')
    corpus_ids: list[str] = []
    tokenized_corpus: list[list[str]] = []

    with CORPUS_PATH.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            corpus_ids.append(doc['_id'])
            text = (doc.get('title', '') + ' ' + doc['text']).strip()
            tokenized_corpus.append(tokenize(text))
            if (i + 1) % 10_000 == 0:
                print(f'  {i + 1:,} documents loaded …', flush=True)

    print(f'  {len(corpus_ids):,} documents total')

    # ── Build index ───────────────────────────────────────────────────────────
    print(f'\nBuilding BM25Okapi index  (k1={K1}, b={B}) …')
    t0    = time.time()
    bm25  = BM25Okapi(tokenized_corpus, k1=K1, b=B)
    elapsed = time.time() - t0
    print(f'  Built in {elapsed:.1f}s')
    print(f'  Vocab size : {len(bm25.idf):,} terms')

    # ── Serialize ─────────────────────────────────────────────────────────────
    index_path = OUT_DIR / 'bm25_index.pkl'
    ids_path   = OUT_DIR / 'corpus_ids.json'
    meta_path  = OUT_DIR / 'index_meta.json'

    print(f'\nSaving index → {index_path} …')
    with index_path.open('wb') as f:
        pickle.dump(bm25, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'  {index_path.stat().st_size / 1e6:.1f} MB written')

    ids_path.write_text(json.dumps(corpus_ids), encoding='utf-8')
    print(f'  Corpus IDs → {ids_path}')

    meta = {
        'k1'         : K1,
        'b'          : B,
        'n_docs'     : len(corpus_ids),
        'vocab_size' : len(bm25.idf),
        'corpus_path': str(CORPUS_PATH),
        'build_time_s': round(elapsed, 2),
        'built_at'   : time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'  Metadata   → {meta_path}')

    print('\nDone.')
    print(f'  Load with:  pickle.load(open("{index_path}", "rb"))')


if __name__ == '__main__':
    main()
