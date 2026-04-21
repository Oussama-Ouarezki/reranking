"""
Build a BioASQ snippet corpus and training dataset from raw BioASQ JSON.

Each BioASQ question has a `snippets` list — exact passage extracts from PubMed
abstracts. This script treats every unique snippet as a retrievable passage and
produces:

  data/bioasq/processed/snippets/corpus.jsonl     — snippet corpus (BEIR format)
  data/bioasq/processed/snippets/queries.jsonl    — questions with ≥1 snippet
  data/bioasq/processed/snippets/qrels.tsv        — query → snippet relevance
  data/bioasq/finetune/train_triples.tsv          — (query, positive, negative) text
  data/bioasq/finetune/train_triples_ids.tsv      — (qid, pos_id, neg_id)

Snippet IDs: "{pmid}_{begin_offset}_{end_offset}"  (unique per passage position)
Negatives  : random snippets from other questions  (one negative per positive)

Usage:
    python scripts/prepare_snippet_corpus.py
    python scripts/prepare_snippet_corpus.py --neg-per-pos 3 --seed 42
"""

import json
import re
import random
import argparse
from collections import defaultdict
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_pmid(url: str) -> str | None:
    m = re.search(r'pubmed/(\d+)', url or '')
    return m.group(1) if m else None


def snippet_id(pmid: str, begin: int, end: int) -> str:
    return f"{pmid}_{begin}_{end}"


# ── core processing ───────────────────────────────────────────────────────────

def build_snippet_data(questions):
    """
    Returns:
        corpus   : {sid: text}
        qrels    : {qid: [sid, ...]}   (positive snippets per query)
        queries  : {qid: body}
    """
    corpus  = {}   # sid → text
    qrels   = defaultdict(list)
    queries = {}

    for q in questions:
        snippets = q.get('snippets', [])
        if not snippets:
            continue

        qid  = q['id']
        body = q['body']
        queries[qid] = body

        for sn in snippets:
            pmid = extract_pmid(sn.get('document', ''))
            if not pmid:
                continue

            begin = sn.get('offsetInBeginSection', 0)
            end   = sn.get('offsetInEndSection', 0)
            text  = sn.get('text', '').strip()

            if not text:
                continue

            sid = snippet_id(pmid, begin, end)
            corpus[sid] = text
            if sid not in qrels[qid]:
                qrels[qid].append(sid)

    return corpus, dict(qrels), queries


def build_triples(qrels, queries, corpus, neg_per_pos, rng):
    """
    For each (qid, positive_sid) pair, sample `neg_per_pos` negatives
    from snippets that are NOT relevant to qid.
    """
    all_sids      = list(corpus.keys())
    positive_sets = {qid: set(sids) for qid, sids in qrels.items()}

    triples_text = []   # (query_text, pos_text, neg_text)
    triples_ids  = []   # (qid, pos_sid, neg_sid)

    for qid, pos_sids in qrels.items():
        query_text  = queries[qid]
        pos_set     = positive_sets[qid]
        # candidate negatives: any snippet not relevant to this query
        negatives   = [s for s in all_sids if s not in pos_set]

        if not negatives:
            continue

        for pos_sid in pos_sids:
            sampled_negs = rng.sample(negatives, min(neg_per_pos, len(negatives)))
            for neg_sid in sampled_negs:
                triples_text.append((query_text, corpus[pos_sid], corpus[neg_sid]))
                triples_ids.append((qid, pos_sid, neg_sid))

    return triples_text, triples_ids


# ── I/O helpers ───────────────────────────────────────────────────────────────

def write_corpus(corpus, path, batch_size=5000):
    """Write corpus.jsonl in batches so large corpora don't require full RAM."""
    items = list(corpus.items())
    total = len(items)
    print(f"Writing corpus ({total} snippets) in batches of {batch_size}...")
    with open(path, 'w', encoding='utf-8') as f:
        for i in range(0, total, batch_size):
            batch = items[i : i + batch_size]
            for sid, text in batch:
                json.dump({'_id': sid, 'title': '', 'text': text},
                          f, ensure_ascii=False)
                f.write('\n')
            print(f"  [{min(i + batch_size, total)}/{total}]")
    print(f"  Saved → {path}")


def write_queries(queries, path):
    with open(path, 'w', encoding='utf-8') as f:
        for qid, text in queries.items():
            json.dump({'_id': qid, 'text': text}, f, ensure_ascii=False)
            f.write('\n')
    print(f"  Saved → {path}  ({len(queries)} queries)")


def write_qrels(qrels, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('query-id\tcorpus-id\tscore\n')
        for qid, sids in qrels.items():
            for sid in sids:
                f.write(f"{qid}\t{sid}\t1\n")
    total = sum(len(v) for v in qrels.values())
    print(f"  Saved → {path}  ({total} relevance pairs)")


def write_triples(triples_text, triples_ids, text_path, ids_path, batch_size=5000):
    total = len(triples_text)
    print(f"Writing {total} triples in batches of {batch_size}...")

    with open(text_path, 'w', encoding='utf-8') as ft, \
         open(ids_path,  'w', encoding='utf-8') as fi:

        ft.write('query\tpositive\tnegative\n')
        fi.write('qid\tpos_id\tneg_id\n')

        for i in range(0, total, batch_size):
            t_batch = triples_text[i : i + batch_size]
            i_batch = triples_ids [i : i + batch_size]

            for (q, p, n) in t_batch:
                ft.write(f"{q}\t{p}\t{n}\n")
            for (qid, pos, neg) in i_batch:
                fi.write(f"{qid}\t{pos}\t{neg}\n")

            print(f"  [{min(i + batch_size, total)}/{total}]")

    print(f"  Saved → {text_path}")
    print(f"  Saved → {ids_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build BioASQ snippet corpus and training triples."
    )
    parser.add_argument('--neg-per-pos', type=int, default=1,
                        help="Negatives sampled per positive (default: 1)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for negative sampling (default: 42)")
    args = parser.parse_args()

    base_dir    = Path('/home/oussama/Desktop/reranking_project')
    raw_file    = base_dir / 'data' / 'bioasq' / 'raw' / 'training13b.json'
    snippet_dir = base_dir / 'data' / 'bioasq' / 'processed' / 'snippets'
    finetune_dir = base_dir / 'data' / 'bioasq' / 'finetune'

    snippet_dir.mkdir(parents=True, exist_ok=True)
    finetune_dir.mkdir(parents=True, exist_ok=True)

    if not raw_file.exists():
        print(f"Error: {raw_file} not found.")
        return

    print("Loading BioASQ training data...")
    with open(raw_file, encoding='utf-8') as f:
        data = json.load(f)
    questions = data.get('questions', [])
    print(f"  {len(questions)} questions loaded")

    # ── build corpus / qrels / queries ───────────────────────────────────────
    print("\nExtracting snippets...")
    corpus, qrels, queries = build_snippet_data(questions)

    dropped = len(questions) - len(queries)
    print(f"  Unique snippets (corpus size) : {len(corpus)}")
    print(f"  Queries with snippets         : {len(queries)}")
    print(f"  Queries dropped (no snippets) : {dropped}")
    total_pairs = sum(len(v) for v in qrels.values())
    print(f"  Total query-snippet pairs     : {total_pairs}")

    # ── write corpus / queries / qrels ───────────────────────────────────────
    print()
    write_corpus(corpus,  snippet_dir / 'corpus.jsonl')
    write_queries(queries, snippet_dir / 'queries.jsonl')
    write_qrels(qrels,    snippet_dir / 'qrels.tsv')

    # ── build and write training triples ─────────────────────────────────────
    print(f"\nBuilding training triples  (neg_per_pos={args.neg_per_pos}, seed={args.seed})...")
    rng = random.Random(args.seed)
    triples_text, triples_ids = build_triples(
        qrels, queries, corpus, args.neg_per_pos, rng
    )

    print()
    write_triples(
        triples_text, triples_ids,
        finetune_dir / 'train_triples.tsv',
        finetune_dir / 'train_triples_ids.tsv',
    )

    print("\nDone.")
    print(f"  Snippet corpus  → {snippet_dir}/corpus.jsonl")
    print(f"  Queries         → {snippet_dir}/queries.jsonl")
    print(f"  Qrels           → {snippet_dir}/qrels.tsv")
    print(f"  Train triples   → {finetune_dir}/train_triples.tsv")
    print(f"  Train IDs       → {finetune_dir}/train_triples_ids.tsv")


if __name__ == '__main__':
    main()
