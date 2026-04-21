"""
Evaluate lit5_dataset1_train.jsonl reranking quality against BioASQ qrels.

Format:
  - ctxs: 20 candidate documents (docid + text)
  - target: space-separated 1-based indices giving the reranked order

Metrics at K = 5, 10, 20:
  - nDCG@K  — ranking quality (primary)
  - MRR     — position of first relevant doc
  - Recall@K — fraction of relevant docs retrieved
  - P@K      — precision (relevant / K)

Usage:
    python data/bioasq/training_strat/evaluate_lit5_dataset.py
    python data/bioasq/training_strat/evaluate_lit5_dataset.py --dataset lit5_dataset2_train.jsonl
"""

import json
import math
import argparse
from collections import defaultdict
from pathlib import Path

BASE     = Path('/home/oussama/Desktop/reranking_project')
QRELS    = BASE / 'data' / 'bioasq' / 'processed' / 'qrels.tsv'
DATA_DIR = BASE / 'data' / 'bioasq' / 'training_strat'

K_VALUES = [5, 10, 20]


# ── load qrels ────────────────────────────────────────────────────────────────
def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            if int(score) > 0:
                qrels[qid].add(did)
    return qrels


# ── metrics ───────────────────────────────────────────────────────────────────
def dcg(relevances, k):
    return sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(relevances[:k], start=1)
    )


def ndcg_at_k(ranked_rels, k):
    ideal = sorted(ranked_rels, reverse=True)
    idcg  = dcg(ideal, k)
    return dcg(ranked_rels, k) / idcg if idcg > 0 else 0.0


def mrr(ranked_rels):
    for rank, rel in enumerate(ranked_rels, start=1):
        if rel > 0:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_rels, total_relevant, k):
    if total_relevant == 0:
        return 0.0
    return sum(ranked_rels[:k]) / total_relevant


def precision_at_k(ranked_rels, k):
    return sum(ranked_rels[:k]) / k


# ── evaluate ──────────────────────────────────────────────────────────────────
def evaluate(dataset_path, qrels):
    scores = {
        'ndcg': {k: [] for k in K_VALUES},
        'mrr':  [],
        'recall': {k: [] for k in K_VALUES},
        'precision': {k: [] for k in K_VALUES},
    }
    skipped = 0

    with open(dataset_path, encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            qid  = rec['id']
            ctxs = rec['ctxs']

            # target: 1-based indices → reranked docid list
            order      = [int(x) - 1 for x in rec['target'].split()]
            ranked_ids = [ctxs[i]['docid'] for i in order]

            gold = qrels.get(qid, set())
            if not gold:
                skipped += 1
                continue

            ranked_rels = [1 if did in gold else 0 for did in ranked_ids]
            total_rel   = len(gold)

            scores['mrr'].append(mrr(ranked_rels))
            for k in K_VALUES:
                scores['ndcg'][k].append(ndcg_at_k(ranked_rels, k))
                scores['recall'][k].append(recall_at_k(ranked_rels, total_rel, k))
                scores['precision'][k].append(precision_at_k(ranked_rels, k))

    n = len(scores['mrr'])
    return scores, n, skipped


def mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='lit5_dataset1_train.jsonl')
    args = parser.parse_args()

    dataset_path = DATA_DIR / args.dataset

    print(f'Loading qrels from {QRELS} …')
    qrels = load_qrels(QRELS)
    print(f'  {sum(len(v) for v in qrels.values()):,} relevant pairs across {len(qrels):,} queries')

    print(f'\nEvaluating {dataset_path.name} …')
    scores, n, skipped = evaluate(dataset_path, qrels)
    print(f'  {n} queries evaluated  |  {skipped} skipped (no qrels)')

    sep = '─' * 58
    print(f'\n{sep}')
    print(f'  {"Metric":<16}' + ''.join(f'  {"@"+str(k):>8}' for k in K_VALUES))
    print(f'  {"──────":<16}' + ''.join(f'  {"────────":>8}' for _ in K_VALUES))

    print(f'  {"nDCG":<16}' + ''.join(
        f'  {mean(scores["ndcg"][k]):>8.4f}' for k in K_VALUES))
    print(f'  {"Recall":<16}' + ''.join(
        f'  {mean(scores["recall"][k]):>8.4f}' for k in K_VALUES))
    print(f'  {"Precision":<16}' + ''.join(
        f'  {mean(scores["precision"][k]):>8.4f}' for k in K_VALUES))
    print(f'  {"MRR":<16}  {mean(scores["mrr"]):>8.4f}  (no cutoff)')
    print(f'{sep}')
    print(f'  Dataset    : {args.dataset}')
    print(f'  Queries    : {n}')
    print(f'  Candidates : {20} per query (reranked)')
    print(f'{sep}\n')


if __name__ == '__main__':
    main()
