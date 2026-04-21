"""
Compare BM25 Recall@20 and Recall@100 across all saved truncation variants.

Corpora evaluated:
  - truncated        (head, first 280 tokens)
  - truncated_left   (tail, last 280 tokens)
  - truncated_middle (sentence-aware, head_ratio=0.85)

No tokenizer needed — corpora are already on disk.
Saves: data/training/images/compare_truncation_recall.png

Usage:
    python data/training/scripts/compare_truncation_recall.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from rank_bm25 import BM25Okapi

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────
BASE         = Path('/home/oussama/Desktop/reranking_project')
QUERIES_FILE = BASE / 'data' / 'training' / 'truncated' / 'queries.jsonl'
QRELS_FILE   = BASE / 'data' / 'training' / 'truncated' / 'qrels.tsv'
OUT_PNG      = BASE / 'data' / 'training' / 'images' / 'compare_truncation_recall.png'

K_VALUES = [20, 100]

VARIANTS = [
    ('Head (first 280)',        BASE / 'data' / 'training' / 'truncated'        / 'corpus.jsonl'),
    ('Tail (last 280)',         BASE / 'data' / 'training' / 'truncated_left'   / 'corpus.jsonl'),
    ('Middle (sent-aware 0.85)',BASE / 'data' / 'training' / 'truncated_middle' / 'corpus.jsonl'),
]


# ── Load queries + qrels (shared) ─────────────────────────────────────────────

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


# ── BM25 eval ─────────────────────────────────────────────────────────────────

def evaluate(corpus_path: Path, queries, relevant) -> dict[int, float]:
    doc_ids, texts = [], []
    with corpus_path.open(encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_ids.append(doc['_id'])
            texts.append((doc.get('title', '') + ' ' + doc['text']).strip())

    bm25  = BM25Okapi([t.lower().split() for t in texts], k1=0.7, b=0.9)
    max_k = max(K_VALUES)
    k_set = set(K_VALUES)
    recall_at_k: dict[int, list[float]] = defaultdict(list)

    for qid, qtext in queries.items():
        rel = relevant.get(qid, set())
        if not rel:
            continue
        scores   = bm25.get_scores(qtext.lower().split())
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:max_k]
        hits = 0
        for rank, idx in enumerate(top_idxs, start=1):
            if doc_ids[idx] in rel:
                hits += 1
            if rank in k_set:
                recall_at_k[rank].append(hits / len(rel))

    return {k: sum(v) / len(v) for k, v in recall_at_k.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

print('Loading queries + qrels …')
queries, relevant = load_queries_and_qrels()
print(f'  {len(queries):,} queries  |  {sum(len(v) for v in relevant.values()):,} pairs\n')

results: dict[str, dict[int, float]] = {}

for label, corpus_path in VARIANTS:
    print(f'── {label}')
    print(f'   {corpus_path}')
    recall = evaluate(corpus_path, queries, relevant)
    results[label] = recall
    for k in K_VALUES:
        print(f'   Recall@{k:<4}: {recall[k]:.4f}')
    print()

# ── Summary table ─────────────────────────────────────────────────────────────
header = f"  {'Variant':<30}" + ''.join(f"  {'Recall@'+str(k):>12}" for k in K_VALUES)
print(header)
print('  ' + '─' * (30 + 14 * len(K_VALUES)))
for label, _ in VARIANTS:
    row = f'  {label:<30}' + ''.join(f'  {results[label][k]:>12.4f}' for k in K_VALUES)
    print(row)

# ── Plot ──────────────────────────────────────────────────────────────────────
labels    = [label for label, _ in VARIANTS]
bar_width = 0.3
x         = range(len(labels))
colors    = ['#4CAF50', '#FF9800']

fig, ax = plt.subplots(figsize=(10, 6))

for i, (k, color) in enumerate(zip(K_VALUES, colors)):
    offsets = [xi + (i - len(K_VALUES) / 2 + 0.5) * bar_width for xi in x]
    vals    = [results[lbl][k] for lbl in labels]
    bars    = ax.bar(offsets, vals, width=bar_width, color=color,
                     label=f'Recall@{k}', alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel('Mean Recall@K', fontsize=12)
ax.set_title('BM25 Recall@20 and Recall@100\nAcross Truncation Strategies — BioASQ Training Set',
             fontsize=13)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=11)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f'\nPlot saved → {OUT_PNG}')
