"""
BM25 Recall@K sweep over head_ratio values for sentence-aware middle truncation.

For each head_ratio in [0.85, 0.90, 0.95, 1.0], truncates the corpus in memory,
builds a BM25 index, and computes mean Recall@10, @20, @100.
Plots results as a grouped bar chart.

Saves: data/training/images/bm25_recall_head_ratio.png

Usage:
    python data/training/scripts/plot_bm25_recall_head_ratio.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────
BASE         = Path('/home/oussama/Desktop/reranking_project')
SRC_CORPUS   = BASE / 'data' / 'bioasq' / 'processed' / 'corpus.jsonl'
QUERIES_FILE = BASE / 'data' / 'training' / 'truncated' / 'queries.jsonl'
QRELS_FILE   = BASE / 'data' / 'training' / 'truncated' / 'qrels.tsv'
TOKENIZER    = str(BASE / 'checkpoints' / 'LiT5-Distill-base')
OUT_PNG      = BASE / 'data' / 'training' / 'images' / 'bm25_recall_head_ratio.png'

LIMIT       = 280
HEAD_RATIOS = [0.85, 0.90, 0.95, 1.0]
K_VALUES    = [10, 20, 100]


# ── Truncation ────────────────────────────────────────────────────────────────

def truncate_sentence_aware(text: str, tokenizer, limit: int, head_ratio: float) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text

    head_budget = int(limit * head_ratio)
    tail_budget = limit - head_budget

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    head_sents, head_count = [], 0
    for sent in sentences:
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        if head_count + len(sent_ids) > head_budget:
            break
        head_sents.append(sent)
        head_count += len(sent_ids)

    tail_sents, tail_count = [], 0
    for sent in reversed(sentences):
        sent_ids = tokenizer.encode(sent, add_special_tokens=False)
        if tail_count + len(sent_ids) > tail_budget:
            break
        tail_sents.insert(0, sent)
        tail_count += len(sent_ids)

    head_set = set(id(s) for s in head_sents)
    tail_sents = [s for s in tail_sents if id(s) not in head_set]

    return ' '.join(head_sents + tail_sents)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_raw_corpus() -> tuple[list[str], list[str], list[str]]:
    ids, titles, texts = [], [], []
    with SRC_CORPUS.open(encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc['_id'])
            titles.append(doc.get('title', ''))
            texts.append(doc['text'])
    return ids, titles, texts


def load_queries_and_qrels() -> tuple[dict[str, str], dict[str, set[str]]]:
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


# ── BM25 recall ───────────────────────────────────────────────────────────────

def build_and_evaluate(doc_ids: list[str], corpus_texts: list[str],
                       queries: dict[str, str],
                       relevant: dict[str, set[str]]) -> dict[int, float]:
    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized, k1=0.7, b=0.9)

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


# ── Main sweep ────────────────────────────────────────────────────────────────

print('Loading tokenizer …')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
print(f'  {tokenizer.__class__.__name__} ready')

print('Loading raw corpus …')
raw_ids, raw_titles, raw_texts = load_raw_corpus()
print(f'  {len(raw_ids):,} documents')

print('Loading queries + qrels …')
queries, relevant = load_queries_and_qrels()
print(f'  {len(queries):,} queries  |  {sum(len(v) for v in relevant.values()):,} pairs\n')

# results[ratio][k] = mean recall
results: dict[float, dict[int, float]] = {}

for ratio in HEAD_RATIOS:
    label = f'head={ratio:.2f}' if ratio < 1.0 else 'head=1.00\n(head-only)'
    print(f'── head_ratio={ratio} ──────────────────────────')
    corpus_texts = [
        (raw_titles[i] + ' ' + truncate_sentence_aware(raw_texts[i], tokenizer, LIMIT, ratio)).strip()
        for i in range(len(raw_ids))
    ]
    print(f'  Building BM25 index …', flush=True)
    recall = build_and_evaluate(raw_ids, corpus_texts, queries, relevant)
    results[ratio] = recall
    for k in K_VALUES:
        print(f'    Recall@{k:<4} : {recall[k]:.4f}')
    print()

# ── Print summary table ───────────────────────────────────────────────────────
header = f"  {'head_ratio':<12}" + ''.join(f"  {'Recall@'+str(k):>12}" for k in K_VALUES)
print(header)
print('  ' + '─' * (12 + 14 * len(K_VALUES)))
for ratio in HEAD_RATIOS:
    row = f'  {ratio:<12.2f}' + ''.join(f'  {results[ratio][k]:>12.4f}' for k in K_VALUES)
    print(row)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_ratios = len(HEAD_RATIOS)
n_k      = len(K_VALUES)
bar_width = 0.2
x = range(n_ratios)
colors = ['#2196F3', '#4CAF50', '#FF9800']
labels = [f'Recall@{k}' for k in K_VALUES]

fig, ax = plt.subplots(figsize=(10, 6))

for i, (k, color, lbl) in enumerate(zip(K_VALUES, colors, labels)):
    offsets = [xi + (i - n_k / 2 + 0.5) * bar_width for xi in x]
    vals    = [results[r][k] for r in HEAD_RATIOS]
    bars    = ax.bar(offsets, vals, width=bar_width, color=color, label=lbl, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

ratio_labels = [f'{r:.2f}' if r < 1.0 else '1.00\n(head-only)' for r in HEAD_RATIOS]
ax.set_xticks(list(x))
ax.set_xticklabels(ratio_labels, fontsize=11)
ax.set_xlabel('head_ratio  (fraction of 280-token budget from the front)', fontsize=11)
ax.set_ylabel('Mean Recall@K', fontsize=12)
ax.set_title('BM25 Recall@K vs Head Ratio\nSentence-Aware Middle Truncation — BioASQ Training Set',
             fontsize=13)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=10)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f'\nPlot saved → {OUT_PNG}')
