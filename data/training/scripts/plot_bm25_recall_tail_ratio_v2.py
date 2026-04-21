"""
BM25 Recall@K sweep — tail-heavy ratios, optimised edition.

Optimisation: all sentence splits and token lengths are precomputed ONCE via
batch encoding; each ratio's truncation is pure Python arithmetic (no tokenizer
calls in the sweep loop).

tail_ratios evaluated: [0.8, 0.7, 0.6, 0.4, 0.3]
  tail=0.8 → head=56,  tail=224
  tail=0.7 → head=84,  tail=196
  tail=0.6 → head=112, tail=168
  tail=0.4 → head=168, tail=112
  tail=0.3 → head=196, tail=84

Saves: data/training/images/bm25_recall_tail_ratio_v2.png

Usage:
    python data/training/scripts/plot_bm25_recall_tail_ratio_v2.py
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
OUT_PNG      = BASE / 'data' / 'training' / 'images' / 'bm25_recall_tail_ratio_v2.png'

LIMIT        = 280
TAIL_RATIOS  = [0.8, 0.7, 0.6, 0.4, 0.3]
K_VALUES     = [10, 20, 100]
ENCODE_BATCH = 4096   # sentences per tokenizer batch


# ── Step 1: load raw corpus ───────────────────────────────────────────────────

def load_raw_corpus():
    ids, titles, texts = [], [], []
    with SRC_CORPUS.open(encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc['_id'])
            titles.append(doc.get('title', ''))
            texts.append(doc['text'])
    return ids, titles, texts


# ── Step 2: precompute sentence splits + token lengths (done ONCE) ────────────

def precompute(titles, texts, tokenizer):
    """
    Returns per-doc lists of (sentences, token_lengths, total_tokens).
    All tokenizer work happens here — never again during the ratio sweep.
    """
    print('  Splitting into sentences …', flush=True)
    # include title as first pseudo-sentence
    doc_sents: list[list[str]] = []
    for title, text in zip(titles, texts):
        combined = (title + ' ' + text).strip() if title else text
        sents = re.split(r'(?<=[.!?])\s+', combined.strip())
        doc_sents.append(sents)

    # flatten all sentences for batch encoding
    flat  = [s for sents in doc_sents for s in sents]
    total = len(flat)
    print(f'  Batch-encoding {total:,} sentences …', flush=True)

    flat_lens: list[int] = []
    for start in range(0, total, ENCODE_BATCH):
        batch = flat[start:start + ENCODE_BATCH]
        enc   = tokenizer(batch, add_special_tokens=False)
        flat_lens.extend(len(ids) for ids in enc['input_ids'])
        if (start // ENCODE_BATCH) % 10 == 0:
            print(f'    {min(start + ENCODE_BATCH, total):,} / {total:,}', flush=True)

    # reconstruct per-doc lengths
    doc_lens: list[list[int]] = []
    idx = 0
    for sents in doc_sents:
        n = len(sents)
        doc_lens.append(flat_lens[idx:idx + n])
        idx += n

    doc_totals = [sum(ln) for ln in doc_lens]
    return doc_sents, doc_lens, doc_totals


# ── Step 3: fast truncation using precomputed data ────────────────────────────

def truncate_fast(sents: list[str], lens: list[int],
                  total: int, limit: int, head_ratio: float) -> str:
    if total <= limit:
        return ' '.join(sents)

    head_budget = int(limit * head_ratio)
    tail_budget = limit - head_budget

    head_sents, head_count = [], 0
    for s, ln in zip(sents, lens):
        if head_count + ln > head_budget:
            break
        head_sents.append(s)
        head_count += ln

    tail_sents, tail_count = [], 0
    for s, ln in zip(reversed(sents), reversed(lens)):
        if tail_count + ln > tail_budget:
            break
        tail_sents.insert(0, s)
        tail_count += ln

    # drop tail sentences that overlap with head
    n_head      = len(head_sents)
    n_tail      = len(tail_sents)
    tail_start  = len(sents) - n_tail
    if tail_start < n_head:
        tail_sents = tail_sents[n_head - tail_start:]

    return ' '.join(head_sents + tail_sents)


def build_corpus_for_ratio(doc_ids, doc_sents, doc_lens, doc_totals,
                           limit: int, head_ratio: float) -> list[str]:
    return [
        truncate_fast(doc_sents[i], doc_lens[i], doc_totals[i], limit, head_ratio)
        for i in range(len(doc_ids))
    ]


# ── Step 4: BM25 build + recall eval ─────────────────────────────────────────

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


def evaluate(doc_ids, corpus_texts, queries, relevant) -> dict[int, float]:
    bm25  = BM25Okapi([t.lower().split() for t in corpus_texts], k1=0.7, b=0.9)
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

print('Loading tokenizer …')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
print(f'  {tokenizer.__class__.__name__} ready')

print('\nLoading raw corpus …')
raw_ids, raw_titles, raw_texts = load_raw_corpus()
print(f'  {len(raw_ids):,} documents')

print('\nPrecomputing sentence splits + token lengths …')
doc_sents, doc_lens, doc_totals = precompute(raw_titles, raw_texts, tokenizer)
print('  Precomputation done.\n')

print('Loading queries + qrels …')
queries, relevant = load_queries_and_qrels()
print(f'  {len(queries):,} queries  |  {sum(len(v) for v in relevant.values()):,} pairs\n')

results: dict[float, dict[int, float]] = {}

for tail_ratio in TAIL_RATIOS:
    head_ratio = 1.0 - tail_ratio
    head_tok   = int(LIMIT * head_ratio)
    tail_tok   = LIMIT - head_tok
    print(f'── tail_ratio={tail_ratio:.2f}  (head={head_tok}, tail={tail_tok}) ──')
    corpus_texts = build_corpus_for_ratio(
        raw_ids, doc_sents, doc_lens, doc_totals, LIMIT, head_ratio
    )
    print('  Building BM25 + evaluating …', flush=True)
    recall = evaluate(raw_ids, corpus_texts, queries, relevant)
    results[tail_ratio] = recall
    for k in K_VALUES:
        print(f'    Recall@{k:<4} : {recall[k]:.4f}')
    print()

# ── Summary table ─────────────────────────────────────────────────────────────
header = f"  {'tail_ratio':<12}" + ''.join(f"  {'Recall@'+str(k):>12}" for k in K_VALUES)
print(header)
print('  ' + '─' * (12 + 14 * len(K_VALUES)))
for tr in TAIL_RATIOS:
    row = f'  {tr:<12.2f}' + ''.join(f'  {results[tr][k]:>12.4f}' for k in K_VALUES)
    print(row)

# ── Plot ──────────────────────────────────────────────────────────────────────
n_ratios  = len(TAIL_RATIOS)
n_k       = len(K_VALUES)
bar_width = 0.2
x         = range(n_ratios)
colors    = ['#2196F3', '#4CAF50', '#FF9800']

fig, ax = plt.subplots(figsize=(11, 6))

for i, (k, color) in enumerate(zip(K_VALUES, colors)):
    offsets = [xi + (i - n_k / 2 + 0.5) * bar_width for xi in x]
    vals    = [results[tr][k] for tr in TAIL_RATIOS]
    bars    = ax.bar(offsets, vals, width=bar_width, color=color,
                     label=f'Recall@{k}', alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

x_labels = [
    f'tail={tr:.1f}\nh={int(LIMIT*(1-tr))} t={LIMIT - int(LIMIT*(1-tr))}'
    for tr in TAIL_RATIOS
]
ax.set_xticks(list(x))
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_xlabel('tail_ratio  (fraction of 280-token budget from the end)', fontsize=11)
ax.set_ylabel('Mean Recall@K', fontsize=12)
ax.set_title('BM25 Recall@K vs Tail Ratio (v2 — tail-heavy region)\n'
             'Sentence-Aware Truncation — BioASQ Training Set', fontsize=13)
ax.set_ylim(0, 0.75)
ax.legend(fontsize=10)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f'\nPlot saved → {OUT_PNG}')
