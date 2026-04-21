"""
Sentence-aware middle truncation of BioASQ training corpus.

Takes head_ratio of the token budget from the front (complete sentences) and
the remainder from the back, avoiding mid-sentence cuts and overlapping spans.

Reads from  : data/bioasq/processed/corpus.jsonl
Writes to   : data/training/truncated_middle/corpus.jsonl

Usage:
    python data/training/scripts/truncate_corpus_middle.py
    python data/training/scripts/truncate_corpus_middle.py --limit 280 --head-ratio 0.85
    python data/training/scripts/truncate_corpus_middle.py --limit 280 --head-ratio 0.90
"""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

BASE      = Path('/home/oussama/Desktop/reranking_project')
SRC_PATH  = BASE / 'data' / 'bioasq' / 'processed' / 'corpus.jsonl'
OUT_DIR   = BASE / 'data' / 'training' / 'truncated_middle'
OUT_PATH  = OUT_DIR / 'corpus.jsonl'

DEFAULT_LIMIT     = 280
DEFAULT_HEAD_RATIO = 0.85
DEFAULT_TOKENIZER = str(BASE / 'checkpoints' / 'LiT5-Distill-base')


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

    # avoid overlap when head and tail reach the same sentences
    head_set = set(id(s) for s in head_sents)
    tail_sents = [s for s in tail_sents if id(s) not in head_set]

    return ' '.join(head_sents + tail_sents)


def main():
    parser = argparse.ArgumentParser(
        description='Sentence-aware middle truncation of corpus texts.'
    )
    parser.add_argument('--limit', type=int, default=DEFAULT_LIMIT,
                        help=f'Max T5 tokens per document (default: {DEFAULT_LIMIT})')
    parser.add_argument('--head-ratio', type=float, default=DEFAULT_HEAD_RATIO,
                        help=f'Fraction of budget taken from the front (default: {DEFAULT_HEAD_RATIO})')
    parser.add_argument('--tokenizer', type=str, default=DEFAULT_TOKENIZER,
                        help='Tokenizer name or path')
    args = parser.parse_args()

    print(f'Loading tokenizer from {args.tokenizer} …')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f'  Tokenizer loaded: {tokenizer.__class__.__name__}')
    print(f'  Limit: {args.limit} tokens  |  head ratio: {args.head_ratio}  '
          f'(head={int(args.limit * args.head_ratio)}, tail={args.limit - int(args.limit * args.head_ratio)})')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = truncated = 0
    with SRC_PATH.open(encoding='utf-8') as src, \
         OUT_PATH.open('w', encoding='utf-8') as dst:
        for line in src:
            doc = json.loads(line)
            original_ids = tokenizer.encode(doc['text'], add_special_tokens=False)
            doc['text']  = truncate_sentence_aware(
                doc['text'], tokenizer, args.limit, args.head_ratio
            )
            if len(original_ids) > args.limit:
                truncated += 1
            dst.write(json.dumps(doc, ensure_ascii=False) + '\n')
            total += 1
            if total % 10_000 == 0:
                print(f'  {total:,} / 49,528 processed …', flush=True)

    kept_pct = 100 * (total - truncated) / total if total else 0
    print(f'\nDocuments processed : {total:,}')
    print(f'Truncated           : {truncated:,}  ({100 - kept_pct:.1f}%)')
    print(f'Already within limit: {total - truncated:,}  ({kept_pct:.1f}%)')
    print(f'Token limit         : {args.limit}  |  head ratio: {args.head_ratio}')
    print(f'Output              : {OUT_PATH}')


if __name__ == '__main__':
    main()
