"""
Truncate BioASQ training corpus documents to a fixed token limit.

Reads from  : data/bioasq/processed/corpus.jsonl
Writes to   : data/training/truncated/corpus.jsonl

Only the 'text' field is truncated; 'title' and '_id' are kept as-is.
Tokenisation uses the actual T5 tokenizer (same as LiT5-Distill) so the
limit is in real model tokens, not whitespace words.  The truncated tokens
are decoded back to a string so the output is human-readable text.

Usage:
    python data/training/scripts/truncate_corpus.py
    python data/training/scripts/truncate_corpus.py --limit 256
    python data/training/scripts/truncate_corpus.py --limit 280 --tokenizer castorini/LiT5-Distill-base
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

BASE      = Path('/home/oussama/Desktop/reranking_project')
SRC_PATH  = BASE / 'data' / 'bioasq' / 'processed' / 'corpus.jsonl'
OUT_DIR   = BASE / 'data' / 'training' / 'truncated'
OUT_PATH  = OUT_DIR / 'corpus.jsonl'

DEFAULT_LIMIT     = 280
DEFAULT_TOKENIZER = str(BASE / 'checkpoints' / 'LiT5-Distill-base')


def truncate_text(text: str, tokenizer, limit: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= limit:
        return text
    return tokenizer.decode(ids[:limit], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description='Truncate corpus document texts to a real T5 token limit.'
    )
    parser.add_argument(
        '--limit', type=int, default=DEFAULT_LIMIT,
        help=f'Maximum number of T5 tokens per document text (default: {DEFAULT_LIMIT})'
    )
    parser.add_argument(
        '--tokenizer', type=str, default=DEFAULT_TOKENIZER,
        help='Tokenizer name or path (default: checkpoints/LiT5-Distill-base)'
    )
    args = parser.parse_args()

    print(f'Loading tokenizer from {args.tokenizer} …')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f'  Tokenizer loaded: {tokenizer.__class__.__name__}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = truncated = 0
    with SRC_PATH.open(encoding='utf-8') as src, \
         OUT_PATH.open('w', encoding='utf-8') as dst:
        for line in src:
            doc = json.loads(line)
            original_ids = tokenizer.encode(doc['text'], add_special_tokens=False)
            doc['text']  = truncate_text(doc['text'], tokenizer, args.limit)
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
    print(f'Token limit         : {args.limit} T5 tokens')
    print(f'Output              : {OUT_PATH}')


if __name__ == '__main__':
    main()
