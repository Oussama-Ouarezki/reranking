"""duoT5 pairwise reranker (all-pairs tournament).

Factored out from scripts/cascade_bioasq.py. For each candidate pair (d_i, d_j)
we score P(d_i > d_j); the aggregate score for a doc is the sum of its win
probabilities against every other doc.

Because pairwise tournament is O(n²), we cap the input to TOURNAMENT_TOP_N
(default 20) — caller is expected to pre-filter via monoT5 or BM25.
"""

import itertools
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

from .. import config

TOKEN_TRUE = "▁true"
TOKEN_FALSE = "▁false"

MAX_LENGTH = 1024
DOC_MAX_TOKENS = 200
TOURNAMENT_TOP_N = 20


def _truncate_doc(text: str, max_tokens: int = DOC_MAX_TOKENS) -> str:
    toks = text.split()
    return " ".join(toks[:max_tokens]) if len(toks) > max_tokens else text


class DuoT5Reranker:
    name = "duot5"

    def __init__(self, checkpoint=None, batch_size: int = 4, device: str | None = None):
        self.checkpoint = str(checkpoint or config.CHECKPOINTS["duot5"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.true_id = self.tokenizer.convert_tokens_to_ids(TOKEN_TRUE)
        self.false_id = self.tokenizer.convert_tokens_to_ids(TOKEN_FALSE)

    def _score_pairs(self, query: str, pairs: list[tuple[str, str]]) -> list[float]:
        inputs = [
            f"Query: {query} Document0: {_truncate_doc(p0)} Document1: {_truncate_doc(p1)} Relevant:"
            for p0, p1 in pairs
        ]
        enc = self.tokenizer(
            inputs, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(self.device)
        decoder_input = torch.zeros((len(inputs), 1), dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input,
            ).logits
        tf = logits[:, 0, [self.true_id, self.false_id]]
        return torch.softmax(tf, dim=-1)[:, 0].cpu().tolist()

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        # Keep a tail of low-relevance candidates pinned at the end so we can
        # safely return a full ranking even when truncating to TOURNAMENT_TOP_N.
        head = candidates[:TOURNAMENT_TOP_N]
        tail = candidates[TOURNAMENT_TOP_N:]

        ids = [c[0] for c in head]
        texts = [c[1] for c in head]
        agg = {d: 0.0 for d in ids}

        all_pairs = list(itertools.permutations(range(len(ids)), 2))
        for start in range(0, len(all_pairs), self.batch_size):
            batch = all_pairs[start : start + self.batch_size]
            pair_texts = [(texts[i], texts[j]) for i, j in batch]
            wins = self._score_pairs(query, pair_texts)
            for (i, _j), p in zip(batch, wins):
                agg[ids[i]] += p

        ranked_head = sorted(agg.items(), key=lambda x: x[1], reverse=True)
        # tail: assign descending small scores so they sit below all head docs
        if tail:
            tail_min = min(s for _, s in ranked_head) if ranked_head else 0.0
            tail_scored = [
                (docid, tail_min - 1.0 - i * 1e-3)
                for i, (docid, _text) in enumerate(tail)
            ]
            return ranked_head + tail_scored
        return ranked_head
