"""LiT5-Distill listwise reranker with sliding window.

Factored out from scripts/evaluate_lit5_bioasq_test.py. The model decodes a
permutation string like "[3] > [1] > [2] > ..." for each window of passages.

The sliding window starts at the tail end and steps backward by STRIDE, so
the most likely top-results are processed multiple times (standard LiT5 setup).
"""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from .. import config

QUERY_PREFIX = "Search Query:"
PASS_PREFIX = "Passage:"
SUFFIX = " Relevance Ranking:"

WINDOW_SIZE = 20
STRIDE = 10
TEXT_MAXLENGTH = 350
MAX_NEW_TOKENS = 140


def _parse_ranking(perm_text: str, n: int) -> list[int]:
    nums: list[int] = []
    seen: set[int] = set()
    for tok in perm_text.replace(",", " ").replace(">", " ").split():
        try:
            v = int(tok.strip("[]()."))
            if 1 <= v <= n and (v - 1) not in seen:
                nums.append(v - 1)
                seen.add(v - 1)
        except ValueError:
            continue
    for i in range(n):
        if i not in seen:
            nums.append(i)
    return nums


class LiT5Reranker:
    name = "lit5"

    def __init__(self, checkpoint=None, lora_adapter=None, device: str | None = None):
        self.checkpoint = str(checkpoint or config.CHECKPOINTS["lit5"])
        self.lora_adapter = str(lora_adapter) if lora_adapter else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Tokenizer: prefer the adapter dir if it ships one, else the base.
        tok_src = self.lora_adapter or self.checkpoint
        self.tokenizer = T5Tokenizer.from_pretrained(
            tok_src, legacy=False, use_fast=True
        )

        base = T5ForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.float16,
        ).to(self.device)

        if self.lora_adapter:
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(base, self.lora_adapter).to(self.device)
            # Merge LoRA into base weights for inference-time speed/parity with full FT.
            self.model = peft_model.merge_and_unload()
        else:
            self.model = base

        self.model.eval()

    def _rerank_window(self, query: str, window: list[tuple[str, str]]) -> list[tuple[str, str]]:
        n = len(window)
        strings = [
            f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i+1}] {text}{SUFFIX}"
            for i, (_, text) in enumerate(window)
        ]
        # Tokenize each (query, passage) pair independently — TEXT_MAXLENGTH tokens each
        enc = self.tokenizer(
            strings,
            max_length=TEXT_MAXLENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)  # [n, TEXT_MAXLENGTH]

        with torch.no_grad():
            # FiD: run the encoder on every passage independently, then
            # concatenate hidden states so the decoder attends to all passages.
            encoder_out = self.model.encoder(
                input_ids=enc["input_ids"],           # [n, TEXT_MAXLENGTH]
                attention_mask=enc["attention_mask"],  # [n, TEXT_MAXLENGTH]
            )
            # → [1, n * TEXT_MAXLENGTH, hidden_dim]
            hidden = encoder_out.last_hidden_state.view(1, n * TEXT_MAXLENGTH, -1)
            attn_mask = enc["attention_mask"].view(1, n * TEXT_MAXLENGTH)

            out = self.model.generate(
                encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                attention_mask=attn_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        perm_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        perm = _parse_ranking(perm_text, n)
        return [window[i] for i in perm]

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        ranked = list(candidates)
        n = len(ranked)
        start = max(0, n - WINDOW_SIZE)
        while True:
            end = min(start + WINDOW_SIZE, n)
            ranked[start:end] = self._rerank_window(query, ranked[start:end])
            if start == 0:
                break
            start = max(0, start - STRIDE)

        # LiT5 returns a permutation, not scores → assign descending score by rank
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(ranked)]
