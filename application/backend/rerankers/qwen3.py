"""Qwen3-Reranker-4B — generative pointwise reranker, FP16.

The official Qwen3-Reranker recipe: format (instruction, query, doc) inside a
chat template, take the last-token logits, and read P(yes) / [P(yes)+P(no)] as
the relevance score.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reduce allocator fragmentation on GPUs with limited headroom after the 4B weights.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

CHECKPOINT = "Qwen/Qwen3-Reranker-4B"
INSTRUCTION = (
    "Given a biomedical question, retrieve PubMed abstracts that answer the question."
)
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# PubMed abstracts are ~300 words; 1024 tokens is ample and avoids the ~2 GiB
# activation spike that 8192 caused on 11 GiB GPUs already holding the model weights.
MAX_LENGTH = 1024


class Qwen3Reranker4B:
    name = "qwen3_reranker_4b"

    def __init__(self, batch_size: int = 1, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        self.token_yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_no = self.tokenizer.convert_tokens_to_ids("no")
        self.prefix_ids = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

    def _format_pair(self, query: str, doc: str) -> str:
        return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"

    def _score_batch(self, query: str, docs: list[str]) -> list[float]:
        bodies = [self._format_pair(query, d) for d in docs]
        body_max = MAX_LENGTH - len(self.prefix_ids) - len(self.suffix_ids)
        enc = self.tokenizer(
            bodies,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=body_max,
        )
        for i, ids in enumerate(enc["input_ids"]):
            enc["input_ids"][i] = self.prefix_ids + ids + self.suffix_ids

        padded = self.tokenizer.pad(
            enc, padding=True, return_tensors="pt", max_length=MAX_LENGTH
        )
        padded = {k: v.to(self.device) for k, v in padded.items()}

        with torch.no_grad():
            logits = self.model(**padded).logits[:, -1, :]  # (batch, vocab)

        yes_logit = logits[:, self.token_yes]
        no_logit = logits[:, self.token_no]
        stacked = torch.stack([no_logit, yes_logit], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = log_probs[:, 1].exp().float().cpu().tolist()
        return scores

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        docids = [c[0] for c in candidates]
        texts = [c[1] for c in candidates]
        scores: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            scores.extend(self._score_batch(query, texts[i : i + self.batch_size]))
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)
