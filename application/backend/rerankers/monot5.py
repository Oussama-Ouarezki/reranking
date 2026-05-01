"""monoT5 pointwise reranker.

Factored out from scripts/rerank_monot5.py::MonoT5Reranker. Same scoring
logic — softmax over the {▁true, ▁false} tokens at the first decoder step.
"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

from .. import config

TOKEN_TRUE = "▁true"
TOKEN_FALSE = "▁false"


class MonoT5Reranker:
    name = "monot5"

    def __init__(self, checkpoint=None, batch_size: int = 50, device: str | None = None):
        self.checkpoint = str(checkpoint or config.CHECKPOINTS["monot5"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.checkpoint,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        self.true_id = self.tokenizer.convert_tokens_to_ids(TOKEN_TRUE)
        self.false_id = self.tokenizer.convert_tokens_to_ids(TOKEN_FALSE)

    def _score_batch(self, queries: list[str], passages: list[str]) -> list[float]:
        inputs = [
            f"Query: {q} Document: {p} Relevant:"
            for q, p in zip(queries, passages)
        ]
        enc = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=350,
            return_tensors="pt",
        ).to(self.device)
        decoder_input = torch.zeros((len(inputs), 1), dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input,
            ).logits  # (batch, 1, vocab)

        tf_logits = logits[:, 0, [self.true_id, self.false_id]]  # (batch, 2)
        probs = torch.softmax(tf_logits, dim=-1)
        return probs[:, 0].cpu().tolist()

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
            batch = texts[i : i + self.batch_size]
            scores.extend(self._score_batch([query] * len(batch), batch))
        return sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)
