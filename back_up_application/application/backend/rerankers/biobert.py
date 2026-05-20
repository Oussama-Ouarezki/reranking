"""BioBERT cross-encoder reranker.

Pointwise cross-encoder using nboost/pt-biobert-base-msmarco — bioBERT
fine-tuned on MS-MARCO passage ranking.  Takes (query, passage) pairs and
emits a single relevance logit; higher means more relevant.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CHECKPOINT = "NeuML/biomedbert-base-reranker"


class BioBERTReranker:
    name = "biobert"

    def __init__(self, batch_size: int = 16, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

    def _score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        enc = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc, return_dict=True).logits.view(-1).float()
        return logits.cpu().tolist()

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []
        docids = [c[0] for c in candidates]
        pairs = [(query, c[1]) for c in candidates]
        scores: list[float] = []
        for i in range(0, len(pairs), self.batch_size):
            scores.extend(self._score_batch(pairs[i : i + self.batch_size]))
        return sorted(zip(docids, scores), key=lambda x: x[1], reverse=True)
