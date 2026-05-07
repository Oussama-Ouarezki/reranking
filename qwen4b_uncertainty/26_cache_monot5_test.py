"""Cache monoT5-base-msmarco-100k scores for the Task13BGoldenEnriched test set.

Reads:  qwen4b_uncertainty/data/bm25_top50_test.jsonl
Writes: qwen4b_uncertainty/data/monot5_scores_test.jsonl
        one record per query: {qid, type, scores: [{docid, monot5_prob, bm25_score}]}

Resumable: skips qids already present in the output file.
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
IN   = BASE / "qwen4b_uncertainty/data/bm25_top50_test.jsonl"
OUT  = BASE / "qwen4b_uncertainty/data/monot5_scores_test.jsonl"

CHECKPOINT = str(BASE / "checkpoints/monot5-base-msmarco-100k")
TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"
MAX_LENGTH  = 350
BATCH_SIZE  = 50


class Scorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
        self.model = T5ForConditionalGeneration.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        self.true_id  = self.tokenizer.convert_tokens_to_ids(TOKEN_TRUE)
        self.false_id = self.tokenizer.convert_tokens_to_ids(TOKEN_FALSE)
        print(f"monoT5 loaded on {self.device}  "
              f"(true_id={self.true_id}, false_id={self.false_id})")

    def _score_batch(self, queries: list[str], passages: list[str]) -> list[float]:
        inputs = [f"Query: {q} Document: {p} Relevant:"
                  for q, p in zip(queries, passages)]
        enc = self.tokenizer(
            inputs, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(self.device)
        decoder_input = torch.zeros(
            (len(inputs), 1), dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input,
            ).logits  # (batch, 1, vocab)
        tf_logits = logits[:, 0, [self.true_id, self.false_id]]
        probs = torch.softmax(tf_logits, dim=-1)
        return probs[:, 0].cpu().tolist()

    def score(self, query: str, docs: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            out.extend(self._score_batch([query] * len(batch), batch))
        return out


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                continue
    return done


def main() -> None:
    queries = []
    with IN.open() as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"{len(queries)} queries from {IN}")

    done = already_done(OUT)
    if done:
        print(f"resuming: {len(done)} already cached")
    todo = [q for q in queries if q["qid"] not in done]
    print(f"{len(todo)} to score")

    scorer = Scorer()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for q in tqdm(todo, unit="q"):
            docs  = [h["contents"] for h in q["hits"]]
            probs = scorer.score(q["query"], docs)
            scores = [
                {"docid": h["docid"], "monot5_prob": p, "bm25_score": h["bm25_score"]}
                for h, p in zip(q["hits"], probs)
            ]
            f.write(json.dumps({"qid": q["qid"], "type": q["type"], "scores": scores}) + "\n")
            f.flush()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
