"""Cache Qwen3-Reranker-4B scores for the Task13BGoldenEnriched test set.

Reads:  qwen4b_uncertainty/data/bm25_top50_test.jsonl
Writes: qwen4b_uncertainty/data/qwen_scores_test.jsonl
        one record per query: {qid, type, scores: [{docid, qwen_prob, bm25_score}]}

Resumable: skips qids already present in the output file.
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parents[1]
IN = BASE / "qwen4b_uncertainty/data/bm25_top50_test.jsonl"
OUT = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"

CHECKPOINT = "Qwen/Qwen3-Reranker-4B"
INSTRUCTION = "Given a biomedical question, retrieve PubMed abstracts that answer the question."
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_LENGTH = 1024
BATCH_SIZE = 1


class Scorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def _format(self, q: str, d: str) -> str:
        return f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i : i + BATCH_SIZE]
            bodies = [self._format(query, d) for d in batch]
            body_max = MAX_LENGTH - len(self.prefix_ids) - len(self.suffix_ids)
            enc = self.tokenizer(
                bodies, padding=False, truncation="longest_first",
                return_attention_mask=False, max_length=body_max,
            )
            for j, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][j] = self.prefix_ids + ids + self.suffix_ids
            padded = self.tokenizer.pad(
                enc, padding=True, return_tensors="pt", max_length=MAX_LENGTH
            )
            padded = {k: v.to(self.device) for k, v in padded.items()}
            with torch.no_grad():
                logits = self.model(**padded).logits[:, -1, :]
            yes_l = logits[:, self.token_yes]
            no_l = logits[:, self.token_no]
            stacked = torch.stack([no_l, yes_l], dim=1)
            log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
            out.extend(log_probs[:, 1].exp().float().cpu().tolist())
            if self.device == "cuda":
                torch.cuda.empty_cache()
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
            docs = [h["contents"] for h in q["hits"]]
            probs = scorer.score(q["query"], docs)
            if probs and max(probs) < 0.01:
                print(f"  warn: degenerate scores for {q['qid']} (max={max(probs):.4f})")
            scores = [
                {"docid": h["docid"], "qwen_prob": p, "bm25_score": h["bm25_score"]}
                for h, p in zip(q["hits"], probs)
            ]
            f.write(json.dumps({"qid": q["qid"], "type": q["type"], "scores": scores}) + "\n")
            f.flush()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
