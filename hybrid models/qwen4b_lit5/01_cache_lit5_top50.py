"""Cache LiT5-Distill listwise permutations over BM25 top-50 (test set).

Mirrors application/backend/rerankers/lit5.py (FiD encoder, sliding window 20,
stride 10, FP16). Used downstream by:
  - 03_eval_lit5_bm25.py   pure LiT5 reranking of BM25
  - 05_eval_disagree_rrf.py disagreement-gated RRF with Qwen

Reads:  qwen4b_uncertainty/data/bm25_top50_test.jsonl
Writes: qwen4b_lit5/data/lit5_bm25_top50_test.jsonl
        one record per query: {qid, type, perm: [docid, ...]}   # 50 docids

Resumable: skips qids already present in the output file.
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

BASE = Path(__file__).resolve().parents[1]
IN  = BASE / "qwen4b_uncertainty/data/bm25_top50_test.jsonl"
OUT = BASE / "qwen4b_lit5/data/lit5_bm25_top50_test.jsonl"

CHECKPOINT = str(BASE / "checkpoints/LiT5-Distill-base")
QUERY_PREFIX = "Search Query:"
PASS_PREFIX  = "Passage:"
SUFFIX       = " Relevance Ranking:"

WINDOW_SIZE    = 20
STRIDE         = 10
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


class LiT5Scorer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(
            CHECKPOINT, legacy=False, use_fast=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()
        print(f"LiT5 loaded on {self.device}")

    def _rerank_window(
        self, query: str, window: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        n = len(window)
        strings = [
            f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i+1}] {text}{SUFFIX}"
            for i, (_, text) in enumerate(window)
        ]
        enc = self.tokenizer(
            strings,
            max_length=TEXT_MAXLENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            encoder_out = self.model.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
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
        self, query: str, candidates: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
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
        return ranked


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

    scorer = LiT5Scorer()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for q in tqdm(todo, unit="q"):
            cands = [(h["docid"], h["contents"]) for h in q["hits"]]
            ranked = scorer.rerank(q["query"], cands)
            rec = {
                "qid": q["qid"],
                "type": q["type"],
                "perm": [d for d, _ in ranked],
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if scorer.device == "cuda":
                torch.cuda.empty_cache()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
