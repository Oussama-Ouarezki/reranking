"""Cache LiT5-Distill permutation over Qwen-top-20 (single window, no sliding).

For each test query: take the 20 docs with highest qwen_prob, feed to LiT5 in
qwen-descending order, record the LiT5-reordered docid list.

Reads:  qwen4b_uncertainty/data/bm25_top50_test.jsonl   (docid -> contents)
        qwen4b_uncertainty/data/qwen_scores_test.jsonl  (qwen_prob)
Writes: qwen4b_lit5/data/lit5_qwen_top20_test.jsonl
        one record per query: {qid, type, qwen_top20: [docid,...],
                               perm: [docid,...]}      # both length 20

Resumable: skips qids already present in the output file.
"""

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

BASE = Path(__file__).resolve().parents[1]
HITS_F = BASE / "qwen4b_uncertainty/data/bm25_top50_test.jsonl"
QWEN_F = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
OUT    = BASE / "qwen4b_lit5/data/lit5_qwen_top20_test.jsonl"

CHECKPOINT = str(BASE / "checkpoints/LiT5-Distill-base")
QUERY_PREFIX = "Search Query:"
PASS_PREFIX  = "Passage:"
SUFFIX       = " Relevance Ranking:"

TOP_K          = 20
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

    def rerank_window(
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


def load_contents() -> dict[str, dict[str, str]]:
    """qid -> {docid -> contents, '__query__' -> query, '__type__' -> type}"""
    out: dict[str, dict[str, str]] = {}
    with HITS_F.open() as f:
        for line in f:
            r = json.loads(line)
            d = {h["docid"]: h["contents"] for h in r["hits"]}
            d["__query__"] = r["query"]
            d["__type__"] = r["type"]
            out[r["qid"]] = d
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
    contents = load_contents()
    qwen_recs: list[dict] = []
    with QWEN_F.open() as f:
        for line in f:
            qwen_recs.append(json.loads(line))
    print(f"{len(qwen_recs)} queries from {QWEN_F}")

    done = already_done(OUT)
    if done:
        print(f"resuming: {len(done)} already cached")
    todo = [r for r in qwen_recs if r["qid"] not in done]
    print(f"{len(todo)} to score")

    scorer = LiT5Scorer()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for r in tqdm(todo, unit="q"):
            qid = r["qid"]
            qmap = contents[qid]
            query = qmap["__query__"]

            ranked = sorted(r["scores"], key=lambda s: s["qwen_prob"], reverse=True)
            top20 = ranked[:TOP_K]
            window = [(s["docid"], qmap[s["docid"]]) for s in top20]

            ordered = scorer.rerank_window(query, window)
            rec = {
                "qid": qid,
                "type": r["type"],
                "qwen_top20": [d for d, _ in window],
                "perm":       [d for d, _ in ordered],
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if scorer.device == "cuda":
                torch.cuda.empty_cache()
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
