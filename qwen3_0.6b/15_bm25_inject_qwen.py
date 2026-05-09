"""Cache Qwen3-Reranker-0.6B scores with BM25 prior injected into the prompt.

Tests two text-injection formats:
  norm    →  "[BM25=0.85] {doc}"            two-decimal min-max score per query
  bucket  →  "[BM25=high] {doc}"            tertile bucket (low / medium / high)

Each produces its own score file:
  qwen3_0.6b/data/qwen06b_scores_test_inject_norm.jsonl
  qwen3_0.6b/data/qwen06b_scores_test_inject_bucket.jsonl

Then evaluates Pure-Qwen ordering for: baseline (no inject), norm, bucket.
Reports global + per-type nDCG@1/@5/@10 and MRR@10.

Reads:   qwen3_0.6b/data/bm25_top50_test.jsonl
         qwen3_0.6b/data/qwen06b_scores_test.jsonl   (baseline, already cached)
Writes:  qwen3_0.6b/data/qwen06b_scores_test_inject_<mode>.jsonl
         qwen3_0.6b/results/bm25_inject_qwen_compare.json
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from ir_measures import ScoredDoc

from _common import (
    BASE, METRIC_NAMES, TYPES, evaluate, load_qrels, load_qwen_scores,
)

CHECKPOINT = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = "Given a biomedical question, retrieve PubMed abstracts that answer the question."
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
MAX_LENGTH = 1024
BATCH_SIZE = 8

BM25_F      = BASE / "qwen3_0.6b/data/bm25_top50_test.jsonl"
BASELINE_F  = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"


def inject_norm(text: str, score: float, lo: float, hi: float) -> str:
    span = hi - lo
    val = 0.0 if span <= 0 else (score - lo) / span
    return f"[BM25={val:.2f}] {text}"


def inject_bucket(text: str, score: float, t1: float, t2: float) -> str:
    if score >= t2:
        tag = "high"
    elif score >= t1:
        tag = "medium"
    else:
        tag = "low"
    return f"[BM25={tag}] {text}"


class Scorer:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(CHECKPOINT, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        self.tok_yes = self.tok.convert_tokens_to_ids("yes")
        self.tok_no  = self.tok.convert_tokens_to_ids("no")
        self.pre_ids = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suf_ids = self.tok.encode(SUFFIX, add_special_tokens=False)

    def _format(self, q: str, d: str) -> str:
        return f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        out: list[float] = []
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            bodies = [self._format(query, d) for d in batch]
            body_max = MAX_LENGTH - len(self.pre_ids) - len(self.suf_ids)
            enc = self.tok(bodies, padding=False, truncation="longest_first",
                           return_attention_mask=False, max_length=body_max)
            for j, ids in enumerate(enc["input_ids"]):
                enc["input_ids"][j] = self.pre_ids + ids + self.suf_ids
            padded = self.tok.pad(enc, padding=True, return_tensors="pt",
                                  max_length=MAX_LENGTH)
            padded = {k: v.to(self.device) for k, v in padded.items()}
            with torch.no_grad():
                logits = self.model(**padded).logits[:, -1, :]
            yes_l = logits[:, self.tok_yes]
            no_l  = logits[:, self.tok_no]
            stacked = torch.stack([no_l, yes_l], dim=1)
            lp = torch.nn.functional.log_softmax(stacked, dim=1)
            out.extend(lp[:, 1].exp().float().cpu().tolist())
            if self.device == "cuda":
                torch.cuda.empty_cache()
        return out


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["qid"])
            except Exception:
                continue
    return done


def cache_scores(mode: str, out_path: Path) -> None:
    """mode in {'norm', 'bucket'}.  Reuses bm25 file."""
    done = already_done(out_path)
    print(f"[{mode}] resume: {len(done)} qids already scored")

    rows = []
    with BM25_F.open() as f:
        for line in f:
            r = json.loads(line)
            if r["qid"] in done:
                continue
            rows.append(r)
    if not rows:
        print(f"[{mode}] nothing to do.")
        return

    scorer = Scorer()
    with out_path.open("a") as out_f:
        for r in tqdm(rows, unit="q", desc=f"qwen[{mode}]"):
            qid = r["qid"]
            qtype = r.get("type")
            query = r["query"]
            hits = r["hits"]
            scores_b = [float(h["bm25_score"]) for h in hits]
            lo, hi = min(scores_b), max(scores_b)
            srt = sorted(scores_b)
            n = len(srt)
            t1 = srt[n // 3] if n >= 3 else srt[0]
            t2 = srt[(2 * n) // 3] if n >= 3 else srt[-1]

            docs_in = []
            for h in hits:
                txt = h["contents"]
                s = float(h["bm25_score"])
                if mode == "norm":
                    docs_in.append(inject_norm(txt, s, lo, hi))
                elif mode == "bucket":
                    docs_in.append(inject_bucket(txt, s, t1, t2))
                else:
                    docs_in.append(txt)

            probs = scorer.score(query, docs_in)
            rec = {
                "qid": qid,
                "type": qtype,
                "scores": [
                    {"docid": h["docid"],
                     "qwen_prob": float(p),
                     "bm25_score": float(h["bm25_score"])}
                    for h, p in zip(hits, probs)
                ],
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()


def evaluate_variant(jsonl_path: Path, label: str, qrels) -> dict:
    rows = load_qwen_scores(jsonl_path)
    qtypes = {r["qid"]: r["type"] for r in rows}
    run = []
    for r in rows:
        for s in r["scores"]:
            run.append(ScoredDoc(r["qid"], s["docid"], float(s["qwen_prob"])))
    out = {"global": evaluate(run, qrels)}
    type_qids: dict[str, set[str]] = {t: set() for t in TYPES}
    for qid, t in qtypes.items():
        if t in type_qids:
            type_qids[t].add(qid)
    for t in TYPES:
        out[t] = evaluate(run, qrels, type_qids[t])
    print(f"\n=== {label} ===")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>9}" for m in METRIC_NAMES))
    for sc in ["global"] + TYPES:
        vals = "  ".join(f"{out[sc][m]:>9.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {vals}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+",
                        default=["norm", "bucket"],
                        choices=["norm", "bucket"])
    parser.add_argument("--skip-scoring", action="store_true",
                        help="Only run the evaluation phase.")
    args = parser.parse_args()

    out_paths = {
        m: BASE / f"qwen3_0.6b/data/qwen06b_scores_test_inject_{m}.jsonl"
        for m in args.modes
    }
    if not args.skip_scoring:
        for mode, p in out_paths.items():
            cache_scores(mode, p)

    qrels = load_qrels()
    results: dict[str, dict] = {}
    results["baseline"] = evaluate_variant(BASELINE_F, "Baseline (no inject)",
                                           qrels)
    for mode, p in out_paths.items():
        results[f"inject_{mode}"] = evaluate_variant(
            p, f"+ BM25 inject [{mode}]", qrels)

    # Compact comparison
    print("\n" + "=" * 100)
    print("DELTAS vs baseline   (positive = injection helped)")
    print("=" * 100)
    for variant in [k for k in results if k != "baseline"]:
        print(f"\n[{variant}]")
        print(f"  {'scope':<8}  " + "  ".join(f"{m:>9}" for m in METRIC_NAMES))
        for sc in ["global"] + TYPES:
            d = [results[variant][sc][m] - results["baseline"][sc][m]
                 for m in METRIC_NAMES]
            print(f"  {sc:<8}  " + "  ".join(f"{x:>+9.4f}" for x in d))

    out_json = BASE / "qwen3_0.6b/results/bm25_inject_qwen_compare.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
