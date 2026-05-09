"""LiT5 ablation on top of LF α=0.999 (Qwen3-0.6B + BM25 minmax) — TEST set.

Three variants:
  A. LiT5 top-20 (single 20-passage window).
  B. LiT5 rolling window over all 50 (window=20, stride=10).
  C. LiT5 rolling window over top-30 (window=20, stride=10), tail 30..50 pinned.

Reads:  qwen3_0.6b/data/qwen06b_scores_test.jsonl
        qwen3_0.6b/data/bm25_top50_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: optimal alpha/results/lit5_ablation_lf999_<variant>.json
        optimal alpha/results/lit5_ablation_lf999.tsv  (summary)
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
import ir_measures
from ir_measures import nDCG, RR, Qrel, ScoredDoc

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
BM25_F   = BASE / "qwen3_0.6b/data/bm25_top50_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
LIT5_CKPT = BASE / "checkpoints/LiT5-Distill-base"
OUT_DIR  = BASE / "optimal alpha/results"

ALPHA = 0.999
WINDOW = 20
STRIDE = 10

QUERY_PREFIX = "Search Query:"
PASS_PREFIX = "Passage:"
SUFFIX = " Relevance Ranking:"
TEXT_MAXLENGTH = 350
MAX_NEW_TOKENS = 140

METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, RR @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@10"]
TYPES = ["summary", "factoid", "list", "yesno"]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def parse_ranking(perm_text: str, n: int) -> list[int]:
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


def rerank_window(model, tokenizer, device, query: str,
                  window: list[tuple[str, str]]) -> list[int]:
    n = len(window)
    strings = [
        f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i+1}] {text}{SUFFIX}"
        for i, (_, text) in enumerate(window)
    ]
    enc = tokenizer(
        strings, max_length=TEXT_MAXLENGTH, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        encoder_out = model.encoder(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        hidden = encoder_out.last_hidden_state.view(1, n * TEXT_MAXLENGTH, -1)
        attn = enc["attention_mask"].view(1, n * TEXT_MAXLENGTH)
        out = model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    perm_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return parse_ranking(perm_text, n)


def rolling_rerank(model, tokenizer, device, query: str,
                   docs: list[tuple[str, str]],
                   window: int = WINDOW, stride: int = STRIDE) -> list[str]:
    """RankGPT-style rolling rerank — bottom-up sliding window."""
    if len(docs) <= window:
        perm = rerank_window(model, tokenizer, device, query, docs)
        return [docs[i][0] for i in perm]
    order = list(docs)
    end = len(order)
    start = max(0, end - window)
    while True:
        win = order[start:end]
        perm = rerank_window(model, tokenizer, device, query, win)
        permuted = [win[i] for i in perm]
        order = order[:start] + permuted + order[end:]
        if start == 0:
            break
        end -= stride
        start = max(0, end - window)
    return [d for d, _ in order]


def load_qrels_test() -> list[Qrel]:
    qrels: list[Qrel] = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def evaluate_run(run: list[ScoredDoc], qrels: list[Qrel],
                 qtypes: dict[str, str]) -> dict:
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    out = {}
    for scope, qid_set in [("global", set(qtypes))] + [(t, type_qids[t]) for t in TYPES]:
        sub_run = [r for r in run if r.query_id in qid_set]
        sub_q = [q for q in qrels if q.query_id in qid_set]
        if not sub_run or not sub_q:
            out[scope] = {n: float("nan") for n in METRIC_NAMES}
        else:
            res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
            out[scope] = {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}
    return out


def main() -> None:
    # Load Qwen + BM25 scores
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    qtypes = {r["qid"]: r["type"] for r in rows}

    bm25 = {}
    with BM25_F.open() as f:
        for line in f:
            row = json.loads(line)
            bm25[row["qid"]] = {
                "query": row["query"],
                "hits": {h["docid"]: h["contents"] for h in row["hits"]},
            }

    # LF α=0.999 ranking per qid
    lf_rank: dict[str, list[str]] = {}
    for r in rows:
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        fused = ALPHA * q + (1.0 - ALPHA) * minmax(b)
        order = np.argsort(-fused)
        lf_rank[r["qid"]] = [items[i]["docid"] for i in order]

    qrels = load_qrels_test()

    # ── LiT5 model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading LiT5-Distill from {LIT5_CKPT}")
    tokenizer = T5Tokenizer.from_pretrained(str(LIT5_CKPT), legacy=False, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(LIT5_CKPT),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    qids = sorted(lf_rank.keys())

    def run_variant(name: str, head_size: int, rolling: bool):
        run: list[ScoredDoc] = []
        for qid in tqdm(qids, unit="q", desc=name):
            entry = bm25.get(qid)
            if not entry:
                continue
            query = entry["query"]
            hit_map = entry["hits"]
            ordered = lf_rank[qid]
            head_ids = ordered[:head_size]
            tail_ids = ordered[head_size:]
            head_pairs = [(d, hit_map.get(d, "")) for d in head_ids]

            if rolling:
                head_ranked = rolling_rerank(model, tokenizer, device, query,
                                             head_pairs, WINDOW, STRIDE)
            else:
                perm = rerank_window(model, tokenizer, device, query, head_pairs)
                head_ranked = [head_ids[i] for i in perm]

            n_head = len(head_ranked)
            for r_idx, docid in enumerate(head_ranked):
                run.append(ScoredDoc(qid, docid, float(n_head + 100 - r_idx)))
            for t_idx, docid in enumerate(tail_ids):
                run.append(ScoredDoc(qid, docid, -1.0 - t_idx * 1e-3))
        return run

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = [
        ("A. LiT5 top-20 (single window)", 20, False, "lf999_lit5_top20"),
        ("B. LiT5 rolling all 50 (w20,s10)", 50, True, "lf999_lit5_rolling50"),
        ("C. LiT5 rolling top-30 (w20,s10)", 30, True, "lf999_lit5_rolling30"),
    ]

    summary_rows = []
    all_metrics = {}
    for label, head_size, rolling, key in variants:
        out_json = OUT_DIR / f"lit5_ablation_{key}.json"
        if out_json.exists():
            print(f"\n[skip] {label} — already cached at {out_json}")
            metrics = json.loads(out_json.read_text())
        else:
            run = run_variant(label, head_size, rolling)
            metrics = evaluate_run(run, qrels, qtypes)
            out_json.write_text(json.dumps(metrics, indent=2))
        all_metrics[key] = metrics
        for scope in ["global"] + TYPES:
            row = {"variant": label, "scope": scope, **metrics[scope]}
            summary_rows.append(row)

        m = metrics["global"]
        print(f"\n=== {label} ===")
        print(f"  global: nDCG@1={m['ndcg@1']:.4f}  nDCG@5={m['ndcg@5']:.4f}  "
              f"nDCG@10={m['ndcg@10']:.4f}  MRR@10={m['mrr@10']:.4f}")

    # TSV summary
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    OUT_DIR.joinpath("lit5_ablation_lf999.tsv").write_text(df.to_csv(sep="\t", index=False))
    print(f"\nwrote {OUT_DIR / 'lit5_ablation_lf999.tsv'}")

    # Final comparison table
    print("\n" + "=" * 90)
    print("  FINAL — global metrics (after LF α=0.999)")
    print("=" * 90)
    print(f"  {'variant':<40}  " + "  ".join(f"{m:>9}" for m in METRIC_NAMES))
    for label, _, _, key in variants:
        m = all_metrics[key]["global"]
        vals = "  ".join(f"{m[k]:>9.4f}" for k in METRIC_NAMES)
        print(f"  {label:<40}  {vals}")


if __name__ == "__main__":
    main()
