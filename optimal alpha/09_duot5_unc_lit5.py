"""LF α=0.999 → duoT5 tournament on ranks 15..25 → top-20 → LiT5 listwise rerank.

Pipeline (TEST set):
  1. Qwen3-0.6B P(yes) on BM25 top-50.
  2. Linear-fusion blend with BM25 (α=0.999, both min-max-normalised per query).
  3. duoT5 tournament over LF positions 15..25 (uncertainty band, 11 docs).
  4. New top-20 = LF positions 1..14 + top-6 from the band reorder.
  5. LiT5-Distill listwise rerank of that new top-20 in a single 20-passage window.
  6. Tail = remaining docs in original LF/duoT5 order, pinned below.

Reads:  qwen3_0.6b/data/qwen06b_scores_test.jsonl
        qwen3_0.6b/data/bm25_top50_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: optimal alpha/results/lit5_ablation_lf999_duot5_unc_lit5.json
"""

import json
import sys
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
sys.path.insert(0, str(BASE / "application"))

from backend.rerankers.duot5 import DuoT5Reranker  # noqa: E402

SCORES_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
BM25_F   = BASE / "qwen3_0.6b/data/bm25_top50_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
LIT5_CKPT = BASE / "checkpoints/LiT5-Distill-base"
OUT_JSON = BASE / "optimal alpha/results/lit5_ablation_lf999_duot5_unc_lit5.json"

ALPHA = 0.999
UNC_LO, UNC_HI = 14, 25  # LF ranks 15..25 (0-indexed [14, 25))
HEAD_KEEP = 14           # keep LF top-14 unchanged
BAND_TOP6 = 6            # promote 6 from the duoT5 reorder into head
TOP20 = 20

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


def lit5_rerank_window(model, tokenizer, device, query: str,
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


def main() -> None:
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

    # LF α=0.999 ranking
    lf_rank: dict[str, list[str]] = {}
    for r in rows:
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        fused = ALPHA * q + (1.0 - ALPHA) * minmax(b)
        order = np.argsort(-fused)
        lf_rank[r["qid"]] = [items[i]["docid"] for i in order]

    qrels: list[Qrel] = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading duoT5…")
    duo = DuoT5Reranker(batch_size=16, device=device)
    print(f"Loading LiT5-Distill from {LIT5_CKPT}")
    tokenizer = T5Tokenizer.from_pretrained(str(LIT5_CKPT), legacy=False, use_fast=True)
    lit5 = T5ForConditionalGeneration.from_pretrained(
        str(LIT5_CKPT),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    lit5.eval()

    qids = sorted(lf_rank.keys())
    run: list[ScoredDoc] = []

    for qid in tqdm(qids, unit="q"):
        entry = bm25.get(qid)
        if not entry:
            continue
        query = entry["query"]
        hit_map = entry["hits"]
        ordered = lf_rank[qid]

        head14 = ordered[:UNC_LO]                # ranks 1..14
        band   = ordered[UNC_LO:UNC_HI]          # ranks 15..25
        rest   = ordered[UNC_HI:]                # ranks 26..50

        # duoT5 tournament on the 11-doc band
        if band:
            band_pairs = [(d, hit_map.get(d, "")) for d in band]
            band_ranked = duo.rerank(query, band_pairs)
            band_order = [d for d, _ in band_ranked]
        else:
            band_order = []

        band_top6 = band_order[:BAND_TOP6]
        band_rest = band_order[BAND_TOP6:]

        new_top20 = head14 + band_top6           # 14 + 6 = 20

        # LiT5 listwise on the new top-20
        win_pairs = [(d, hit_map.get(d, "")) for d in new_top20]
        perm = lit5_rerank_window(lit5, tokenizer, device, query, win_pairs)
        head_ranked = [new_top20[i] for i in perm]

        tail = band_rest + rest                  # below the LiT5 head
        n_head = len(head_ranked)
        for r_idx, docid in enumerate(head_ranked):
            run.append(ScoredDoc(qid, docid, float(n_head + 100 - r_idx)))
        for t_idx, docid in enumerate(tail):
            run.append(ScoredDoc(qid, docid, -1.0 - t_idx * 1e-3))

    # Evaluate
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    out_metrics = {}
    for scope, qid_set in [("global", set(qtypes))] + [(t, type_qids[t]) for t in TYPES]:
        sub_run = [r for r in run if r.query_id in qid_set]
        sub_q = [q for q in qrels if q.query_id in qid_set]
        if not sub_run or not sub_q:
            out_metrics[scope] = {n: float("nan") for n in METRIC_NAMES}
        else:
            res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
            out_metrics[scope] = {METRIC_NAMES[i]: float(res[METRICS[i]])
                                  for i in range(len(METRICS))}

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_metrics, indent=2))
    print(f"\n=== LF α=0.999 → duoT5(15-25) → LiT5 top-20 ===")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>9}" for m in METRIC_NAMES))
    for sc in ["global"] + TYPES:
        m = out_metrics[sc]
        vals = "  ".join(f"{m[k]:>9.4f}" for k in METRIC_NAMES)
        print(f"  {sc:<8}  {vals}")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
