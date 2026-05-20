"""Exp 3 / Exp 4: duoT5 tournament on top-20 from a base ranking.

Two modes:
  --base qwen   → top-20 by Qwen prob (Exp 3)
  --base lf     → top-20 by linear-fusion score (Exp 4)

Pipeline:
  1. Load BM25 top-50 hits (with text) from bm25_top50_test.jsonl
  2. Load base ranking (qwen prob or lf fused score)
  3. Take top-20 docs per query
  4. Run all-pairs duoT5 tournament; aggregate wins
  5. Tail (ranks 21-50) pinned below
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from ir_measures import ScoredDoc

from _common import (
    BASE, QWEN_LF_DYNAMIC10_ALPHA, load_qrels, load_qwen_scores,
    minmax, report, write_trec,
)

BM25_F   = BASE / "qwen3_0.6b/data/bm25_top50_test.jsonl"
SCORES_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
DUOT5_CKPT = BASE / "checkpoints/duot5-base-med-msmarco"

TOURNAMENT_TOP_N = 20
MAX_LENGTH       = 1024
DOC_MAX_TOKENS   = 350
BATCH_SIZE       = 16
TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"


def truncate(text: str, max_tok: int = DOC_MAX_TOKENS) -> str:
    toks = text.split()
    return " ".join(toks[:max_tok]) if len(toks) > max_tok else text


def score_pairs(model, tok, true_id, false_id, device,
                query: str, pairs: list[tuple[str, str]]) -> list[float]:
    inputs = [
        f"Query: {query} Document0: {truncate(p0)} Document1: {truncate(p1)} Relevant:"
        for p0, p1 in pairs
    ]
    enc = tok(inputs, padding=True, truncation=True,
              max_length=MAX_LENGTH, return_tensors="pt").to(device)
    decoder_input = torch.zeros((len(inputs), 1), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input,
        ).logits
    tf = logits[:, 0, [true_id, false_id]]
    return torch.softmax(tf, dim=-1)[:, 0].cpu().tolist()


def tournament(model, tok, true_id, false_id, device,
               query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    head = candidates[:TOURNAMENT_TOP_N]
    tail = candidates[TOURNAMENT_TOP_N:]
    ids   = [c[0] for c in head]
    texts = [c[1] for c in head]
    agg   = {d: 0.0 for d in ids}

    pairs = list(itertools.permutations(range(len(ids)), 2))
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start : start + BATCH_SIZE]
        pair_texts = [(texts[i], texts[j]) for i, j in batch]
        wins = score_pairs(model, tok, true_id, false_id, device, query, pair_texts)
        for (i, _j), p in zip(batch, wins):
            agg[ids[i]] += p

    ranked_head = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    if tail:
        tail_min = min(s for _, s in ranked_head) if ranked_head else 0.0
        tail_scored = [(docid, tail_min - 1.0 - i * 1e-3)
                       for i, (docid, _text) in enumerate(tail)]
        return ranked_head + tail_scored
    return ranked_head


def load_bm25_hits() -> dict[str, dict]:
    data = {}
    with BM25_F.open() as f:
        for line in f:
            row = json.loads(line)
            data[row["qid"]] = {
                "query": row["query"],
                "type": row["type"],
                "hits": {h["docid"]: h["contents"] for h in row["hits"]},
            }
    return data


def base_ranking(rows: list[dict], mode: str) -> dict[str, list[str]]:
    """Return {qid: [docid, ...]} ordered by base score desc."""
    out: dict[str, list[str]] = {}
    for r in rows:
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        if mode == "qwen":
            scores = q
        elif mode == "lf":
            alpha = QWEN_LF_DYNAMIC10_ALPHA.get(r["type"], 1.0)
            scores = alpha * q + (1.0 - alpha) * minmax(b)
        else:
            raise ValueError(mode)
        order = np.argsort(-scores)
        out[r["qid"]] = [items[i]["docid"] for i in order]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", choices=["qwen", "lf"], required=True,
                    help="qwen=Exp3, lf=Exp4")
    args = ap.parse_args()

    tag = f"qwen06b_{args.base}_duot5_top20"
    out_trec = BASE / f"qwen3_0.6b/results/run_{args.base}_duot5_top20.tsv"
    out_json = BASE / f"qwen3_0.6b/results/metrics_{args.base}_duot5_top20.json"
    out_scores = BASE / f"qwen3_0.6b/data/duot5_scores_{args.base}.jsonl"

    rows = load_qwen_scores(SCORES_F)
    qtypes = {r["qid"]: r["type"] for r in rows}
    qrels = load_qrels()
    bm25 = load_bm25_hits()
    rank_by = base_ranking(rows, args.base)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading duoT5 from {DUOT5_CKPT}")
    tok = AutoTokenizer.from_pretrained(str(DUOT5_CKPT))
    model = T5ForConditionalGeneration.from_pretrained(
        str(DUOT5_CKPT),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    true_id = tok.convert_tokens_to_ids(TOKEN_TRUE)
    false_id = tok.convert_tokens_to_ids(TOKEN_FALSE)

    run: list[ScoredDoc] = []
    jsonl_records = []
    missing = 0
    qids = sorted(rank_by.keys())

    for qid in tqdm(qids, unit="q"):
        bm25_entry = bm25.get(qid)
        if not bm25_entry:
            continue
        query = bm25_entry["query"]
        hit_map = bm25_entry["hits"]

        ordered_docids = rank_by[qid]
        candidates: list[tuple[str, str]] = []
        for docid in ordered_docids:
            text = hit_map.get(docid, "")
            if not text:
                missing += 1
            candidates.append((docid, text))

        ranked = tournament(model, tok, true_id, false_id, device, query, candidates)
        for docid, score in ranked:
            run.append(ScoredDoc(qid, docid, float(score)))
        jsonl_records.append({
            "qid": qid, "type": qtypes.get(qid),
            "ranked": [{"rank": i + 1, "docid": d, "duo_score": s}
                       for i, (d, s) in enumerate(ranked)],
        })

    if missing:
        print(f"Warning: {missing} doc(s) had no text in bm25 cache")

    label = "Exp 3: Qwen + duoT5 top-20" if args.base == "qwen" else "Exp 4: Qwen+LF + duoT5 top-20"
    metrics = report(label, run, qrels, qtypes)

    write_trec(out_trec, run, tag=tag)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    out_scores.parent.mkdir(parents=True, exist_ok=True)
    with out_scores.open("w") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {out_trec}")
    print(f"wrote {out_json}")
    print(f"wrote {out_scores}")


if __name__ == "__main__":
    main()
