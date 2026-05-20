"""Exp 5: duoT5 (general, MSMARCO — checkpoints/duot5-base-msmarco) tournament
on the top-20 docs by Qwen3-0.6B probability.

Same pipeline as 05_run_duot5_top20.py --base qwen but with a different checkpoint.
"""

import itertools
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, AutoTokenizer
from ir_measures import ScoredDoc

from _common import BASE, load_qrels, load_qwen_scores, report, write_trec

BM25_F     = BASE / "qwen3_0.6b/data/bm25_top50_test.jsonl"
SCORES_F   = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
DUOT5_CKPT = BASE / "checkpoints/duot5-base-msmarco"
OUT_TREC   = BASE / "qwen3_0.6b/results/run_qwen_duot5msmarco_top20.tsv"
OUT_JSON   = BASE / "qwen3_0.6b/results/metrics_qwen_duot5msmarco_top20.json"
OUT_SCORES = BASE / "qwen3_0.6b/data/duot5msmarco_scores_qwen.jsonl"

TOURNAMENT_TOP_N = 20
MAX_LENGTH       = 1024
DOC_MAX_TOKENS   = 350
BATCH_SIZE       = 16
TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"


def truncate(text: str, max_tok: int = DOC_MAX_TOKENS) -> str:
    toks = text.split()
    return " ".join(toks[:max_tok]) if len(toks) > max_tok else text


def score_pairs(model, tok, true_id, false_id, device, query, pairs):
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


def tournament(model, tok, true_id, false_id, device, query, candidates):
    head = candidates[:TOURNAMENT_TOP_N]
    tail = candidates[TOURNAMENT_TOP_N:]
    ids = [c[0] for c in head]
    texts = [c[1] for c in head]
    agg = {d: 0.0 for d in ids}
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
                       for i, (docid, _) in enumerate(tail)]
        return ranked_head + tail_scored
    return ranked_head


def main() -> None:
    rows = load_qwen_scores(SCORES_F)
    qtypes = {r["qid"]: r["type"] for r in rows}
    qrels = load_qrels()

    bm25 = {}
    with BM25_F.open() as f:
        for line in f:
            row = json.loads(line)
            bm25[row["qid"]] = {
                "query": row["query"],
                "hits": {h["docid"]: h["contents"] for h in row["hits"]},
            }

    rank_by: dict[str, list[str]] = {}
    for r in rows:
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        order = np.argsort(-q)
        rank_by[r["qid"]] = [items[i]["docid"] for i in order]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading duoT5 (msmarco) from {DUOT5_CKPT}")
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
    qids = sorted(rank_by.keys())

    for qid in tqdm(qids, unit="q"):
        bm25_entry = bm25.get(qid)
        if not bm25_entry:
            continue
        query = bm25_entry["query"]
        hit_map = bm25_entry["hits"]
        candidates = [(d, hit_map.get(d, "")) for d in rank_by[qid]]
        ranked = tournament(model, tok, true_id, false_id, device, query, candidates)
        for docid, score in ranked:
            run.append(ScoredDoc(qid, docid, float(score)))
        jsonl_records.append({
            "qid": qid, "type": qtypes.get(qid),
            "ranked": [{"rank": i + 1, "docid": d, "duo_score": s}
                       for i, (d, s) in enumerate(ranked)],
        })

    metrics = report("Exp 5: Qwen + duoT5 (MSMARCO) top-20", run, qrels, qtypes)
    write_trec(OUT_TREC, run, tag="qwen06b_duot5msmarco_top20")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    OUT_SCORES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SCORES.open("w") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {OUT_TREC}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_SCORES}")


if __name__ == "__main__":
    main()
