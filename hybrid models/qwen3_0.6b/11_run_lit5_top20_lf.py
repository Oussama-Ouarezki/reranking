"""Cache LiT5-Distill ranking on top-20 by per-type LF score.

Per-type α* tuned for Recall@20:
    summary 0.99, factoid 0.99, list 0.99, yesno 0.82
"""

import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from ir_measures import ScoredDoc

from _common import BASE, load_qrels, load_qwen_scores, minmax, report, write_trec

LF_ALPHA = {"summary": 0.99, "factoid": 0.99, "list": 0.99, "yesno": 0.82}

BM25_F     = BASE / "qwen3_0.6b/data/bm25_top50_test.jsonl"
SCORES_F   = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
LIT5_CKPT  = BASE / "checkpoints/LiT5-Distill-base"
OUT_TREC   = BASE / "qwen3_0.6b/results/run_lf_lit5_top20.tsv"
OUT_JSON   = BASE / "qwen3_0.6b/results/metrics_lf_lit5_top20.json"
OUT_SCORES = BASE / "qwen3_0.6b/data/lit5_scores_lf.jsonl"

QUERY_PREFIX = "Search Query:"
PASS_PREFIX = "Passage:"
SUFFIX = " Relevance Ranking:"
WINDOW_SIZE = 20
TEXT_MAXLENGTH = 350
MAX_NEW_TOKENS = 140


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
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        a = LF_ALPHA.get(r["type"], 1.0)
        fused = a * q + (1.0 - a) * minmax(b)
        order = np.argsort(-fused)
        rank_by[r["qid"]] = [items[i]["docid"] for i in order]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading LiT5-Distill from {LIT5_CKPT}")
    tokenizer = T5Tokenizer.from_pretrained(str(LIT5_CKPT), legacy=False, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(
        str(LIT5_CKPT),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    run: list[ScoredDoc] = []
    jsonl_records = []
    qids = sorted(rank_by.keys())

    for qid in tqdm(qids, unit="q"):
        bm25_entry = bm25.get(qid)
        if not bm25_entry:
            continue
        query = bm25_entry["query"]
        hit_map = bm25_entry["hits"]
        ordered = rank_by[qid]
        head_ids = ordered[:WINDOW_SIZE]
        tail_ids = ordered[WINDOW_SIZE:]
        head_window = [(d, hit_map.get(d, "")) for d in head_ids]
        perm = rerank_window(model, tokenizer, device, query, head_window)
        head_ranked = [head_ids[i] for i in perm]

        n_head = len(head_ranked)
        n_total = n_head + len(tail_ids)
        per_qid = []
        for r_idx, docid in enumerate(head_ranked):
            score = float(n_total - r_idx)
            run.append(ScoredDoc(qid, docid, score))
            per_qid.append({"rank": r_idx + 1, "docid": docid, "score": score})
        for t_idx, docid in enumerate(tail_ids):
            score = -1.0 - t_idx * 1e-3
            run.append(ScoredDoc(qid, docid, score))
            per_qid.append({"rank": n_head + t_idx + 1, "docid": docid, "score": score})

        jsonl_records.append({"qid": qid, "type": qtypes.get(qid),
                              "alpha": LF_ALPHA.get(qtypes.get(qid, ""), 1.0),
                              "ranked": per_qid})

    metrics = report("Exp 7: Qwen+LF + LiT5 top-20", run, qrels, qtypes)
    write_trec(OUT_TREC, run, tag="qwen06b_lf_lit5_top20")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"alphas": LF_ALPHA, "metrics": metrics}, indent=2))
    OUT_SCORES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SCORES.open("w") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {OUT_TREC}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_SCORES}")


if __name__ == "__main__":
    main()
