"""Exp 8: LF (per-type α) → duoT5 on uncertainty band (LF-rank 15..25)
            → LiT5 on the resulting top-20.

α: summary=0.99, factoid=0.99, list=0.99, yesno=0.85

Pipeline per query:
  1. Rank docs by LF score.
  2. Take docs at LF-positions 15..25 (11 docs) — the "uncertainty band"
     around the top-20 boundary.
  3. duoT5 (checkpoints/duot5-base-msmarco) tournament on those 11 docs.
  4. New top-20 = LF positions 1..14 (unchanged) + top-6 from duoT5 reorder.
  5. LiT5-Distill listwise rerank on the new top-20.
  6. Tail = remaining docs in LF order, pinned strictly below the top-20.

Compares to: pure Qwen, LF only, and LF→LiT5 (cached).
"""

import itertools
import json

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from ir_measures import ScoredDoc

from _common import BASE, METRIC_NAMES, TYPES, evaluate, load_qrels, load_qwen_scores, minmax, report, write_trec

LF_ALPHA = {"summary": 0.99, "factoid": 0.99, "list": 0.99, "yesno": 0.85}

UNC_LO = 14   # 0-indexed: LF rank 15
UNC_HI = 25   # exclusive: LF ranks 15..25 inclusive
TOP_K = 20

BM25_F     = BASE / "qwen3_0.6b/data/bm25_top50_test.jsonl"
SCORES_F   = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
DUOT5_CKPT = BASE / "checkpoints/duot5-base-msmarco"
LIT5_CKPT  = BASE / "checkpoints/LiT5-Distill-base"
OUT_TREC   = BASE / "qwen3_0.6b/results/run_lf_duot5unc_lit5.tsv"
OUT_JSON   = BASE / "qwen3_0.6b/results/metrics_lf_duot5unc_lit5.json"
OUT_NEW20  = BASE / "qwen3_0.6b/data/lf_duot5unc_top20.jsonl"

# duoT5 settings
DUO_MAX_LEN = 1024
DUO_DOC_TOK = 350
DUO_BATCH = 16
TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"

# LiT5 settings
QUERY_PREFIX = "Search Query:"
PASS_PREFIX = "Passage:"
SUFFIX = " Relevance Ranking:"
TEXT_MAXLENGTH = 350
MAX_NEW_TOKENS = 140


def truncate(text: str, max_tok: int = DUO_DOC_TOK) -> str:
    toks = text.split()
    return " ".join(toks[:max_tok]) if len(toks) > max_tok else text


# ---------------- duoT5 ----------------
def duo_score_pairs(model, tok, true_id, false_id, device, query, pairs):
    inputs = [
        f"Query: {query} Document0: {truncate(p0)} Document1: {truncate(p1)} Relevant:"
        for p0, p1 in pairs
    ]
    enc = tok(inputs, padding=True, truncation=True,
              max_length=DUO_MAX_LEN, return_tensors="pt").to(device)
    decoder_input = torch.zeros((len(inputs), 1), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input,
        ).logits
    tf = logits[:, 0, [true_id, false_id]]
    return torch.softmax(tf, dim=-1)[:, 0].cpu().tolist()


def duo_tournament_band(model, tok, true_id, false_id, device, query, band):
    """band = [(docid, text), ...] (size up to 11). Returns ranked list of docids."""
    ids = [c[0] for c in band]
    texts = [c[1] for c in band]
    if len(ids) <= 1:
        return ids
    agg = {d: 0.0 for d in ids}
    pairs = list(itertools.permutations(range(len(ids)), 2))
    for start in range(0, len(pairs), DUO_BATCH):
        batch = pairs[start : start + DUO_BATCH]
        pair_texts = [(texts[i], texts[j]) for i, j in batch]
        wins = duo_score_pairs(model, tok, true_id, false_id, device, query, pair_texts)
        for (i, _j), p in zip(batch, wins):
            agg[ids[i]] += p
    return [d for d, _ in sorted(agg.items(), key=lambda x: x[1], reverse=True)]


# ---------------- LiT5 ----------------
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


def lit5_rerank_window(model, tokenizer, device, query, window):
    n = len(window)
    strings = [
        f"{QUERY_PREFIX} {query} {PASS_PREFIX} [{i+1}] {text}{SUFFIX}"
        for i, (_, text) in enumerate(window)
    ]
    enc = tokenizer(strings, max_length=TEXT_MAXLENGTH, padding="max_length",
                    truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        encoder_out = model.encoder(input_ids=enc["input_ids"],
                                    attention_mask=enc["attention_mask"])
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


# ---------------- Main ----------------
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

    # 1. LF ranking per query
    lf_order: dict[str, list[str]] = {}
    pure_qwen_order: dict[str, list[str]] = {}
    for r in rows:
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        a = LF_ALPHA.get(r["type"], 1.0)
        fused = a * q + (1.0 - a) * minmax(b)
        order = np.argsort(-fused)
        lf_order[r["qid"]] = [items[i]["docid"] for i in order]
        order_q = np.argsort(-q)
        pure_qwen_order[r["qid"]] = [items[i]["docid"] for i in order_q]

    # Reference runs
    ref_qwen: list[ScoredDoc] = []
    ref_lf: list[ScoredDoc] = []
    for qid, ord_ in pure_qwen_order.items():
        for i, d in enumerate(ord_):
            ref_qwen.append(ScoredDoc(qid, d, float(len(ord_) - i)))
    for qid, ord_ in lf_order.items():
        for i, d in enumerate(ord_):
            ref_lf.append(ScoredDoc(qid, d, float(len(ord_) - i)))

    # 2-3. duoT5 on uncertainty band
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading duoT5 (msmarco) from {DUOT5_CKPT}")
    duo_tok = AutoTokenizer.from_pretrained(str(DUOT5_CKPT))
    duo_model = T5ForConditionalGeneration.from_pretrained(
        str(DUOT5_CKPT),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    duo_model.eval()
    true_id = duo_tok.convert_tokens_to_ids(TOKEN_TRUE)
    false_id = duo_tok.convert_tokens_to_ids(TOKEN_FALSE)

    new_top20: dict[str, list[str]] = {}
    new_full_order: dict[str, list[str]] = {}  # for tail handling

    qids = sorted(lf_order.keys())
    for qid in tqdm(qids, unit="q", desc="duoT5 band"):
        bm25_entry = bm25.get(qid)
        if not bm25_entry:
            continue
        query = bm25_entry["query"]
        hit_map = bm25_entry["hits"]
        order = lf_order[qid]
        head14 = order[:UNC_LO]                 # positions 1..14
        band = order[UNC_LO:UNC_HI]             # positions 15..25
        rest = order[UNC_HI:]                   # positions 26..50

        band_payload = [(d, hit_map.get(d, "")) for d in band]
        band_ranked = duo_tournament_band(
            duo_model, duo_tok, true_id, false_id, device, query, band_payload,
        )
        band_top6 = band_ranked[:6]
        band_remainder = band_ranked[6:]

        new_top20[qid] = head14 + band_top6
        new_full_order[qid] = head14 + band_top6 + band_remainder + rest

    # Free duoT5
    del duo_model
    torch.cuda.empty_cache() if device == "cuda" else None

    # 5. LiT5 on new top-20
    print(f"Loading LiT5-Distill from {LIT5_CKPT}")
    lit5_tok = T5Tokenizer.from_pretrained(str(LIT5_CKPT), legacy=False, use_fast=True)
    lit5_model = T5ForConditionalGeneration.from_pretrained(
        str(LIT5_CKPT),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    lit5_model.eval()

    run: list[ScoredDoc] = []
    jsonl_records = []

    for qid in tqdm(qids, unit="q", desc="LiT5 top20"):
        bm25_entry = bm25.get(qid)
        if not bm25_entry:
            continue
        query = bm25_entry["query"]
        hit_map = bm25_entry["hits"]
        head_ids = new_top20[qid]
        head_window = [(d, hit_map.get(d, "")) for d in head_ids]
        perm = lit5_rerank_window(lit5_model, lit5_tok, device, query, head_window)
        head_ranked = [head_ids[i] for i in perm]

        full_order = new_full_order[qid]
        used = set(head_ranked)
        tail_ids = [d for d in full_order if d not in used]

        n_total = len(head_ranked) + len(tail_ids)
        per_qid = []
        for r_idx, docid in enumerate(head_ranked):
            score = float(n_total - r_idx)
            run.append(ScoredDoc(qid, docid, score))
            per_qid.append({"rank": r_idx + 1, "docid": docid, "score": score})
        for t_idx, docid in enumerate(tail_ids):
            score = -1.0 - t_idx * 1e-3
            run.append(ScoredDoc(qid, docid, score))
            per_qid.append({"rank": len(head_ranked) + t_idx + 1, "docid": docid, "score": score})

        jsonl_records.append({"qid": qid, "type": qtypes.get(qid),
                              "alpha": LF_ALPHA.get(qtypes.get(qid, ""), 1.0),
                              "ranked": per_qid})

    # 6. Reports
    print()
    m_qwen = report("Pure Qwen (BM25 top50 → Qwen prob)", ref_qwen, qrels, qtypes)
    m_lf   = report("LF only (α: 0.99/0.99/0.99/0.85)", ref_lf, qrels, qtypes)
    m_new  = report("Exp 8: LF → duoT5(unc 15..25) → LiT5 top20", run, qrels, qtypes)

    # Optional: read cached LF→LiT5 metrics for direct compare
    cached_lf_lit5 = BASE / "qwen3_0.6b/results/metrics_lf_lit5_top20.json"
    m_lf_lit5 = None
    if cached_lf_lit5.exists():
        try:
            data = json.loads(cached_lf_lit5.read_text())
            m_lf_lit5 = data.get("metrics", data)
        except Exception:
            m_lf_lit5 = None

    # Compact comparison
    print("\n" + "=" * 110)
    print("COMPARISON  (Δ = Exp 8 − reference)")
    print("=" * 110)
    refs = [("pure Qwen", m_qwen), ("LF only", m_lf)]
    if m_lf_lit5 is not None:
        refs.append(("LF → LiT5 top20", m_lf_lit5))
    for sc in ["global"] + TYPES:
        print(f"\n[{sc}]")
        print(f"  {'config':<28}  " + "  ".join(f"{m:>9}" for m in METRIC_NAMES))
        print(f"  {'Exp 8':<28}  " + "  ".join(f"{m_new[sc][m]:>9.4f}" for m in METRIC_NAMES))
        for name, ref in refs:
            print(f"  {name:<28}  " + "  ".join(f"{ref[sc][m]:>9.4f}" for m in METRIC_NAMES))
            deltas = "  ".join(f"{m_new[sc][m]-ref[sc][m]:>+9.4f}" for m in METRIC_NAMES)
            print(f"  {'  Δ vs '+name:<28}  {deltas}")

    write_trec(OUT_TREC, run, tag="lf_duot5unc_lit5")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "alphas": LF_ALPHA,
        "uncertainty_band": [UNC_LO + 1, UNC_HI],  # 1-indexed inclusive
        "top_k_lit5": TOP_K,
        "metrics": {
            "exp8_lf_duot5unc_lit5": m_new,
            "pure_qwen": m_qwen,
            "lf_only": m_lf,
            "lf_lit5_top20": m_lf_lit5,
        },
    }, indent=2))
    OUT_NEW20.parent.mkdir(parents=True, exist_ok=True)
    with OUT_NEW20.open("w") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")
    print(f"\nwrote {OUT_TREC}")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_NEW20}")


if __name__ == "__main__":
    main()
