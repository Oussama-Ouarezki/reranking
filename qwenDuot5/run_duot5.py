"""Run duoT5 pairwise reranking on top-20 Qwen-ranked documents.

Pipeline:
  1. Load BM25 top-50 hits (with document text) from bm25_top50_test.jsonl
  2. Load Qwen scores from qwen_scores_test.jsonl
  3. For each query take the top-20 docs by qwen_prob
  4. Run duoT5 all-pairs tournament on those 20 docs
  5. Write results:
     - qwenDuot5/run_duot5.tsv  (TREC format, for ir-measures)
     - qwenDuot5/scores_duot5.jsonl  (per-query ranked list with scores)
     - qwenDuot5/ndcg10_mrr10.txt  (evaluation summary)
"""

import itertools
import json
from pathlib import Path

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import ir_measures
from ir_measures import nDCG, RR, Qrel, ScoredDoc

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
QWEN_SCORES   = ROOT / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
BM25_TOP50    = ROOT / "qwen4b_uncertainty/data/bm25_top50_test.jsonl"
QRELS_PATH    = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
CHECKPOINT    = ROOT / "checkpoints/duot5-base-msmarco"
OUT_DIR       = ROOT / "qwenDuot5"

TOURNAMENT_TOP_N = 20
MAX_LENGTH       = 1024
DOC_MAX_TOKENS   = 350
BATCH_SIZE       = 16

TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"


# ── duoT5 helpers ──────────────────────────────────────────────────────────────

def _truncate(text: str, max_tok: int = DOC_MAX_TOKENS) -> str:
    toks = text.split()
    return " ".join(toks[:max_tok]) if len(toks) > max_tok else text


def score_pairs(model, tokenizer, true_id, false_id, device,
                query: str, pairs: list[tuple[str, str]]) -> list[float]:
    inputs = [
        f"Query: {query} Document0: {_truncate(p0)} Document1: {_truncate(p1)} Relevant:"
        for p0, p1 in pairs
    ]
    enc = tokenizer(inputs, padding=True, truncation=True,
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


def tournament(model, tokenizer, true_id, false_id, device,
               query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """All-pairs tournament on candidates[:TOURNAMENT_TOP_N], tail pinned below."""
    head = candidates[:TOURNAMENT_TOP_N]
    tail = candidates[TOURNAMENT_TOP_N:]

    ids   = [c[0] for c in head]
    texts = [c[1] for c in head]
    agg   = {d: 0.0 for d in ids}

    all_pairs = list(itertools.permutations(range(len(ids)), 2))
    for start in range(0, len(all_pairs), BATCH_SIZE):
        batch      = all_pairs[start : start + BATCH_SIZE]
        pair_texts = [(texts[i], texts[j]) for i, j in batch]
        wins       = score_pairs(model, tokenizer, true_id, false_id, device, query, pair_texts)
        for (i, _j), p in zip(batch, wins):
            agg[ids[i]] += p

    ranked_head = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    if tail:
        tail_min    = min(s for _, s in ranked_head) if ranked_head else 0.0
        tail_scored = [(docid, tail_min - 1.0 - i * 1e-3)
                       for i, (docid, _text) in enumerate(tail)]
        return ranked_head + tail_scored
    return ranked_head


# ── data loading ───────────────────────────────────────────────────────────────

def load_bm25(path: Path) -> dict[str, dict]:
    """Returns {qid: {"query": str, "hits": {docid: contents}}}"""
    data = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = row["qid"]
            data[qid] = {
                "query": row["query"],
                "hits":  {h["docid"]: h["contents"] for h in row["hits"]},
            }
    return data


def load_qwen(path: Path) -> dict[str, list[dict]]:
    """Returns {qid: [{docid, qwen_prob, bm25_score}, ...]} sorted by qwen_prob desc."""
    data = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = row["qid"]
            scores = sorted(row["scores"], key=lambda x: x["qwen_prob"], reverse=True)
            data[qid] = scores
    return data


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Returns {qid: {docid: relevance}}"""
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, docid, rel = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading duoT5 from {CHECKPOINT} ...")

    tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
    model = T5ForConditionalGeneration.from_pretrained(
        str(CHECKPOINT),
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    true_id  = tokenizer.convert_tokens_to_ids(TOKEN_TRUE)
    false_id = tokenizer.convert_tokens_to_ids(TOKEN_FALSE)

    print("Loading data ...")
    bm25_data  = load_bm25(BM25_TOP50)
    qwen_data  = load_qwen(QWEN_SCORES)
    qrels_dict = load_qrels(QRELS_PATH)

    # Only process queries that are in qwen data
    qids = sorted(qwen_data.keys())
    print(f"Queries to rerank: {len(qids)}")

    OUT_DIR.mkdir(exist_ok=True)
    trec_lines    = []
    jsonl_records = []
    missing_text  = 0

    for qi, qid in enumerate(qids):
        bm25_entry = bm25_data.get(qid, {})
        query      = bm25_entry.get("query", "")
        hit_map    = bm25_entry.get("hits", {})

        # Top-20 by qwen_prob
        top20 = qwen_data[qid][:TOURNAMENT_TOP_N]

        candidates: list[tuple[str, str]] = []
        for item in top20:
            docid   = item["docid"]
            contents = hit_map.get(docid, "")
            if not contents:
                missing_text += 1
            candidates.append((docid, contents))

        if not candidates:
            continue

        ranked = tournament(model, tokenizer, true_id, false_id, device,
                            query, candidates)

        jsonl_records.append({"qid": qid, "query": query, "ranked": [
            {"rank": r + 1, "docid": docid, "duo_score": score}
            for r, (docid, score) in enumerate(ranked)
        ]})

        for rank, (docid, score) in enumerate(ranked, start=1):
            trec_lines.append(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\tduot5_qwen20")

        if (qi + 1) % 20 == 0 or (qi + 1) == len(qids):
            print(f"  [{qi+1}/{len(qids)}] done")

    # Write TREC run
    trec_path = OUT_DIR / "run_duot5.tsv"
    with open(trec_path, "w") as f:
        f.write("\n".join(trec_lines) + "\n")
    print(f"\nTREC run → {trec_path}")

    # Write per-query JSONL
    jsonl_path = OUT_DIR / "scores_duot5.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in jsonl_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Scores    → {jsonl_path}")

    if missing_text:
        print(f"Warning: {missing_text} doc(s) had no text in bm25 cache (scored as empty string)")

    # ── evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating ...")
    run_records = []
    for line in trec_lines:
        parts = line.split("\t")
        run_records.append(ScoredDoc(query_id=parts[0], doc_id=parts[2], score=float(parts[4])))

    qrel_records = []
    for qid, docs in qrels_dict.items():
        for docid, rel in docs.items():
            qrel_records.append(Qrel(query_id=qid, doc_id=docid, relevance=rel))

    metrics = ir_measures.calc_aggregate([nDCG @ 10, RR @ 10], qrel_records, run_records)

    eval_path = OUT_DIR / "ndcg10_mrr10.txt"
    with open(eval_path, "w") as f:
        for metric, value in metrics.items():
            line = f"{metric}\t{value:.4f}"
            print(f"  {line}")
            f.write(line + "\n")
    print(f"Eval      → {eval_path}")


if __name__ == "__main__":
    main()
