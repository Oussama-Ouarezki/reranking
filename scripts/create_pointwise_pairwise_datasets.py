"""
Create pointwise (monoT5) and pairwise (duoT5) fine-tuning datasets from the
DeepSeek listwise-reranked BioASQ data using hard negative mining.

Hard negative strategy:
  1. Positives  : gold_passages from candidates.jsonl (qrel-verified, text included)
  2. Hard negs  : from the 88 pre-mined CE-hard negatives in candidates.jsonl,
                  re-ranked by their position in the DeepSeek permutation.
                  Docs the LLM also rated highly but are not gold = hardest negatives.
  3. Easy negs  : 2 easy negatives per query for calibration.

Outputs (in data/bioasq/finetune/):
  pointwise_monot5.tsv        — query \t passage \t label (1/0)
  pointwise_monot5_ids.tsv    — qid \t doc_id \t label
  pairwise_duot5.tsv          — query \t doc_a \t doc_b \t label (0=A wins, 1=B wins)
  pairwise_duot5_ids.tsv      — qid \t doc_a_id \t doc_b_id \t label
"""

import json
import random
import csv
import os
from collections import defaultdict

random.seed(42)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT        = "/home/oussama/Desktop/reranking_project"
LISTWISE    = f"{ROOT}/data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl"
CANDIDATES  = f"{ROOT}/data/bioasq/hard_negatives_full/candidates.jsonl"
OUT_DIR     = f"{ROOT}/data/bioasq/finetune"

# hard-negative mining params
MAX_HARD_NEG_PER_QUERY = 10   # top-K from DeepSeek-ranked hard negatives
MAX_EASY_NEG_PER_QUERY = 2    # easy negatives for calibration
# pairwise: we create both (pos, neg) and (neg, pos) so label distribution is 50/50
MAX_HARD_NEG_PER_POS   = 5    # limit pairs per positive to control dataset size

# ── load DeepSeek permutation (reranker-ranked order per query) ────────────────
print("Loading DeepSeek permutation …")
deepseek_perm = {}       # qid → {docid: rank}   (rank 0 = best)
with open(LISTWISE) as f:
    for line in f:
        d = json.loads(line)
        deepseek_perm[d["qid"]] = {docid: i for i, docid in enumerate(d["permutation"])}

# ── load candidates (gold passages + hard negatives with text) ─────────────────
print("Loading candidates …")
candidates = {}
with open(CANDIDATES) as f:
    for line in f:
        d = json.loads(line)
        candidates[d["qid"]] = d

print(f"Loaded {len(deepseek_perm)} listwise queries, {len(candidates)} candidate queries")

# ── build dataset rows ─────────────────────────────────────────────────────────
pointwise_rows = []   # (qid, doc_id, query_text, passage_text, label)
pairwise_rows  = []   # (qid, doc_a_id, doc_b_id, query_text, text_a, text_b, label)

skipped_no_pos = 0
skipped_no_neg = 0

for qid, perm_ranks in deepseek_perm.items():
    if qid not in candidates:
        continue
    cand = candidates[qid]
    query_text = cand["query"]

    # ── positives ──────────────────────────────────────────────────────────────
    gold = cand["gold_passages"]          # list of {docid, title, text}
    if not gold:
        skipped_no_pos += 1
        continue

    # ── hard negatives ranked by DeepSeek permutation position ────────────────
    # Lower permutation rank = higher confidence from DeepSeek = harder negative
    hard_negs_sorted = sorted(
        cand["hard_negatives"],
        key=lambda x: perm_ranks.get(x["docid"], 9999)
    )
    hard_negs = hard_negs_sorted[:MAX_HARD_NEG_PER_QUERY]

    if not hard_negs:
        skipped_no_neg += 1
        continue

    easy_negs = cand["easy_negatives"][:MAX_EASY_NEG_PER_QUERY]

    # helper: build passage text as title + text (truncate generously — tokenizer
    # will do the real truncation during training)
    def passage_text(entry):
        title = entry.get("title", "").strip()
        body  = entry.get("text", "").strip()
        return (title + " " + body).strip() if title else body

    # ── pointwise rows ─────────────────────────────────────────────────────────
    for pos in gold:
        pointwise_rows.append((qid, pos["docid"], query_text, passage_text(pos), 1))

    for neg in hard_negs:
        pointwise_rows.append((qid, neg["docid"], query_text, passage_text(neg), 0))

    for neg in easy_negs:
        pointwise_rows.append((qid, neg["docid"], query_text, passage_text(neg), 0))

    # ── pairwise rows ──────────────────────────────────────────────────────────
    negs_for_pairs = hard_negs[:MAX_HARD_NEG_PER_POS]
    for pos in gold:
        for neg in negs_for_pairs:
            # (pos, neg) → label 0  means doc_a (pos) is more relevant
            pairwise_rows.append((
                qid, pos["docid"], neg["docid"],
                query_text, passage_text(pos), passage_text(neg), 0
            ))
            # (neg, pos) → label 1  means doc_b (pos) is more relevant
            pairwise_rows.append((
                qid, neg["docid"], pos["docid"],
                query_text, passage_text(neg), passage_text(pos), 1
            ))

print(f"\nSkipped (no gold positives): {skipped_no_pos}")
print(f"Skipped (no hard negatives): {skipped_no_neg}")
print(f"\nPointwise rows: {len(pointwise_rows):,}")
print(f"Pairwise rows:  {len(pairwise_rows):,}")

# ── shuffle ────────────────────────────────────────────────────────────────────
random.shuffle(pointwise_rows)
random.shuffle(pairwise_rows)

# ── write outputs ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

# pointwise text TSV (monoT5 fine-tuning)
pt_text_path = f"{OUT_DIR}/pointwise_monot5.tsv"
with open(pt_text_path, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["query", "passage", "label"])
    for qid, did, q, p, label in pointwise_rows:
        w.writerow([q, p, label])

# pointwise IDs TSV (for reference / analysis)
pt_ids_path = f"{OUT_DIR}/pointwise_monot5_ids.tsv"
with open(pt_ids_path, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["qid", "doc_id", "label"])
    for qid, did, q, p, label in pointwise_rows:
        w.writerow([qid, did, label])

# pairwise text TSV (duoT5 fine-tuning)
pw_text_path = f"{OUT_DIR}/pairwise_duot5.tsv"
with open(pw_text_path, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["query", "doc_a", "doc_b", "label"])
    for qid, aid, bid, q, ta, tb, label in pairwise_rows:
        w.writerow([q, ta, tb, label])

# pairwise IDs TSV
pw_ids_path = f"{OUT_DIR}/pairwise_duot5_ids.tsv"
with open(pw_ids_path, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["qid", "doc_a_id", "doc_b_id", "label"])
    for qid, aid, bid, q, ta, tb, label in pairwise_rows:
        w.writerow([qid, aid, bid, label])

# ── stats ──────────────────────────────────────────────────────────────────────
pos_count = sum(1 for *_, label in pointwise_rows if label == 1)
neg_count = sum(1 for *_, label in pointwise_rows if label == 0)
pair_0    = sum(1 for *_, label in pairwise_rows if label == 0)
pair_1    = sum(1 for *_, label in pairwise_rows if label == 1)

print(f"\n── Pointwise ({pt_text_path}) ──")
print(f"  Positive examples : {pos_count:>8,}")
print(f"  Negative examples : {neg_count:>8,}")
print(f"  Pos:Neg ratio     : 1:{neg_count/max(pos_count,1):.1f}")

print(f"\n── Pairwise ({pw_text_path}) ──")
print(f"  Label-0 pairs (A wins) : {pair_0:>8,}")
print(f"  Label-1 pairs (B wins) : {pair_1:>8,}")
print(f"  Total pairs            : {len(pairwise_rows):>8,}")

print("\nDone.")
