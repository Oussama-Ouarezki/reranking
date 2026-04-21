"""
Convert dataset1_rerank.jsonl and dataset2_rerank.jsonl into the LiT5-Distill
fine-tuning format expected by castorini/LiT5's FiD/src/data.py:

  {
    "id":       "<query_id>",
    "question": "<query text>",
    "ctxs": [
      {"docid": "<doc_id>", "text": "<passage text>"},
      ...  # in the shuffled order presented to the LLM
    ],
    "target": "3 1 7 2 ..."   # 1-indexed positions in ctxs, best → worst
  }

The target is derived from ranked_output: for each rank (1, 2, ...) find the
position of that doc in the shuffled ctxs list, output that 1-indexed position.

Passages are truncated so query + passage <= 512 whitespace tokens.

Input:  data/bioasq/training_strat/dataset{1,2}_rerank.jsonl
Output: data/bioasq/training_strat/lit5_dataset{1,2}.jsonl
"""

import json
from pathlib import Path

CORPUS_PATH = Path("data/bioasq/processed/corpus.jsonl")
IN_DIR      = Path("data/bioasq/training_strat")
OUT_DIR     = Path("data/bioasq/training_strat")
MAX_TOKENS  = 512

DATASETS = [
    ("dataset1_rerank.jsonl",       "lit5_dataset1.jsonl"),
    ("dataset2_rerank.jsonl",       "lit5_dataset2.jsonl"),
    ("dataset1_rerank_train.jsonl", "lit5_dataset1_train.jsonl"),
    ("dataset2_rerank_train.jsonl", "lit5_dataset2_train.jsonl"),
    ("validation.jsonl",            "lit5_validation.jsonl"),
]


def truncate_passage(query: str, passage: str, max_tokens: int = MAX_TOKENS) -> str:
    q_tokens = query.split()
    p_tokens = passage.split()
    budget   = max_tokens - len(q_tokens)
    return " ".join(p_tokens[:max(budget, 0)])


# ── Load corpus ───────────────────────────────────────────────────────────────
print("Loading corpus …")
corpus = {}
with CORPUS_PATH.open() as f:
    for line in f:
        doc = json.loads(line)
        corpus[doc["_id"]] = (doc.get("title", "") + " " + doc["text"]).strip()
print(f"  {len(corpus):,} documents loaded.")


# ── Convert each dataset ──────────────────────────────────────────────────────
for in_name, out_name in DATASETS:
    in_path  = IN_DIR  / in_name
    out_path = OUT_DIR / out_name

    if not in_path.exists():
        print(f"Skipping {in_name} (not found)")
        continue

    records   = []
    skipped   = 0

    with in_path.open() as f:
        raw = [json.loads(line) for line in f if line.strip()]

    for rec in raw:
        qid      = rec["query_id"]
        query    = rec["question"] if "question" in rec else rec["query"]
        shuffled = rec["shuffled_input"]   # list of {doc_id, bm25_score}
        ranked   = rec["ranked_output"]    # list of {rank, doc_id}

        # Build doc_id → shuffled position (1-indexed)
        pos_map = {entry["doc_id"]: idx + 1 for idx, entry in enumerate(shuffled)}

        # Build target: positions in shuffled list, ordered best → worst
        target_positions = []
        for item in sorted(ranked, key=lambda x: x["rank"]):
            doc_id = item["doc_id"]
            if doc_id in pos_map:
                target_positions.append(str(pos_map[doc_id]))

        if not target_positions:
            skipped += 1
            continue

        target = " ".join(target_positions)

        # Build ctxs in shuffled order with passage text
        ctxs = []
        for entry in shuffled:
            doc_id = entry["doc_id"]
            text   = corpus.get(doc_id, "")
            ctxs.append({
                "docid": doc_id,
                "text":  truncate_passage(query, text),
            })

        records.append({
            "id":       qid,
            "question": query,
            "ctxs":     ctxs,
            "target":   target,
        })

    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"{in_name} → {out_name}: {len(records)} records written, {skipped} skipped")

    # Quick sanity check on first record
    sample = records[0]
    print(f"  Sample id:      {sample['id']}")
    print(f"  Question:       {sample['question'][:80]}")
    print(f"  Num ctxs:       {len(sample['ctxs'])}")
    print(f"  Target:         {sample['target']}")
    print(f"  First ctx text: {sample['ctxs'][0]['text'][:80]} …")
    print()
