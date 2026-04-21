"""
Sample 20 records from dataset1_rerank.jsonl and 20 from dataset2_rerank.jsonl,
concatenate into a shared validation set, and remove them from the originals.

Output:
  dataset1_rerank_train.jsonl   (180 records)
  dataset2_rerank_train.jsonl   (180 records)
  validation.jsonl              ( 40 records: 20 from each)

Originals are NOT modified.
"""

import json
import random
from pathlib import Path

HERE      = Path(__file__).parent
SEED      = 42
N_VAL     = 20

rng = random.Random(SEED)

val_records = []

for fname, train_out in [
    ("dataset1_rerank.jsonl", "dataset1_rerank_train.jsonl"),
    ("dataset2_rerank.jsonl", "dataset2_rerank_train.jsonl"),
]:
    records = [json.loads(l) for l in (HERE / fname).open() if l.strip()]
    rng.shuffle(records)

    val   = records[:N_VAL]
    train = records[N_VAL:]

    val_records.extend(val)

    with (HERE / train_out).open("w") as f:
        for rec in train:
            f.write(json.dumps(rec) + "\n")
    print(f"{train_out}: {len(train)} records")

# save combined validation set
val_path = HERE / "validation.jsonl"
with val_path.open("w") as f:
    for rec in val_records:
        f.write(json.dumps(rec) + "\n")
print(f"validation.jsonl: {len(val_records)} records (20 from each dataset)")
