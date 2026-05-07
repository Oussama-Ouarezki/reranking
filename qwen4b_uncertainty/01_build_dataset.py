"""Build a 500-query evaluation set: 125 newest per type, with ideal/exact answers.

Reads:
  data/bioasq/processed/queries.jsonl   (has _id, text, type)
  data/bioasq/processed/qrels.tsv       (qrels filter: must have >=1 gold)
  data/bioasq/raw/training13b.json      (source of ideal_answer / exact_answer)

Writes:
  qwen4b_uncertainty/data/queries_500.jsonl
"""

import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
QUERIES = BASE / "data/bioasq/processed/queries.jsonl"
QRELS = BASE / "data/bioasq/processed/qrels.tsv"
RAW = BASE / "data/bioasq/raw/training13b.json"
OUT = BASE / "qwen4b_uncertainty/data/queries_2000.jsonl"

PER_TYPE = 500
TYPES = ["summary", "factoid", "list", "yesno"]


def oid_ts(oid: str) -> int:
    return int(oid[:8], 16)


def load_qrel_qids() -> set[str]:
    qids: set[str] = set()
    with QRELS.open() as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and int(parts[2]) > 0:
                qids.add(parts[0])
    return qids


def load_raw_answers() -> dict[str, dict]:
    raw = json.loads(RAW.read_text())
    out: dict[str, dict] = {}
    for q in raw["questions"]:
        out[q["id"]] = {
            "ideal_answer": q.get("ideal_answer"),
            "exact_answer": q.get("exact_answer"),
        }
    return out


def main() -> None:
    qrel_qids = load_qrel_qids()
    answers = load_raw_answers()
    print(f"qrel qids: {len(qrel_qids):,}  raw questions: {len(answers):,}")

    by_type: dict[str, list[dict]] = defaultdict(list)
    with QUERIES.open() as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] not in qrel_qids:
                continue
            t = q.get("type")
            if t in TYPES:
                by_type[t].append(q)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    counts: dict[str, int] = {}
    with OUT.open("w") as out:
        for t in TYPES:
            qs = sorted(by_type[t], key=lambda q: oid_ts(q["_id"]), reverse=True)
            picked = qs[:PER_TYPE]
            counts[t] = len(picked)
            for q in picked:
                ans = answers.get(q["_id"], {})
                rec = {
                    "_id": q["_id"],
                    "text": q["text"],
                    "type": t,
                    "ideal_answer": ans.get("ideal_answer"),
                    "exact_answer": ans.get("exact_answer"),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"Wrote {written} queries to {OUT}")
    print(f"Per-type counts: {counts}")
    for t, c in counts.items():
        assert c == PER_TYPE, f"type {t}: only {c} (need {PER_TYPE})"


if __name__ == "__main__":
    main()
