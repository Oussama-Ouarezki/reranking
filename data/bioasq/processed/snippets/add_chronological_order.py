"""
Add chronological_order (1 = oldest) and created_at (ISO 8601) fields to
data/bioasq/processed/queries.jsonl by decoding the timestamp embedded in
each MongoDB ObjectID (_id field, first 8 hex chars = Unix timestamp).
"""

import json
import struct
from datetime import datetime, timezone
from pathlib import Path

INPUT = Path("data/bioasq/processed/queries.jsonl")
OUTPUT = Path("data/bioasq/processed/queries.jsonl")


def objectid_timestamp(oid: str) -> int:
    """Extract Unix timestamp from the first 4 bytes of a MongoDB ObjectID."""
    return struct.unpack(">I", bytes.fromhex(oid[:8]))[0]


def main():
    records = []
    with INPUT.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Sort by embedded timestamp, preserving stable order for ties
    records.sort(key=lambda r: objectid_timestamp(r["_id"]))

    with OUTPUT.open("w") as f:
        for rank, rec in enumerate(records, start=1):
            ts = objectid_timestamp(rec["_id"])
            rec["chronological_order"] = rank
            rec["created_at"] = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            f.write(json.dumps(rec) + "\n")

    print(f"Done — {len(records)} queries written to {OUTPUT}")
    print(f"  Oldest: rank 1  created_at={datetime.fromtimestamp(objectid_timestamp(records[0]['_id']), tz=timezone.utc).date()}")
    print(f"  Newest: rank {len(records)}  created_at={datetime.fromtimestamp(objectid_timestamp(records[-1]['_id']), tz=timezone.utc).date()}")


if __name__ == "__main__":
    main()
