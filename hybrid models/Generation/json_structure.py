"""
json_structure.py — Inspect the structure of a JSON or JSONL file.

Usage:
    python json_structure.py <path_to_file.json>
    python json_structure.py <path_to_file.jsonl>
"""

import json
import sys
from collections import Counter
from pathlib import Path


def get_type_label(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def schema_fingerprint(value):
    """Return a hashable fingerprint of the top-level key set of a dict."""
    if isinstance(value, dict):
        return frozenset(value.keys())
    return get_type_label(value)


def is_id_like(key: str) -> bool:
    """True if the key looks like a generated ID rather than a meaningful field name."""
    return len(key) >= 16 and all(c in "0123456789abcdefABCDEF" for c in key)


def should_collapse(d: dict) -> bool:
    """
    Return True if this dict looks like a map of ID→record (all keys look like
    IDs, or all values share the same schema fingerprint).
    """
    if not d:
        return False
    keys = list(d.keys())
    # All keys look like hex IDs
    if all(is_id_like(k) for k in keys):
        return True
    # All values are dicts with the same top-level key set
    if all(isinstance(v, dict) for v in d.values()):
        fingerprints = {schema_fingerprint(v) for v in d.values()}
        if len(fingerprints) == 1 and len(keys) > 1:
            return True
    return False


def describe_schema(value, indent=0, max_depth=8):
    prefix = "  " * indent

    if indent >= max_depth:
        print(f"{prefix}...")
        return

    if isinstance(value, dict):
        if not value:
            print(f"{prefix}{{}}")
            return

        if should_collapse(value):
            # Pick the most "complete" record (most non-null leaf values)
            def completeness(v):
                return sum(1 for x in v.values() if x is not None)

            representative = max(value.values(), key=completeness)
            n = len(value)
            label = "hex ID" if all(is_id_like(k) for k in value.keys()) else "key"
            print(f"{prefix}{{<{label}>}}  ×{n} (same schema, showing one)")
            describe_schema(representative, indent + 1, max_depth)
        else:
            for key, val in value.items():
                if isinstance(val, dict):
                    print(f"{prefix}{key}: object")
                    describe_schema(val, indent + 1, max_depth)
                elif isinstance(val, list):
                    item_types = Counter(get_type_label(i) for i in val)
                    types_str = ", ".join(item_types)
                    print(f"{prefix}{key}: array[{types_str}] ({len(val)} items)")
                    if val and isinstance(val[0], (dict, list)):
                        describe_schema(val[0], indent + 1, max_depth)
                else:
                    print(f"{prefix}{key}: {get_type_label(val)}")

    elif isinstance(value, list):
        if not value:
            print(f"{prefix}(empty array)")
            return
        item_types = Counter(get_type_label(i) for i in value)
        types_str = ", ".join(item_types)
        print(f"{prefix}array[{types_str}] ({len(value)} items)")
        if value and isinstance(value[0], (dict, list)):
            describe_schema(value[0], indent + 1, max_depth)

    else:
        print(f"{prefix}{get_type_label(value)}")


def inspect_json(path: Path):
    print(f"\n{path.name}  ({path.stat().st_size:,} bytes)\n")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"root: {get_type_label(data)}")
    describe_schema(data)
    print()


def inspect_jsonl(path: Path):
    print(f"\n{path.name}  ({path.stat().st_size:,} bytes)  [JSONL]\n")
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    print(f"{len(lines)} records — showing schema of record 0:\n")
    if lines:
        describe_schema(lines[0])
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python json_structure.py <file.json|file.jsonl>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Error: file not found — {path}")
        sys.exit(1)

    try:
        if path.suffix.lower() == ".jsonl":
            inspect_jsonl(path)
        else:
            inspect_json(path)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()