"""Aggregate metrics across the four experiments into one comparison table."""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "qwen3_0.6b/results"

EXPERIMENTS = [
    ("Exp 1  Pure Qwen3-0.6B",          "metrics_pure_qwen.json",         False),
    ("Exp 2  Qwen + per-type LF",       "metrics_qwen_lf.json",           True),
    ("Exp 3  Qwen + duoT5 top-20",      "metrics_qwen_duot5_top20.json",  False),
    ("Exp 4  Qwen+LF + duoT5 top-20",   "metrics_lf_duot5_top20.json",    False),
]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@10"]
SCOPES = ["global", "summary", "factoid", "list", "yesno"]


def load(path: Path, has_meta: bool) -> dict:
    raw = json.loads(path.read_text())
    return raw["metrics"] if has_meta else raw


def main() -> None:
    print(f"\n{'='*100}")
    print(f"  qwen3_0.6b — comparison across experiments")
    print(f"{'='*100}")

    rows = []
    for label, fname, has_meta in EXPERIMENTS:
        p = RESULTS / fname
        if not p.exists():
            print(f"  [missing] {p}")
            continue
        rows.append((label, load(p, has_meta)))

    for scope in SCOPES:
        print(f"\n--- scope: {scope} ---")
        header = f"{'experiment':<32}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
        print(header)
        print("-" * len(header))
        for label, m in rows:
            vals = "  ".join(f"{m[scope][met]:>8.4f}" for met in METRIC_NAMES)
            print(f"{label:<32}  {vals}")

    out = RESULTS / "summary.json"
    out.write_text(json.dumps({label: m for label, m in rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
