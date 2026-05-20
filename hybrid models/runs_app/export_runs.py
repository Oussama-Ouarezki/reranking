"""Export all cached evaluation runs into a single JSON file for exploration.

Walks application/cache/runs/<model>/<timestamp>.json and concatenates every
run into one flat list, preserving all fields (aggregate metrics, per-query
metrics, config, timings, etc.).

Output structure:
{
  "exported_at": "<ISO timestamp>",
  "n_runs": <int>,
  "runs": [
    {
      "run_id": "...",
      "model": "monot5",
      "started_at": <unix float>,
      "ended_at":   <unix float>,
      "elapsed_s":  276.1,
      "config":     { ... },
      "comment":    "...",
      "aggregate":  { "ndcg_at": {...}, "mrr_at": {...}, ... },
      "per_query":  { "<qid>": { "metrics": {...}, "top_docids": [...] }, ... }
    },
    ...
  ]
}

Usage (from project root):
  python runs_app/export_runs.py
  python runs_app/export_runs.py --out runs_app/all_runs.json
  python runs_app/export_runs.py --latest-only   # one run per model (newest)
  python runs_app/export_runs.py --no-per-query  # skip per-query for smaller file
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "application/cache/runs"
DEFAULT_OUT = ROOT / "runs_app/all_runs.json"


def load_run(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Ensure run_id is present (older files may not have it)
    if "run_id" not in data:
        data["run_id"] = path.stem
    # Ensure model is present (fill from parent dir name if missing)
    if "model" not in data:
        data["model"] = path.parent.name
    return data


def collect_runs(runs_dir: Path, latest_only: bool) -> list[dict]:
    runs = []
    for model_dir in sorted(runs_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        json_files = sorted(model_dir.glob("*.json"), key=lambda p: p.stem)
        if not json_files:
            continue
        if latest_only:
            json_files = [json_files[-1]]
        for jf in json_files:
            try:
                runs.append(load_run(jf))
            except Exception as e:
                print(f"  WARN: skipping {jf.relative_to(runs_dir.parent)}: {e}")
    return runs


def strip_per_query(runs: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "per_query"} for r in runs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir",    default=str(RUNS_DIR),
                        help="Root directory containing per-model run folders")
    parser.add_argument("--out",         default=str(DEFAULT_OUT),
                        help="Output JSON file path")
    parser.add_argument("--latest-only", action="store_true",
                        help="Keep only the most recent run per model")
    parser.add_argument("--no-per-query", action="store_true",
                        help="Omit per_query data (much smaller file)")
    parser.add_argument("--indent",      type=int, default=2,
                        help="JSON indentation (0 = compact)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_path = Path(args.out)

    print(f"Scanning: {runs_dir}")
    runs = collect_runs(runs_dir, latest_only=args.latest_only)

    if args.no_per_query:
        runs = strip_per_query(runs)

    # Sort by model name then start time for reproducible ordering
    runs.sort(key=lambda r: (r.get("model", ""), r.get("started_at", 0)))

    exported_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "exported_at": exported_at,
        "n_runs":      len(runs),
        "runs":        runs,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = args.indent if args.indent > 0 else None
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=indent, ensure_ascii=False)

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\nExported {len(runs)} runs → {out_path}  ({size_mb:.2f} MB)")

    # Summary table
    print(f"\n{'model':<30}  {'runs':>4}  {'elapsed_s':>10}  {'nDCG@10':>8}")
    print("─" * 60)
    seen: dict[str, list] = {}
    for r in runs:
        seen.setdefault(r.get("model", "?"), []).append(r)
    for model, model_runs in sorted(seen.items()):
        for r in model_runs:
            ndcg10 = r.get("aggregate", {}).get("ndcg_at", {}).get("10", None)
            ndcg_str = f"{ndcg10:.4f}" if ndcg10 is not None else "     -"
            elapsed  = r.get("elapsed_s", None)
            elapsed_str = f"{elapsed:.1f}s" if elapsed is not None else "-"
            print(f"  {model:<28}  {1:>4}  {elapsed_str:>10}  {ndcg_str:>8}")


if __name__ == "__main__":
    main()
