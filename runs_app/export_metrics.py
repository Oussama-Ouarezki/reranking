"""Collect aggregate metrics + elapsed_s from every cached run.

Walks application/cache/runs/<model>/<timestamp>.json, extracts only
aggregate metrics and duration, and writes runs_app/metrics_summary.json.

Usage (from project root):
  python runs_app/export_metrics.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "application/cache/runs"
OUT      = ROOT / "runs_app/metrics_summary.json"


def main() -> None:
    entries = []

    for model_dir in sorted(RUNS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for run_file in sorted(model_dir.glob("*.json")):
            try:
                data = json.loads(run_file.read_text())
            except Exception as e:
                print(f"  WARN {run_file.name}: {e}")
                continue

            entries.append({
                "run_id":     data.get("run_id", run_file.stem),
                "model":      data.get("model", model_dir.name),
                "started_at": data.get("started_at"),
                "elapsed_s":  data.get("elapsed_s"),
                "comment":    data.get("comment"),
                "n_queries":  data.get("config", {}).get("n_questions"),
                "aggregate":  data.get("aggregate"),
            })

    entries.sort(key=lambda r: (r["model"], r.get("started_at") or 0))

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(entries),
        "runs": entries,
    }

    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Saved {len(entries)} runs → {OUT}")

    print(f"\n{'model':<32}  {'elapsed_s':>10}  {'nDCG@10':>8}")
    print("─" * 56)
    for e in entries:
        ndcg = (e.get("aggregate") or {}).get("ndcg_at", {}).get("10")
        print(f"  {e['model']:<30}  {str(round(e['elapsed_s'],1))+'s':>10}  "
              f"{ndcg:.4f}" if ndcg else f"  {e['model']:<30}  {'?':>10}  {'?':>8}")


if __name__ == "__main__":
    main()
