"""
Merge three BioASQ corpus files into two deduplicated output files.

File 1 — corpus_full.jsonl  (priority: Task13BGoldenEnriched → processed → pubmed_full)
File 2 — corpus_full_processed.jsonl  (priority: processed → Task13BGoldenEnriched → pubmed_full)

Output directory: data/bioasq/pubmed_full/full/
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # reranking_project/

TASK13 = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/corpus.jsonl"
PROCESSED = ROOT / "data/bioasq/processed/corpus.jsonl"
PUBMED = ROOT / "data/bioasq/pubmed_full/corpus.jsonl"

OUTPUT_DIR = ROOT / "data/bioasq/pubmed_full/full"

MERGES = [
    {
        "output": OUTPUT_DIR / "corpus_full.jsonl",
        "sources": [TASK13, PROCESSED, PUBMED],
        "label": "Task13BGoldenEnriched → processed → pubmed_full",
    },
    {
        "output": OUTPUT_DIR / "corpus_full_processed.jsonl",
        "sources": [PROCESSED, TASK13, PUBMED],
        "label": "processed → Task13BGoldenEnriched → pubmed_full",
    },
]


def merge(sources: list[Path], output: Path) -> tuple[int, int]:
    seen_ids: set[str] = set()
    total_written = 0
    total_skipped = 0

    with output.open("w", encoding="utf-8") as out:
        for src in sources:
            if not src.exists():
                print(f"  [WARN] File not found, skipping: {src}")
                continue

            src_written = 0
            src_skipped = 0

            with src.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc: dict = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"  [WARN] {src.name}:{line_no} — JSON parse error: {e}")
                        continue

                    doc_id = str(doc.get("_id", ""))
                    if not doc_id:
                        print(f"  [WARN] {src.name}:{line_no} — missing _id, skipping")
                        continue

                    if doc_id in seen_ids:
                        src_skipped += 1
                        continue

                    seen_ids.add(doc_id)
                    _ = out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    src_written += 1

            total_written += src_written
            total_skipped += src_skipped
            print(f"  {src.name}: {src_written} written, {src_skipped} duplicates skipped")

    return total_written, total_skipped


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in MERGES:
        print(f"\n[{cfg['label']}]")
        written, skipped = merge(cfg["sources"], cfg["output"])
        print(f"  -> {written} unique docs | {skipped} duplicates dropped")
        print(f"  -> {cfg['output']}")


if __name__ == "__main__":
    main()
