"""
Download and parse the full PubMed annual baseline from NCBI FTP.

Saves the 200,000 most-recent documents (highest-numbered baseline files
processed first) and stops.  Re-running always yields the same 200k docs:
  - files are processed in a fixed reverse-sorted order
  - the corpus count is re-derived from corpus.jsonl on every resume
  - writing stops at exactly DOC_LIMIT documents

Disk (streaming, no XML kept): ~600 MB – 1 GB for corpus.jsonl
Estimated time: 30-90 minutes depending on connection speed

Supports resume: re-run at any time to continue from where it stopped.

Usage:
    python data/bioasq/preprocessing_scripts/fetch_pubmed_baseline.py
    python data/bioasq/preprocessing_scripts/fetch_pubmed_baseline.py --workers 6
    python data/bioasq/preprocessing_scripts/fetch_pubmed_baseline.py --out-dir /mnt/external/pubmed
    python data/bioasq/preprocessing_scripts/fetch_pubmed_baseline.py --year 24
"""

import argparse
import collections
import gzip
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


BASELINE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
TIMEOUT      = 180   # seconds per request
DOC_LIMIT    = 200_000  # keep only the most-recent N documents


# ── directory listing ─────────────────────────────────────────────────────────

def list_baseline_files(year: int) -> list[str]:
    """Return sorted list of pubmedYYnNNNN.xml.gz filenames from NCBI FTP."""
    print(f"Fetching directory listing from {BASELINE_URL} …")
    req = urllib.request.Request(BASELINE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    pattern = rf"pubmed{year:02d}n\d+\.xml\.gz"
    files   = sorted(set(re.findall(pattern, html)))
    if not files:
        sys.exit(
            f"No files found for year=20{year:02d}. "
            f"Try --year 24 or --year 25. Raw listing snippet:\n{html[:500]}"
        )
    print(f"  Found {len(files):,} files for pubmed{year:02d} baseline.")
    return files


# ── download ──────────────────────────────────────────────────────────────────

def download_file(filename: str, retries: int = 5) -> io.BytesIO:  # noqa: return
    url = BASELINE_URL + filename
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return io.BytesIO(resp.read())
        except (urllib.error.URLError, OSError) as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  [{filename}] attempt {attempt} failed ({exc}), retrying in {wait}s …",
                  flush=True)
            time.sleep(wait)


# ── XML parsing ───────────────────────────────────────────────────────────────

def _iter_text(elem) -> str:
    return "".join(elem.itertext()).strip()


def parse_xml_gz(buf: io.BytesIO):
    """Yield (pmid, title, abstract) from a gzipped PubMed XML buffer."""
    buf.seek(0)
    with gzip.open(buf) as gz:
        for event, elem in ET.iterparse(gz, events=("end",)):
            if elem.tag != "PubmedArticle":
                continue

            pmid_node = elem.find(".//PMID")
            if pmid_node is None or not pmid_node.text:
                elem.clear()
                continue
            pmid = pmid_node.text.strip()

            title = ""
            t_node = elem.find(".//ArticleTitle")
            if t_node is not None:
                title = _iter_text(t_node)

            parts = [_iter_text(ab) for ab in elem.findall(".//AbstractText")]
            abstract = " ".join(p for p in parts if p)

            elem.clear()

            if abstract or title:
                yield pmid, title, abstract


# ── checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(path.read_text().splitlines())


def append_checkpoint(path: Path, filename: str) -> None:
    with path.open("a") as f:
        f.write(filename + "\n")


# ── main pipeline ─────────────────────────────────────────────────────────────


def count_corpus_lines(path: Path) -> int:
    """Count existing documents in corpus.jsonl (fast line count)."""
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def process_buf_limited(buf: io.BytesIO, out_f, budget: int) -> tuple[int, bool]:
    """Parse buf and write up to *budget* JSONL lines.

    Returns (n_written, limit_reached).  Parsing is deterministic so the same
    docs are always selected when budget < total docs in the file.
    """
    n = 0
    for pmid, title, abstract in parse_xml_gz(buf):
        if n >= budget:
            return n, True
        out_f.write(
            json.dumps({"_id": pmid, "title": title, "text": abstract},
                       ensure_ascii=False) + "\n"
        )
        n += 1
    return n, False


def run(files: list[str], out_dir: Path, workers: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path     = out_dir / "corpus.jsonl"
    checkpoint_path = out_dir / "checkpoint.txt"

    # Process newest files first so we collect the most-recent documents.
    # Baseline filenames sort lexicographically by sequence number, so
    # reversing gives highest (most recent) first — deterministic across runs.
    ordered = list(reversed(files))

    done      = load_checkpoint(checkpoint_path)
    remaining = [f for f in ordered if f not in done]

    # Re-derive how many docs already written so resume is always consistent.
    already_written = count_corpus_lines(corpus_path)

    # Trim corpus to DOC_LIMIT if it already exceeds the target.
    if already_written > DOC_LIMIT:
        print(f"\nCorpus has {already_written:,} documents — trimming to {DOC_LIMIT:,} …")
        tmp_path = corpus_path.with_suffix('.tmp')
        with corpus_path.open('r', encoding='utf-8') as src, \
             tmp_path.open('w', encoding='utf-8') as dst:
            for i, line in enumerate(src):
                if i >= DOC_LIMIT:
                    break
                dst.write(line)
        tmp_path.replace(corpus_path)
        already_written = DOC_LIMIT
        print(f"  Trimmed. Corpus now has {already_written:,} documents.")

    budget = DOC_LIMIT - already_written

    print(f"\nTarget   : {DOC_LIMIT:,} most-recent documents")
    print(f"Written  : {already_written:,}  (budget remaining: {budget:,})")
    print(f"Progress : {len(done):,} files done / {len(ordered):,} total  "
          f"({len(remaining):,} remaining)")

    if budget <= 0:
        print(f"Corpus already has {DOC_LIMIT:,} documents. Nothing to do.")
        return
    if not remaining:
        print("All files processed but budget not filled — corpus is complete.")
        return

    print(f"Output   : {corpus_path}")
    print(f"Workers  : {workers} parallel downloads\n")

    t0         = time.time()
    new_docs   = 0
    PREFETCH   = workers

    with corpus_path.open("a", encoding="utf-8", buffering=1 << 20) as out_f, \
         ThreadPoolExecutor(max_workers=PREFETCH) as pool:

        pending: collections.deque[tuple] = collections.deque()

        def submit_next(fname):
            pending.append((fname, pool.submit(download_file, fname)))

        initial      = remaining[:PREFETCH]
        leftover     = remaining[PREFETCH:]
        leftover_idx = 0
        for fname in initial:
            submit_next(fname)

        processed    = 0
        limit_hit    = False

        for _ in range(len(remaining)):
            if not pending or limit_hit:
                break

            fname, fut = pending.popleft()

            if leftover_idx < len(leftover):
                submit_next(leftover[leftover_idx])
                leftover_idx += 1

            try:
                buf              = fut.result()
                file_budget      = budget - new_docs
                ndoc, limit_hit  = process_buf_limited(buf, out_f, file_budget)
                append_checkpoint(checkpoint_path, fname)
                done.add(fname)
                new_docs  += ndoc
                processed += 1

                total_so_far = already_written + new_docs
                elapsed      = time.time() - t0
                rate         = processed / elapsed
                eta          = (len(remaining) - processed) / rate if rate > 0 else 0
                print(
                    f"  [{processed:>5}/{len(remaining)}]  {fname}  "
                    f"+{ndoc:,} docs  total={total_so_far:,}/{DOC_LIMIT:,}  "
                    f"ETA={eta/60:.0f}min",
                    flush=True,
                )

                if limit_hit:
                    print(f"\n  Reached {DOC_LIMIT:,}-document limit — stopping.")
                    break

            except Exception as exc:
                print(f"  ERROR processing {fname}: {exc}  (skipping)", flush=True)

    elapsed      = time.time() - t0
    final_total  = already_written + new_docs
    print(f"\nDone. {final_total:,} documents in corpus ({new_docs:,} added this run) "
          f"in {elapsed/60:.1f} min → {corpus_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download and parse the full PubMed annual baseline."
    )
    parser.add_argument(
        "--year", type=int, default=26,
        help="Two-digit baseline year, e.g. 26 for pubmed26 (default: 26)"
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("data/bioasq/pubmed_full"),
        help="Output directory for corpus.jsonl and checkpoint (default: data/bioasq/pubmed_full)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download threads (default: 4)"
    )
    args = parser.parse_args()

    files = list_baseline_files(args.year)
    run(files, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
