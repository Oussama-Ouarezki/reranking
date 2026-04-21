"""
Fetch PubMed abstracts for all 4 Task13B golden files and save a shared corpus.

Reads:  data/bioasq/raw/Task13BGoldenEnriched/13B{1-4}_golden.json
Writes: data/bioasq/raw/Task13BGoldenEnriched/corpus.jsonl   (title+abstract per PMID)

Supports resume: already-fetched PMIDs are skipped.
"""

import json
import re
import time
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
from pathlib import Path

BASE_DIR    = Path("data/bioasq/raw/Task13BGoldenEnriched")
CORPUS_FILE = BASE_DIR / "corpus.jsonl"
BATCH_SIZE  = 200
BASE_URL    = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
GOLDEN_FILES = [BASE_DIR / f"13B{i}_golden.json" for i in range(1, 5)]


def extract_pmid(url: str) -> str | None:
    m = re.search(r"pubmed/(\d+)", url)
    return m.group(1) if m else None


def collect_all_pmids() -> set:
    pmids = set()
    for path in GOLDEN_FILES:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("questions", []):
            for url in q.get("documents", []):
                pmid = extract_pmid(url)
                if pmid:
                    pmids.add(pmid)
            for snippet in q.get("snippets", []):
                pmid = extract_pmid(snippet.get("document", ""))
                if pmid:
                    pmids.add(pmid)
    return pmids


def load_existing(corpus_file: Path) -> set:
    fetched = set()
    if corpus_file.exists():
        with corpus_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        fetched.add(json.loads(line)["_id"])
                    except Exception:
                        pass
    return fetched


def fetch_batch(pmids: list) -> dict:
    data = {
        "db":      "pubmed",
        "retmode": "xml",
        "id":      ",".join(pmids),
    }
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req     = urllib.request.Request(BASE_URL, data=encoded)

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                root = ET.fromstring(resp.read())

            results = {}
            for article in root.findall(".//PubmedArticle"):
                pmid_node = article.find(".//PMID")
                if pmid_node is None:
                    continue
                pmid = pmid_node.text

                title_node = article.find(".//ArticleTitle")
                title = title_node.text if title_node is not None and title_node.text else ""

                abstract = " ".join(
                    node.text for node in article.findall(".//AbstractText") if node.text
                )
                results[pmid] = (title, abstract)
            return results

        except Exception as e:
            print(f"  Error (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)

    return {}


def save_batch(corpus_file: Path, results: dict) -> None:
    with corpus_file.open("a", encoding="utf-8") as f:
        for pmid, (title, abstract) in results.items():
            text = f"{title} {abstract}".strip()
            if text:
                json.dump({"_id": pmid, "title": title, "text": abstract}, f, ensure_ascii=False)
                f.write("\n")


def main():
    print("Collecting PMIDs from all 4 golden files …")
    all_pmids = collect_all_pmids()
    print(f"  Total unique PMIDs: {len(all_pmids)}")

    already = load_existing(CORPUS_FILE)
    to_fetch = sorted(all_pmids - already)
    print(f"  Already in corpus:  {len(already)}")
    print(f"  To fetch:           {len(to_fetch)}")

    if not to_fetch:
        print("Nothing to fetch — corpus is complete.")
        return

    total_batches = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
    newly_fetched = set()

    for batch_num, i in enumerate(range(0, len(to_fetch), BATCH_SIZE), start=1):
        batch   = to_fetch[i : i + BATCH_SIZE]
        results = fetch_batch(batch)
        save_batch(CORPUS_FILE, results)
        newly_fetched.update(results.keys())

        total_so_far = len(already) + len(newly_fetched)
        print(f"Batch {batch_num}/{total_batches}: fetched {len(results)}/{len(batch)} "
              f"| corpus total: {total_so_far}")

        time.sleep(0.4)   # NCBI: ≤3 req/s without API key

    print(f"\nDone. Corpus saved to {CORPUS_FILE}")
    print(f"  Total documents: {len(already) + len(newly_fetched)}")


if __name__ == "__main__":
    main()
