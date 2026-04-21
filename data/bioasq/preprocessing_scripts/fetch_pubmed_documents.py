import json
import re
import time
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
from pathlib import Path


def extract_pmid(url):
    match = re.search(r'pubmed/(\d+)', url)
    return match.group(1) if match else None


def fetch_batch(pmids, base_url):
    """POST a single batch to PubMed efetch. Returns dict {pmid: (title, abstract)}."""
    data = {
        'db': 'pubmed',
        'retmode': 'xml',
        'id': ','.join(pmids),
    }
    encoded = urllib.parse.urlencode(data).encode('utf-8')
    req = urllib.request.Request(base_url, data=encoded)

    retries = 3
    while retries > 0:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            results = {}
            for article in root.findall('.//PubmedArticle'):
                pmid_node = article.find('.//PMID')
                if pmid_node is None:
                    continue
                pmid = pmid_node.text

                title = ''
                title_node = article.find('.//ArticleTitle')
                if title_node is not None and title_node.text:
                    title = title_node.text

                abstract_parts = [
                    node.text
                    for node in article.findall('.//AbstractText')
                    if node.text
                ]
                abstract = ' '.join(abstract_parts)

                results[pmid] = (title, abstract)
            return results

        except Exception as e:
            retries -= 1
            print(f"  Error: {e}. Retries left: {retries}")
            time.sleep(2)

    return {}


def collect_pmids(questions):
    """
    Returns a dict mapping each question id to the set of PMIDs
    required by that question (documents + snippet documents).
    Also returns the flat set of all unique PMIDs.
    """
    query_pmids = {}
    all_pmids = set()

    for q in questions:
        qid = q['id']
        pmids = set()

        for url in q.get('documents', []):
            pmid = extract_pmid(url)
            if pmid:
                pmids.add(pmid)

        for snippet in q.get('snippets', []):
            pmid = extract_pmid(snippet.get('document', ''))
            if pmid:
                pmids.add(pmid)

        query_pmids[qid] = pmids
        all_pmids.update(pmids)

    return query_pmids, all_pmids


def load_existing_corpus(corpus_file):
    """Return set of PMIDs already saved in corpus.jsonl."""
    fetched = set()
    if corpus_file.exists():
        with open(corpus_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    fetched.add(json.loads(line)['_id'])
                except Exception:
                    pass
    return fetched


def save_batch_to_corpus(corpus_file, batch_results):
    """Append a dict of {pmid: (title, abstract)} to corpus.jsonl."""
    with open(corpus_file, 'a', encoding='utf-8') as f:
        for pmid, (title, abstract) in batch_results.items():
            text = f"{title} {abstract}".strip()
            if text:
                json.dump({'_id': pmid, 'title': '', 'text': text},
                          f, ensure_ascii=False)
                f.write('\n')


def filter_and_save_queries(questions, query_pmids, fetched_pmids,
                             queries_file, qrels_file):
    """
    Keep only questions whose every required PMID was successfully fetched.
    Writes queries.jsonl and qrels.tsv for the kept questions.
    """
    kept = 0
    dropped = 0

    with open(queries_file, 'w', encoding='utf-8') as qf, \
         open(qrels_file, 'w', encoding='utf-8') as rf:

        rf.write('query-id\tcorpus-id\tscore\n')

        for q in questions:
            qid = q['id']
            required = query_pmids[qid]

            if not required:
                dropped += 1
                continue

            if not required.issubset(fetched_pmids):
                missing = required - fetched_pmids
                print(f"  Dropping query {qid}: {len(missing)} PMIDs not fetched")
                dropped += 1
                continue

            json.dump({'_id': qid, 'text': q['body']}, qf, ensure_ascii=False)
            qf.write('\n')

            for pmid in required:
                rf.write(f"{qid}\t{pmid}\t1\n")

            kept += 1

    print(f"\nQueries kept: {kept}  |  dropped: {dropped}")


def main():
    base_dir = Path('/home/oussama/Desktop/reranking_project')
    raw_file  = base_dir / 'data' / 'bioasq' / 'raw' / 'training13b.json'
    out_dir   = base_dir / 'data' / 'bioasq' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_file  = out_dir / 'corpus.jsonl'
    queries_file = out_dir / 'queries.jsonl'
    qrels_file   = out_dir / 'qrels.tsv'

    BATCH_SIZE = 200
    BASE_URL   = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'

    if not raw_file.exists():
        print(f"Error: {raw_file} not found.")
        return

    print("Reading BioASQ training file...")
    with open(raw_file, encoding='utf-8') as f:
        data = json.load(f)

    questions = data.get('questions', [])
    print(f"Total questions: {len(questions)}")

    query_pmids, all_pmids = collect_pmids(questions)
    print(f"Unique PMIDs referenced: {len(all_pmids)}")

    already_fetched = load_existing_corpus(corpus_file)
    to_fetch = sorted(all_pmids - already_fetched)
    print(f"Already in corpus: {len(already_fetched)}  |  To fetch: {len(to_fetch)}")

    # ── Fetch in batches, save each batch immediately ────────────────────────
    total_batches = (len(to_fetch) + BATCH_SIZE - 1) // BATCH_SIZE
    newly_fetched = set()

    for batch_num, i in enumerate(range(0, len(to_fetch), BATCH_SIZE), start=1):
        batch = to_fetch[i : i + BATCH_SIZE]
        print(f"Batch {batch_num}/{total_batches}  ({len(batch)} PMIDs)...")

        results = fetch_batch(batch, BASE_URL)
        save_batch_to_corpus(corpus_file, results)
        newly_fetched.update(results.keys())

        fetched_so_far = len(already_fetched) + len(newly_fetched)
        print(f"  Saved {len(results)}/{len(batch)} this batch  "
              f"| corpus total: {fetched_so_far}")

        time.sleep(0.4)  # NCBI rate limit: ≤3 req/s

    all_fetched = already_fetched | newly_fetched
    print(f"\nFetch complete. Total PMIDs in corpus: {len(all_fetched)}")

    # ── Drop queries with missing documents, write queries + qrels ───────────
    print("\nFiltering queries and writing queries.jsonl + qrels.tsv ...")
    filter_and_save_queries(questions, query_pmids, all_fetched,
                            queries_file, qrels_file)

    print(f"\nOutputs written to {out_dir}/")
    print(f"  corpus.jsonl  — {len(all_fetched)} documents")
    print(f"  queries.jsonl — kept queries")
    print(f"  qrels.tsv     — relevance judgements")


if __name__ == '__main__':
    main()
