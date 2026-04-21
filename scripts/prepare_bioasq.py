import json
import os
import re
import time
import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse
from pathlib import Path
from tqdm import tqdm

def extract_pmid(url):
    match = re.search(r'pubmed/(\d+)', url)
    if match:
        return match.group(1)
    return None

def fetch_pubmed_abstracts(pmids, batch_size=200):
    print(f"Fetching {len(pmids)} abstracts from PubMed in batches of {batch_size}...")
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    pmid_list = list(pmids)
    
    results = {} # pmid -> (title, abstract)
    
    for i in tqdm(range(0, len(pmid_list), batch_size)):
        batch = pmid_list[i:i+batch_size]
        data = {
            'db': 'pubmed',
            'retmode': 'xml',
            'id': ','.join(batch)
        }
        
        encoded_data = urllib.parse.urlencode(data).encode('utf-8')
        req = urllib.request.Request(base_url, data=encoded_data)
        
        success = False
        retries = 3
        while not success and retries > 0:
            try:
                with urllib.request.urlopen(req) as response:
                    xml_data = response.read()
                    
                root = ET.fromstring(xml_data)
                for article in root.findall('.//PubmedArticle'):
                    pmid_node = article.find('.//PMID')
                    if pmid_node is not None:
                        pmid = pmid_node.text
                        
                        title = ""
                        title_node = article.find('.//ArticleTitle')
                        if title_node is not None and title_node.text:
                            title = title_node.text
                            
                        abstract = ""
                        abstract_node = article.find('.//AbstractText')
                        if abstract_node is not None and abstract_node.text:
                            # Sometimes abstracts have multiple sections. We join them.
                            abstract_texts = []
                            for abs_text in article.findall('.//AbstractText'):
                                if abs_text.text:
                                    abstract_texts.append(abs_text.text)
                                # some structured abstracts have labels
                            abstract = " ".join(abstract_texts)
                            
                        results[pmid] = (title, abstract)
                        
                success = True
            except Exception as e:
                print(f"Error fetching batch {i}: {e}. Retrying...")
                time.sleep(2)
                retries -= 1
        
        time.sleep(0.4) # Respect rate limits (3 requests per second)
        
    return results

def main():
    base_dir = Path('/home/oussama/Desktop/reranking_project')
    data_dir = base_dir / 'data' / 'bioasq'
    raw_file = data_dir / 'raw' / 'training13b.json'
    
    out_dir = data_dir / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_file = out_dir / 'corpus.jsonl'
    queries_file = out_dir / 'queries.jsonl'
    qrels_train_file = out_dir / 'qrels.train.passage.tsv'
    qrels_test_file = out_dir / 'qrels.test.document.tsv'
    
    if not raw_file.exists():
        print(f"Error: Raw file {raw_file} not found.")
        return

    with open(raw_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    questions = data.get('questions', [])
    print(f"Loaded {len(questions)} questions.")
    
    unique_pmids = set()
    query_data = [] # qid, text, type, train_pmids (snippets), test_pmids (documents)
    
    for q in questions:
        qid = q['id']
        body = q['body']
        
        test_pmids = set()
        for doc_url in q.get('documents', []):
            pmid = extract_pmid(doc_url)
            if pmid:
                test_pmids.add(pmid)
                unique_pmids.add(pmid)
                
        train_pmids = set()
        for snippet in q.get('snippets', []):
            doc_url = snippet.get('document')
            if doc_url:
                pmid = extract_pmid(doc_url)
                if pmid:
                    train_pmids.add(pmid)
                    unique_pmids.add(pmid)
                    
        query_data.append({
            'qid': qid,
            'body': body,
            'train_pmids': train_pmids,
            'test_pmids': test_pmids
        })
        
    print(f"Total unique PMIDs to fetch: {len(unique_pmids)}")
    
    # Check if we already have partial corpus (to restart quickly)
    existing_corpus = {}
    if corpus_file.exists():
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                existing_corpus[item['_id']] = item['text']
                
    pmids_to_fetch = unique_pmids - set(existing_corpus.keys())
    
    if pmids_to_fetch:
        fetched_results = fetch_pubmed_abstracts(pmids_to_fetch)
        # Update existing_corpus
        for pmid, (title, abstract) in fetched_results.items():
            text = f"{title} {abstract}".strip()
            existing_corpus[pmid] = text
            
        print(f"Total PMIDs successfully fetched from PubMed: {len(fetched_results)}")
    
    # Save corpus
    print("Saving corpus.jsonl...")
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for pmid in unique_pmids:
            if pmid in existing_corpus and existing_corpus[pmid].strip():
                # BEIR structure
                json.dump({
                    '_id': pmid,
                    'title': '',  # We concatenate title and abstract in text
                    'text': existing_corpus[pmid]
                }, f, ensure_ascii=False)
                f.write('\n')
                
    # Save queries
    print("Saving queries.jsonl...")
    with open(queries_file, 'w', encoding='utf-8') as f:
        for q in query_data:
            json.dump({
                '_id': q['qid'],
                'text': q['body']
            }, f, ensure_ascii=False)
            f.write('\n')
            
    # Save qrels train
    print("Saving qrels.train.passage.tsv...")
    with open(qrels_train_file, 'w', encoding='utf-8') as f:
        f.write('query-id\tcorpus-id\tscore\n') # BEIR format
        for q in query_data:
            for pmid in q['train_pmids']:
                if pmid in existing_corpus and existing_corpus[pmid].strip():
                    f.write(f"{q['qid']}\t{pmid}\t1\n")
                    
    # Save qrels test
    print("Saving qrels.test.document.tsv...")
    with open(qrels_test_file, 'w', encoding='utf-8') as f:
        f.write('query-id\tcorpus-id\tscore\n') # BEIR format
        for q in query_data:
            for pmid in q['test_pmids']:
                if pmid in existing_corpus and existing_corpus[pmid].strip():
                    f.write(f"{q['qid']}\t{pmid}\t1\n")

    print("Data preparation complete.")

if __name__ == '__main__':
    main()
