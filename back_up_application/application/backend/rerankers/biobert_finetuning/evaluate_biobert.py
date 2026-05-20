"""Evaluate a BioBERT cross-encoder reranker on BioASQ.

Reranks BM25 top-K (default 50) with the given checkpoint and reports
nDCG@10 and MRR@10 via ir-measures. Use --base_model to compare against
the un-finetuned baseline; use --model with the directory produced by
finetune_biobert.py to score the fine-tuned variant.

Usage
-----
# baseline
python application/backend/rerankers/biobert_finetuning/evaluate_biobert.py \
    --model NeuML/biomedbert-base-reranker --tag biobert-base

# fine-tuned
python application/backend/rerankers/biobert_finetuning/evaluate_biobert.py \
    --model checkpoints/biobert-bioasq-deepseek --tag biobert-deepseek
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import ir_measures
from ir_measures import nDCG, RR

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
DEFAULT_QUERIES = os.path.join(ROOT, "data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl")
DEFAULT_QRELS = os.path.join(ROOT, "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv")
DEFAULT_INDEX = os.path.join(ROOT, "data/bm25_indexing_full/corpus_full/lucene_index")
DEFAULT_OUT_DIR = os.path.join(ROOT, "evaluation")


def load_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries[r["_id"]] = r["text"]
    return queries


def load_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, did, score = line.rstrip("\n").split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def make_searcher(index_path):
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
    os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher(index_path)
    s.set_bm25(k1=0.7, b=0.9)
    return s


def doc_text(searcher, docid: str) -> str:
    doc = searcher.doc(docid)
    if doc is None:
        return ""
    try:
        return json.loads(doc.raw())["contents"]
    except Exception:
        return doc.raw() or ""


@torch.no_grad()
def score_pairs(model, tokenizer, query, passages, device, batch_size, max_length):
    scores = []
    for i in range(0, len(passages), batch_size):
        chunk = passages[i : i + batch_size]
        enc = tokenizer(
            [(query, p) for p in chunk],
            padding=True, truncation=True, max_length=max_length, return_tensors="pt",
        ).to(device)
        logits = model(**enc).logits.view(-1).float()
        scores.extend(logits.cpu().tolist())
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="HF id or local path of the BioBERT cross-encoder")
    parser.add_argument("--queries_path", default=DEFAULT_QUERIES)
    parser.add_argument("--qrels_path", default=DEFAULT_QRELS)
    parser.add_argument("--index_path", default=DEFAULT_INDEX)
    parser.add_argument("--top_k_bm25", default=50, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--max_queries", default=None, type=int)
    parser.add_argument("--tag", default="biobert",
                        help="Run name written into the scores TSV")
    parser.add_argument("--scores_path", default=None,
                        help="Where to append the score row (default evaluation/scores_bioasq_test.tsv)")
    parser.add_argument("--run_out", default=None,
                        help="Optional path to write the TREC run file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    queries = load_queries(args.queries_path)
    qrels = load_qrels(args.qrels_path)
    eligible = [q for q in queries if q in qrels]
    if args.max_queries:
        eligible = eligible[: args.max_queries]
    print(f"Queries to evaluate: {len(eligible)}")

    searcher = make_searcher(args.index_path)

    run = {}  # qid -> {docid: score}
    for i, qid in enumerate(eligible, 1):
        q_text = queries[qid]
        hits = searcher.search(q_text, k=args.top_k_bm25)
        docids = [h.docid for h in hits]
        passages = [doc_text(searcher, d) for d in docids]
        if not passages:
            run[qid] = {}
            continue
        scores = score_pairs(
            model, tokenizer, q_text, passages, device, args.batch_size, args.max_length,
        )
        run[qid] = {d: float(s) for d, s in zip(docids, scores)}

        if i % 25 == 0 or i == len(eligible):
            print(f"  reranked {i}/{len(eligible)} queries", flush=True)

    if args.run_out:
        os.makedirs(os.path.dirname(args.run_out) or ".", exist_ok=True)
        with open(args.run_out, "w") as f:
            for qid, docs in run.items():
                ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)
                for rank, (did, score) in enumerate(ranked, 1):
                    f.write(f"{qid} Q0 {did} {rank} {score:.6f} {args.tag}\n")
        print(f"Run written → {args.run_out}")

    metrics = ir_measures.calc_aggregate([nDCG @ 10, RR @ 10], qrels, run)
    ndcg = metrics[nDCG @ 10]
    mrr = metrics[RR @ 10]
    print(f"\n=== {args.tag} ===")
    print(f"nDCG@10 : {ndcg:.4f}")
    print(f"MRR@10  : {mrr:.4f}")

    out_path = args.scores_path or os.path.join(DEFAULT_OUT_DIR, "scores_bioasq_test.tsv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_header = not os.path.exists(out_path)
    with open(out_path, "a") as f:
        if write_header:
            f.write("tag\tmodel\tn_queries\ttop_k_bm25\tnDCG@10\tMRR@10\n")
        f.write(f"{args.tag}\t{args.model}\t{len(eligible)}\t{args.top_k_bm25}\t{ndcg:.4f}\t{mrr:.4f}\n")
    print(f"Appended scores → {out_path}")


if __name__ == "__main__":
    sys.exit(main())
