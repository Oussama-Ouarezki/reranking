"""
Rerank BM25 candidates with monoT5 and evaluate with nDCG@10 and MRR@10.
No pygaggle needed — pure transformers + ir-measures.

Checkpoint : checkpoints/monot5-base-msmarco-100k
Datasets   : TREC DL19 and DL20 passage tracks

Requirements:
    pip install transformers torch ir-measures

Usage:
    # Evaluate on both DL19 and DL20 (default):
    python scripts/rerank_monot5.py

    # Evaluate on a single dataset:
    python scripts/rerank_monot5.py --dataset dl19
    python scripts/rerank_monot5.py --dataset dl20

    # Override checkpoint or batch size:
    python scripts/rerank_monot5.py --checkpoint checkpoints/monot5-base-msmarco-100k --batch-size 4
"""

import os
import argparse
import time
from collections import defaultdict

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import ir_measures
from ir_measures import nDCG, RR, parse_trec_qrels, parse_trec_run

# ─── dataset config ───────────────────────────────────────────────────────────

DATASETS = {
    "dl19": {
        "queries":  "data/trec/queries.dl19-passage.tsv",
        "qrels":    "data/trec/qrels.dl19-passage.txt",
        "bm25_run": "data/runs/bm25_top100.dl19.txt",
    },
    "dl20": {
        "queries":  "data/trec/queries.dl20-passage.tsv",
        "qrels":    "data/trec/qrels.dl20-passage.txt",
        "bm25_run": "data/runs/bm25_top100.dl20.txt",
    },
}

COLLECTION_PATH     = "data/msmarco-passage/collection.tsv"
DEFAULT_CHECKPOINT  = "checkpoints/monot5-base-msmarco-100k"
DEFAULT_DATASETS    = ["dl19", "dl20"]

# monoT5 uses these tokens to score relevance
TOKEN_TRUE  = "▁true"   # token id 1176
TOKEN_FALSE = "▁false"  # token id 6136


# ─── monoT5 reranker ──────────────────────────────────────────────────────────

class MonoT5Reranker:
    def __init__(self, checkpoint, batch_size=8, device=None):
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        print(f"  Loading model from  : {checkpoint}")
        print(f"  Device              : {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model     = T5ForConditionalGeneration.from_pretrained(
            checkpoint,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.model.eval()

        # token ids for "true" and "false"
        self.true_id  = self.tokenizer.convert_tokens_to_ids(TOKEN_TRUE)
        self.false_id = self.tokenizer.convert_tokens_to_ids(TOKEN_FALSE)
        print(f"  true_id={self.true_id}  false_id={self.false_id}\n")

    def score_batch(self, queries, passages):
        """Score a batch of (query, passage) pairs. Returns list of float scores."""
        inputs = [
            f"Query: {q} Document: {p} Relevant:"
            for q, p in zip(queries, passages)
        ]
        enc = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        decoder_input = torch.zeros(
            (len(inputs), 1), dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                decoder_input_ids=decoder_input
            ).logits  # (batch, 1, vocab)

        # softmax over true / false only
        true_false_logits = logits[:, 0, [self.true_id, self.false_id]]  # (batch, 2)
        probs  = torch.softmax(true_false_logits, dim=-1)
        scores = probs[:, 0].cpu().tolist()   # P("true")
        return scores

    def rerank(self, query, pid_text_pairs):
        """
        pid_text_pairs : list of (pid, passage_text)
        Returns        : list of (pid, score) sorted by score descending
        """
        pids   = [p[0] for p in pid_text_pairs]
        texts  = [p[1] for p in pid_text_pairs]
        all_scores = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts   = texts[i : i + self.batch_size]
            batch_queries = [query] * len(batch_texts)
            all_scores.extend(self.score_batch(batch_queries, batch_texts))

        ranked = sorted(zip(pids, all_scores), key=lambda x: x[1], reverse=True)
        return ranked


# ─── data helpers ─────────────────────────────────────────────────────────────

def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            qid, text = line.strip().split("\t", 1)
            queries[qid] = text
    return queries


def load_collection(path):
    print("Loading passage collection (~1-2 min on first run)...")
    collection = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                collection[parts[0]] = parts[1]
    print(f"  Loaded {len(collection):,} passages\n")
    return collection


def load_bm25_run(path, top_k=100):
    run = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            qid, pid, rank = parts[0], parts[2], int(parts[3])
            if rank <= top_k:
                run[qid].append(pid)
    return run


def write_run_file(results, output_path, tag="monot5"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for qid, passages in results.items():
            for rank, (pid, score) in enumerate(passages, start=1):
                f.write(f"{qid} Q0 {pid} {rank} {score:.6f} {tag}\n")
    print(f"  Run file saved → {output_path}")


# ─── evaluation ───────────────────────────────────────────────────────────────

def evaluate(run_path, qrels_path):
    """
    Compute nDCG@10 and MRR@10 using ir-measures.
    Both metrics are standard for TREC DL19 / DL20.
    """
    qrels   = parse_trec_qrels(qrels_path)
    run     = parse_trec_run(run_path)
    metrics = [nDCG@10, RR@10]          # RR@10 == MRR@10 (single-query reciprocal rank)
    results = ir_measures.calc_aggregate(metrics, qrels, run)
    return {str(m): round(v, 4) for m, v in results.items()}


# ─── per-dataset pipeline ─────────────────────────────────────────────────────

def rerank_dataset(dataset_name, checkpoint, batch_size, top_k):
    cfg = DATASETS[dataset_name]

    # sanity-check files
    for label, path in cfg.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not os.path.exists(COLLECTION_PATH):
        raise FileNotFoundError(f"Missing collection: {COLLECTION_PATH}")

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Dataset    : {dataset_name.upper()}")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Batch size : {batch_size}  |  Top-K : {top_k}")
    print(f"{sep}\n")

    reranker   = MonoT5Reranker(checkpoint, batch_size=batch_size)
    queries    = load_queries(cfg["queries"])
    collection = load_collection(COLLECTION_PATH)
    bm25_run   = load_bm25_run(cfg["bm25_run"], top_k=top_k)

    print(f"Reranking {len(bm25_run)} queries for {dataset_name.upper()} ...")
    results = {}
    start   = time.time()

    for i, (qid, pids) in enumerate(bm25_run.items(), start=1):
        pid_text_pairs = [
            (pid, collection[pid]) for pid in pids if pid in collection
        ]
        if qid not in queries:
            print(f"  Warning: query {qid} not found, skipping.")
            continue
        results[qid] = reranker.rerank(queries[qid], pid_text_pairs)

        if i % 10 == 0 or i == len(bm25_run):
            elapsed = time.time() - start
            print(f"  [{i}/{len(bm25_run)}]  elapsed: {elapsed:.1f}s")

    # save TREC-formatted run file
    ckpt_name   = os.path.basename(checkpoint)
    output_path = f"results/{dataset_name}/monot5_{ckpt_name}.txt"
    write_run_file(results, output_path, tag=f"monot5-{ckpt_name}")

    # evaluate
    print("\nEvaluating ...")
    scores = evaluate(output_path, cfg["qrels"])

    print(f"\n{'─'*40}")
    print(f"  monoT5 results on {dataset_name.upper()}")
    print(f"{'─'*40}")
    print(f"  {'Metric':<20} {'Score'}")
    print(f"  {'──────':<20} {'─────'}")
    for metric, value in scores.items():
        label = "MRR@10" if "RR@10" in metric else metric
        print(f"  {label:<20} {value}")
    print(f"{'─'*40}\n")

    # append to evaluation TSV
    eval_path = f"evaluation/scores_{dataset_name}.tsv"
    os.makedirs("evaluation", exist_ok=True)
    write_header = not os.path.exists(eval_path)
    with open(eval_path, "a") as f:
        if write_header:
            f.write("model\t" + "\t".join(scores.keys()) + "\n")
        f.write(f"monot5_{ckpt_name}\t" + "\t".join(str(v) for v in scores.values()) + "\n")
    print(f"  Scores saved → {eval_path}")

    return scores


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rerank BM25 candidates with monoT5 and evaluate nDCG@10 + MRR@10 on TREC DL19/DL20."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["dl19", "dl20"],
        default=DEFAULT_DATASETS,
        help="Dataset(s) to evaluate (default: dl19 dl20)"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to monoT5 checkpoint (default: {DEFAULT_CHECKPOINT})"
    )
    parser.add_argument("--batch-size", type=int, default=8,  help="Inference batch size")
    parser.add_argument("--top-k",      type=int, default=100, help="Number of BM25 candidates to rerank")
    args = parser.parse_args()

    all_scores = {}
    for dataset in args.dataset:
        scores = rerank_dataset(dataset, args.checkpoint, args.batch_size, args.top_k)
        all_scores[dataset] = scores

    # final summary when evaluating more than one dataset
    if len(args.dataset) > 1:
        print("\n" + "=" * 50)
        print("  FINAL SUMMARY")
        print("=" * 50)
        print(f"  Checkpoint : {args.checkpoint}")
        print(f"  {'Dataset':<8} {'nDCG@10':<12} {'MRR@10'}")
        print(f"  {'───────':<8} {'───────':<12} {'──────'}")
        for dataset, scores in all_scores.items():
            ndcg = scores.get("nDCG@10", "N/A")
            # RR@10 key might appear as "RR(rel=1)@10" depending on ir-measures version
            mrr  = next((v for k, v in scores.items() if "RR" in k), "N/A")
            print(f"  {dataset.upper():<8} {str(ndcg):<12} {mrr}")
        print("=" * 50)


if __name__ == "__main__":
    main()