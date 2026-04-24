"""
Mine hard negatives and easy negatives for BioASQ listwise dataset construction.

Pipeline per query:
  1. BM25 top-100 from full Lucene index (257 907-doc corpus)
  2. Identify gold passages from qrels (up to 10)
  3. Easy negatives  : 2 docs sampled from BM25 tail (ranks 80-100), not gold
  4. Hard negatives  : remaining non-gold BM25 top-100 docs scored by
                       MedCPT Cross-Encoder → sigmoid probability, sorted desc

Selection: 2 000 most recent queries with ≥ 3 gold documents.

Output: data/bioasq/hard_negatives_full/candidates.jsonl
One JSON record per line:
{
  "qid"            : str,
  "query"          : str,
  "n_gold"         : int,
  "gold_passages"  : [{"docid", "title", "text"}, ...],          # ≤ 10
  "hard_negatives" : [{"docid", "title", "text", "ce_score",
                        "bm25_rank"}, ...],                       # sorted by ce_score desc
  "easy_negatives" : [{"docid", "title", "text", "bm25_rank"}]   # 2
}

Resume: progress.json stores completed qids — safe to interrupt and restart.

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/hard_negatives_full/mine_candidates.py
"""

import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
BASE         = Path(__file__).resolve().parents[3]
CORPUS_PATH  = BASE / "data" / "bioasq" / "pubmed_full" / "full" / "corpus_full_processed.jsonl"
INDEX_DIR    = BASE / "data" / "bm25_indexing_full" / "corpus_full_processed" / "lucene_index"
DATA_DIR     = BASE / "data" / "bioasq" / "processed"
OUT_DIR      = BASE / "data" / "bioasq" / "hard_negatives_full"

CROSS_ENCODER   = "ncbi/MedCPT-Cross-Encoder"
BM25_K          = 100
BM25_K1         = 0.7
BM25_B          = 0.9
N_SELECT        = 2000        # most recent queries with ≥ MIN_GOLD gold docs
MIN_GOLD        = 3
MAX_GOLD        = 10          # cap gold passages per query
N_EASY          = 2           # easy negatives per query
EASY_TAIL_START = 79          # BM25 rank 80 (0-indexed)
CE_BATCH        = 64          # pairs per cross-encoder forward pass
MAX_CE_LEN      = 512
SEED            = 42

random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_corpus(path: Path) -> dict[str, dict]:
    """Load corpus into {docid: {title, text}} for fast lookup."""
    corpus: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading corpus", unit=" docs"):
            doc = json.loads(line)
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text":  doc.get("text",  ""),
            }
    return corpus


def load_queries(path: Path) -> dict[str, dict]:
    queries: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            queries[q["_id"]] = {
                "text":  q["text"],
                "order": q.get("chronological_order", 0),
            }
    return queries


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                qid, did, score = parts
            elif len(parts) == 4:
                qid, _, did, score = parts
            else:
                continue
            if int(score) > 0:
                qrels[qid].add(did)
    return qrels


def select_queries(
    queries: dict[str, dict],
    qrels: dict[str, set[str]],
    n: int,
    min_gold: int,
) -> list[tuple[str, dict]]:
    """Return n most recent queries (by chronological_order) with ≥ min_gold gold docs."""
    eligible = [
        (qid, meta) for qid, meta in queries.items()
        if len(qrels.get(qid, set())) >= min_gold
    ]
    eligible.sort(key=lambda x: x[1]["order"], reverse=True)
    return eligible[:n]


# ── Cross-Encoder ─────────────────────────────────────────────────────────────
def load_cross_encoder(model_name: str):
    print(f"Loading cross-encoder: {model_name}  (device={DEVICE})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to(DEVICE)
    return model, tokenizer


@torch.inference_mode()
def score_pairs(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    query: str,
    doc_texts: list[str],
    batch_size: int = CE_BATCH,
) -> list[float]:
    """Return sigmoid probability for each (query, doc_text) pair."""
    scores: list[float] = []
    for i in range(0, len(doc_texts), batch_size):
        batch = [[query, text] for text in doc_texts[i : i + batch_size]]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_CE_LEN,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        logits = model(**enc).logits.squeeze(-1)
        probs  = torch.sigmoid(logits).cpu().tolist()
        # squeeze(-1) returns a scalar when batch_size==1 → ensure list
        if isinstance(probs, float):
            probs = [probs]
        scores.extend(probs)
    return scores


# ── Progress / resume ─────────────────────────────────────────────────────────
def load_progress(path: Path) -> set[str]:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_progress(path: Path, completed: set[str]) -> None:
    path.write_text(json.dumps(list(completed)))


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    from pyserini.search.lucene import LuceneSearcher  # import after JAVA_HOME set

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file      = OUT_DIR / "candidates.jsonl"
    progress_file = OUT_DIR / "progress.json"

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading corpus …")
    corpus = load_corpus(CORPUS_PATH)
    print(f"  {len(corpus):,} documents")

    print("Loading queries & qrels …")
    queries = load_queries(DATA_DIR / "queries.jsonl")
    qrels   = load_qrels(DATA_DIR / "qrels.tsv")

    selected = select_queries(queries, qrels, N_SELECT, MIN_GOLD)
    print(f"  Selected {len(selected):,} queries  "
          f"(order {selected[0][1]['order']}–{selected[-1][1]['order']})")

    # ── Models ────────────────────────────────────────────────────────────────
    print("\nLoading Lucene searcher …")
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)

    ce_model, ce_tokenizer = load_cross_encoder(CROSS_ENCODER)

    # ── Resume ────────────────────────────────────────────────────────────────
    completed = load_progress(progress_file)
    print(f"\nResuming from {len(completed)} completed queries …" if completed
          else "\nStarting fresh …")

    # ── Mine ──────────────────────────────────────────────────────────────────
    t_start = time.time()

    with out_file.open("a", encoding="utf-8") as fout:
        for qid, meta in tqdm(selected, desc="Queries", unit="q"):
            if qid in completed:
                continue

            query_text = meta["text"]
            gold_ids   = qrels[qid]

            # ── BM25 top-100 ──────────────────────────────────────────────────
            hits = searcher.search(query_text, k=BM25_K)
            bm25_ranked: list[tuple[str, int]] = [
                (hit.docid, rank)
                for rank, hit in enumerate(hits, 1)
            ]

            # ── Gold passages (up to MAX_GOLD, must exist in corpus) ──────────
            gold_in_corpus = [did for did in gold_ids if did in corpus]
            # stable order (qrels is a set) — sort for reproducibility
            gold_in_corpus.sort()
            gold_selected  = gold_in_corpus[:MAX_GOLD]
            gold_id_set    = set(gold_selected)

            gold_passages = [
                {"docid": did, **corpus[did]}
                for did in gold_selected
            ]

            # ── Easy negatives: 2 from BM25 tail (ranks 80–100, not gold) ────
            tail = [
                (did, rank)
                for did, rank in bm25_ranked[EASY_TAIL_START:]
                if did not in gold_id_set and did in corpus
            ]
            random.shuffle(tail)
            easy_selected    = tail[:N_EASY]
            easy_id_set      = {did for did, _ in easy_selected}

            easy_negatives = [
                {"docid": did, **corpus[did], "bm25_rank": rank}
                for did, rank in easy_selected
            ]

            # ── Hard negative candidates (non-gold, non-easy, in corpus) ──────
            hard_candidates = [
                (did, rank)
                for did, rank in bm25_ranked
                if did not in gold_id_set
                and did not in easy_id_set
                and did in corpus
            ]

            # ── Score with MedCPT Cross-Encoder ──────────────────────────────
            hard_negatives: list[dict] = []
            if hard_candidates:
                doc_texts = [
                    (corpus[did]["title"] + " " + corpus[did]["text"]).strip()
                    for did, _ in hard_candidates
                ]
                ce_scores = score_pairs(ce_model, ce_tokenizer, query_text, doc_texts)

                hard_negatives = [
                    {
                        "docid":    did,
                        **corpus[did],
                        "ce_score": round(score, 6),
                        "bm25_rank": rank,
                    }
                    for (did, rank), score in zip(hard_candidates, ce_scores)
                ]
                hard_negatives.sort(key=lambda x: x["ce_score"], reverse=True)

            # ── Write record ──────────────────────────────────────────────────
            record = {
                "qid":            qid,
                "query":          query_text,
                "n_gold":         len(gold_passages),
                "gold_passages":  gold_passages,
                "hard_negatives": hard_negatives,
                "easy_negatives": easy_negatives,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            completed.add(qid)
            if len(completed) % 50 == 0:
                save_progress(progress_file, completed)

    save_progress(progress_file, completed)
    elapsed = time.time() - t_start
    print(f"\nDone. {len(completed):,} queries processed in {elapsed/60:.1f} min")
    print(f"Output → {out_file}")


if __name__ == "__main__":
    main()
