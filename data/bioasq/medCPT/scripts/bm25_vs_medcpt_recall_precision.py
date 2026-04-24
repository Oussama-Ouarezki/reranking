"""
BM25 (Pyserini/Lucene) vs MedCPT dense retrieval — Recall@K & Precision@K.

For each cutoff K in (5, 10, 20, 50, 100) two side-by-side subplots are drawn:
  Left  — Mean Recall@K    : BM25 bar vs MedCPT bar
  Right — Mean Precision@K : BM25 bar vs MedCPT bar

Reads from:
  data/bioasq/processed/queries.jsonl
  data/bioasq/processed/qrels.tsv
  data/bm25_indexing_full/corpus_full/lucene_index   (Pyserini BM25)
  data/bioasq/medCPT/corpus_embeddings.npy           (MedCPT; encoded if absent)
  data/bioasq/medCPT/corpus_ids.json
  data/bioasq/medCPT/corpus.index                    (FAISS)

Writes to:
  data/bioasq/medCPT/images/recall_precision_bm25_vs_medcpt.png

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/medCPT/scripts/bm25_vs_medcpt_recall_precision.py
"""

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import json
import time
from collections import defaultdict
from pathlib import Path

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pyserini.search.lucene import LuceneSearcher

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).resolve().parents[4]
DATA_DIR   = BASE / "data/bioasq/processed"
INDEX_DIR  = BASE / "data/bm25_indexing_full/corpus_full/lucene_index"
MEDCPT_DIR = BASE / "data/bioasq/medCPT"
IMG_DIR    = MEDCPT_DIR / "images"

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL   = "ncbi/MedCPT-Query-Encoder"
EMBED_DIM     = 768
ARTICLE_BATCH = 32
QUERY_BATCH   = 64
MAX_LENGTH    = 512
TOP_K         = 100
CUTOFFS       = (5, 10, 20, 50, 100)
BM25_K1, BM25_B = 0.7, 0.9

COLORS = {
    "bm25_recall":    "#4C72B0",
    "medcpt_recall":  "#673AB7",
    "bm25_prec":      "#DD8452",
    "medcpt_prec":    "#2ecc71",
}


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_queries(path: Path) -> tuple[list[str], list[str]]:
    qids, qtexts = [], []
    with path.open() as f:
        for line in f:
            q = json.loads(line)
            qids.append(q["_id"])
            qtexts.append(q["text"])
    print(f"  Queries : {len(qids):,}")
    return qids, qtexts


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open() as f:
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
    print(f"  Qrels   : {sum(len(v) for v in qrels.values()):,} pairs "
          f"({len(qrels):,} queries)")
    return qrels


def load_corpus(path: Path) -> tuple[list[str], list[str], list[str]]:
    ids, titles, texts = [], [], []
    with path.open() as f:
        for line in f:
            doc = json.loads(line)
            ids.append(doc["_id"])
            titles.append(doc.get("title", ""))
            texts.append(doc["text"])
    print(f"  Corpus  : {len(ids):,} documents")
    return ids, titles, texts


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(
    results: dict[str, list[str]],
    qrels:   dict[str, set[str]],
) -> tuple[dict[int, float], dict[int, float], int]:
    recall_sum    = {k: 0.0 for k in CUTOFFS}
    precision_sum = {k: 0.0 for k in CUTOFFS}
    n = 0
    for qid, ranked in results.items():
        gold = qrels.get(qid, set())
        if not gold:
            continue
        n += 1
        for k in CUTOFFS:
            hits = sum(1 for did in ranked[:k] if did in gold)
            recall_sum[k]    += hits / len(gold)
            precision_sum[k] += hits / k
    mean_recall    = {k: 100 * recall_sum[k]    / n for k in CUTOFFS}
    mean_precision = {k: 100 * precision_sum[k] / n for k in CUTOFFS}
    return mean_recall, mean_precision, n


# ── MedCPT encoding ───────────────────────────────────────────────────────────
def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""
    print(f"  Device  : {device}{tag}")
    return device


def encode_articles(titles: list[str], texts: list[str], device: torch.device) -> np.ndarray:
    tok   = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model = AutoModel.from_pretrained(ARTICLE_MODEL).to(device).eval()
    n     = len(titles)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, n, ARTICLE_BATCH), desc="Encoding corpus", unit="batch"):
            e   = min(s + ARTICLE_BATCH, n)
            enc = tok(titles[s:e], texts[s:e], max_length=MAX_LENGTH,
                      padding=True, truncation=True, return_tensors="pt")
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[s:e] = cls / norm
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embs


def encode_queries(qtexts: list[str], device: torch.device) -> np.ndarray:
    tok   = AutoTokenizer.from_pretrained(QUERY_MODEL)
    model = AutoModel.from_pretrained(QUERY_MODEL).to(device).eval()
    n     = len(qtexts)
    embs  = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, n, QUERY_BATCH), desc="Encoding queries", unit="batch"):
            e   = min(s + QUERY_BATCH, n)
            enc = tok(qtexts[s:e], max_length=MAX_LENGTH,
                      padding=True, truncation=True, return_tensors="pt")
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            embs[s:e] = cls / norm
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return embs


# ── FAISS ─────────────────────────────────────────────────────────────────────
def build_or_load_faiss(corpus_embs: np.ndarray, corpus_ids: list[str]):
    index_path = MEDCPT_DIR / "corpus.index"
    ids_path   = MEDCPT_DIR / "corpus_ids.json"
    if index_path.exists():
        print(f"  Loading cached FAISS index → {index_path}")
        index = faiss.read_index(str(index_path))
        saved_ids = json.loads(ids_path.read_text())
        return index, saved_ids
    print("  Building FAISS IndexFlatIP …")
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(corpus_embs)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(corpus_ids), encoding="utf-8")
    print(f"  {index.ntotal:,} vectors indexed")
    return index, corpus_ids


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot(
    bm25_recall:    dict[int, float],
    bm25_prec:      dict[int, float],
    medcpt_recall:  dict[int, float],
    medcpt_prec:    dict[int, float],
    n_queries: int,
) -> None:
    cutoffs   = list(CUTOFFS)
    x         = np.arange(len(cutoffs))
    bar_width = 0.35

    fig, (ax_r, ax_p) = plt.subplots(1, 2, figsize=(16, 6))

    def add_bars(ax, bm25_vals, medcpt_vals, ylabel, title, c_bm25, c_medcpt):
        bars_b = ax.bar(x - bar_width / 2, bm25_vals,   bar_width,
                        label="BM25 (Pyserini)", color=c_bm25,   alpha=0.88)
        bars_m = ax.bar(x + bar_width / 2, medcpt_vals, bar_width,
                        label="MedCPT",          color=c_medcpt, alpha=0.88)
        for bars, color, vals in [(bars_b, c_bm25, bm25_vals),
                                   (bars_m, c_medcpt, medcpt_vals)]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.4,
                        f"{val:.1f}%", ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"@{k}" for k in cutoffs], fontsize=11)
        ax.set_xlabel("Top-K cutoff", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(0, 110)
        ax.set_title(title, fontsize=12, pad=10)
        ax.legend(fontsize=10)

    add_bars(
        ax_r,
        [bm25_recall[k]   for k in cutoffs],
        [medcpt_recall[k] for k in cutoffs],
        "Mean Recall (%)", "Mean Recall@K",
        COLORS["bm25_recall"], COLORS["medcpt_recall"],
    )
    add_bars(
        ax_p,
        [bm25_prec[k]   for k in cutoffs],
        [medcpt_prec[k] for k in cutoffs],
        "Mean Precision (%)", "Mean Precision@K",
        COLORS["bm25_prec"], COLORS["medcpt_prec"],
    )

    fig.suptitle(
        f"BM25 (Pyserini, k1={BM25_K1}, b={BM25_B}) vs MedCPT Dense Retrieval\n"
        f"BioASQ training set — {n_queries} queries",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / "recall_precision_bm25_vs_medcpt.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    MEDCPT_DIR.mkdir(parents=True, exist_ok=True)

    print("── Loading data ──────────────────────────────────────")
    qids, qtexts = load_queries(DATA_DIR / "queries.jsonl")
    qrels        = load_qrels(DATA_DIR / "qrels.tsv")
    corpus_ids, titles, texts = load_corpus(DATA_DIR / "corpus.jsonl")

    device = get_device()

    # ── MedCPT corpus embeddings ───────────────────────────────────────────────
    emb_path = MEDCPT_DIR / "corpus_embeddings.npy"
    print("\n── MedCPT corpus embeddings ──────────────────────────")
    if emb_path.exists():
        print(f"  Loading cached embeddings → {emb_path}")
        corpus_embs = np.load(str(emb_path))
    else:
        print("  Encoding corpus with MedCPT-Article-Encoder …")
        corpus_embs = encode_articles(titles, texts, device)
        np.save(str(emb_path), corpus_embs)
        print(f"  Saved → {emb_path}")

    # ── FAISS ──────────────────────────────────────────────────────────────────
    print("\n── FAISS index ───────────────────────────────────────")
    index, index_ids = build_or_load_faiss(corpus_embs, corpus_ids)
    id_map = {i: did for i, did in enumerate(index_ids)}

    # ── MedCPT retrieval ───────────────────────────────────────────────────────
    print("\n── MedCPT query encoding ─────────────────────────────")
    query_embs = encode_queries(qtexts, device)
    print(f"\n── MedCPT retrieval (top-{TOP_K}) ────────────────────")
    _, nn_indices = index.search(query_embs, TOP_K)
    medcpt_results: dict[str, list[str]] = {
        qids[i]: [id_map[idx] for idx in nn_indices[i] if idx in id_map]
        for i in range(len(qids))
    }

    # ── BM25 (Pyserini) retrieval ─────────────────────────────────────────────
    print(f"\n── BM25 retrieval (Pyserini, top-{TOP_K}) ────────────")
    t0 = time.time()
    searcher = LuceneSearcher(str(INDEX_DIR))
    searcher.set_bm25(k1=BM25_K1, b=BM25_B)
    batch_hits = searcher.batch_search(
        queries=qtexts, qids=qids, k=TOP_K, threads=4
    )
    print(f"  Done in {time.time()-t0:.2f}s")
    bm25_results: dict[str, list[str]] = {
        qid: [hit.docid for hit in hits]
        for qid, hits in batch_hits.items()
    }

    # ── Metrics ────────────────────────────────────────────────────────────────
    print("\n── Computing metrics ─────────────────────────────────")
    bm25_recall, bm25_prec, n_q = compute_metrics(bm25_results, qrels)
    medcpt_recall, medcpt_prec, _  = compute_metrics(medcpt_results, qrels)

    print(f"\n  {'K':<6}  {'BM25 R':>8}  {'MCPT R':>8}  {'BM25 P':>8}  {'MCPT P':>8}")
    print("  " + "─" * 46)
    for k in CUTOFFS:
        print(f"  @{k:<5}  {bm25_recall[k]:>7.2f}%  {medcpt_recall[k]:>7.2f}%  "
              f"{bm25_prec[k]:>7.2f}%  {medcpt_prec[k]:>7.2f}%")

    # ── Plot ───────────────────────────────────────────────────────────────────
    print("\n── Saving plot ───────────────────────────────────────")
    plot(bm25_recall, bm25_prec, medcpt_recall, medcpt_prec, n_q)
    print("\n── Done ──────────────────────────────────────────────")


if __name__ == "__main__":
    main()
