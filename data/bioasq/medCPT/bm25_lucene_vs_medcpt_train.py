"""
BM25 (Lucene, full documents — no truncation) vs MedCPT (512-token truncation)
on the BioASQ training set, both over the full corpus.

Corpus  : data/bioasq/pubmed_full/full/corpus_full_processed.jsonl  (257 812 docs)
Queries : data/bioasq/processed/queries.jsonl                        (training set)
Qrels   : data/bioasq/processed/qrels.tsv

BM25 uses the pre-built Lucene index — no rebuild.
MedCPT embeddings are cached in data/bioasq/medCPT/full_corpus/ and reused on
subsequent runs.

Outputs:
    data/bioasq/medCPT/images/bm25_vs_medcpt_train_recall.png
    data/bioasq/medCPT/bm25_vs_medcpt_train_recall.tsv

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python \\
        data/bioasq/medCPT/bm25_lucene_vs_medcpt_train.py
"""

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import faiss
import numpy as np
import torch
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── Config ────────────────────────────────────────────────────────────────────

BASE         = Path("data/bioasq")
CORPUS_FILE  = Path("data/bioasq/pubmed_full/full/corpus_full_processed.jsonl")
QUERIES_FILE = Path("data/bioasq/processed/queries.jsonl")
QRELS_FILE   = Path("data/bioasq/processed/qrels.tsv")
LUCENE_INDEX = "data/bm25_indexing_full/corpus_full_processed/lucene_index"

EMB_CACHE_DIR = Path("data/bioasq/medCPT/full_corpus")
IMG_DIR       = Path("data/bioasq/medCPT/images")
OUT_TSV       = Path("data/bioasq/medCPT/bm25_vs_medcpt_train_recall.tsv")

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL   = "ncbi/MedCPT-Query-Encoder"
MAX_LENGTH    = 512
ARTICLE_BATCH = 32
QUERY_BATCH   = 64
EMBED_DIM     = 768

BM25_TOP_K     = 100
RECALL_CUTOFFS = [5, 10, 20, 50, 100]

COLORS = {"BM25": "#2196F3", "MedCPT": "#673AB7"}

# ── Setup ─────────────────────────────────────────────────────────────────────

EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU   : {torch.cuda.get_device_name(0)}")

# ── Load corpus ───────────────────────────────────────────────────────────────

print(f"\nLoading corpus from {CORPUS_FILE} …")
corpus_ids, titles, texts = [], [], []
with CORPUS_FILE.open(encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        corpus_ids.append(doc["_id"])
        titles.append(doc.get("title", ""))
        texts.append(doc["text"])
print(f"  {len(corpus_ids):,} documents")

corpus_id_set = set(corpus_ids)

# ── Load training queries ─────────────────────────────────────────────────────

print(f"\nLoading queries from {QUERIES_FILE} …")
qids, qtexts = [], []
with QUERIES_FILE.open(encoding="utf-8") as f:
    for line in f:
        q = json.loads(line)
        qids.append(q["_id"])
        qtexts.append(q["text"])
print(f"  {len(qids):,} queries")

# ── Load qrels ────────────────────────────────────────────────────────────────

print(f"\nLoading qrels from {QRELS_FILE} …")
qrels: dict[str, set[str]] = {}
with QRELS_FILE.open(encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        qid, did = row["query-id"], row["corpus-id"]
        if did not in corpus_id_set or int(row["score"]) < 1:
            continue
        qrels.setdefault(qid, set()).add(did)
print(f"  {sum(len(v) for v in qrels.values()):,} pairs  ({len(qrels):,} queries with ≥1 relevant)")

# ── BM25 retrieval via Lucene (full documents, no truncation) ─────────────────

print(f"\nLoading Lucene index from {LUCENE_INDEX} …")
searcher = LuceneSearcher(LUCENE_INDEX)
searcher.set_bm25(k1=0.9, b=0.4)
print("  Searcher ready.")

print(f"\nBM25 retrieval (top-{BM25_TOP_K}) …")
bm25_top: dict[str, list[str]] = {}
for qid, qtext in tqdm(zip(qids, qtexts), total=len(qids), desc="BM25"):
    hits = searcher.search(qtext, k=BM25_TOP_K)
    bm25_top[qid] = [hit.docid for hit in hits]

# ── MedCPT corpus embeddings (encode once, then cache) ───────────────────────

emb_path = EMB_CACHE_DIR / "corpus_embeddings.npy"
ids_path = EMB_CACHE_DIR / "corpus_ids.json"
idx_path = EMB_CACHE_DIR / "corpus.index"

print(f"\nMedCPT corpus embeddings (cache: {EMB_CACHE_DIR}) …")
if emb_path.exists() and ids_path.exists():
    cached_ids = json.loads(ids_path.read_text())
    if len(cached_ids) == len(corpus_ids):
        print(f"  Reusing cached embeddings ({len(cached_ids):,} docs)")
        corpus_embs = np.load(str(emb_path))
    else:
        print(f"  Cache mismatch ({len(cached_ids):,} vs {len(corpus_ids):,}), re-encoding …")
        emb_path.unlink(missing_ok=True)
        ids_path.unlink(missing_ok=True)
        idx_path.unlink(missing_ok=True)
        corpus_embs = None
else:
    corpus_embs = None

if corpus_embs is None:
    print(f"  Encoding {len(corpus_ids):,} docs with MedCPT-Article-Encoder (max_length={MAX_LENGTH}) …")
    tok_art   = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model_art = AutoModel.from_pretrained(ARTICLE_MODEL).to(device).eval()
    n = len(corpus_ids)
    corpus_embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, n, ARTICLE_BATCH), desc="Encoding corpus", unit="batch"):
            e   = min(s + ARTICLE_BATCH, n)
            enc = tok_art(
                titles[s:e], texts[s:e],
                max_length=MAX_LENGTH, padding=True, truncation=True,
                return_tensors="pt",
            )
            enc  = {k: v.to(device) for k, v in enc.items()}
            cls  = model_art(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
            norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
            corpus_embs[s:e] = cls / norm
    del model_art
    if device.type == "cuda":
        torch.cuda.empty_cache()
    np.save(str(emb_path), corpus_embs)
    ids_path.write_text(json.dumps(corpus_ids))
    print(f"  Embeddings saved → {emb_path}")

# ── FAISS index ───────────────────────────────────────────────────────────────

print("\nBuilding/loading FAISS index …")
if idx_path.exists():
    print(f"  Loading cached index → {idx_path}")
    faiss_index = faiss.read_index(str(idx_path))
else:
    print("  Building IndexFlatIP …")
    faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    faiss_index.add(corpus_embs)
    faiss.write_index(faiss_index, str(idx_path))
    print(f"  {faiss_index.ntotal:,} vectors — saved → {idx_path}")

# ── MedCPT query encoding + retrieval ────────────────────────────────────────

print(f"\nEncoding {len(qids):,} queries with MedCPT-Query-Encoder …")
tok_q   = AutoTokenizer.from_pretrained(QUERY_MODEL)
model_q = AutoModel.from_pretrained(QUERY_MODEL).to(device).eval()
n = len(qids)
query_embs = np.zeros((n, EMBED_DIM), dtype=np.float32)
with torch.no_grad():
    for s in tqdm(range(0, n, QUERY_BATCH), desc="Encoding queries", unit="batch"):
        e   = min(s + QUERY_BATCH, n)
        enc = tok_q(
            qtexts[s:e],
            max_length=MAX_LENGTH, padding=True, truncation=True,
            return_tensors="pt",
        )
        enc  = {k: v.to(device) for k, v in enc.items()}
        cls  = model_q(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
        norm = np.linalg.norm(cls, axis=1, keepdims=True) + 1e-10
        query_embs[s:e] = cls / norm
del model_q
if device.type == "cuda":
    torch.cuda.empty_cache()

print(f"\nMedCPT retrieval (top-{BM25_TOP_K}) …")
_, retrieved = faiss_index.search(query_embs, BM25_TOP_K)
medcpt_top: dict[str, list[str]] = {
    qids[i]: [corpus_ids[idx] for idx in retrieved[i]]
    for i in range(len(qids))
}

# ── Recall@K ──────────────────────────────────────────────────────────────────

def recall_at_k(top: dict[str, list[str]], cutoffs: list[int]) -> dict[int, float]:
    recall_lists: dict[int, list[float]] = defaultdict(list)
    for qid, rel in qrels.items():
        if qid not in top:
            continue
        retrieved_ids = top[qid]
        for k in cutoffs:
            hits = len(set(retrieved_ids[:k]) & rel)
            recall_lists[k].append(hits / len(rel))
    return {k: float(np.mean(v)) for k, v in recall_lists.items()}


print("\nComputing Recall@K …")
bm25_recall   = recall_at_k(bm25_top,   RECALL_CUTOFFS)
medcpt_recall = recall_at_k(medcpt_top, RECALL_CUTOFFS)

# ── Print table ───────────────────────────────────────────────────────────────

sep = "─" * 48
print(f"\n{sep}")
print(f"  BM25 (Lucene, full docs) vs MedCPT (512 tokens)")
print(f"  Training queries — full corpus ({len(corpus_ids):,} docs)")
print(f"{sep}")
print(f"  {'@K':<8}{'BM25':>10}{'MedCPT':>10}{'Δ':>10}")
print(f"  {'──':<8}{'────':>10}{'──────':>10}{'─':>10}")
for k in RECALL_CUTOFFS:
    delta = medcpt_recall[k] - bm25_recall[k]
    sign  = "+" if delta >= 0 else ""
    print(f"  @{k:<7}{bm25_recall[k]:>10.4f}{medcpt_recall[k]:>10.4f}  {sign}{delta:.4f}")
print(f"{sep}\n")

# ── Save TSV ──────────────────────────────────────────────────────────────────

with OUT_TSV.open("w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["cutoff", "bm25_recall", "medcpt_recall", "delta"])
    for k in RECALL_CUTOFFS:
        delta = medcpt_recall[k] - bm25_recall[k]
        w.writerow([k, f"{bm25_recall[k]:.4f}", f"{medcpt_recall[k]:.4f}", f"{delta:+.4f}"])
print(f"TSV saved → {OUT_TSV}")

# ── Bar chart ─────────────────────────────────────────────────────────────────

bar_width = 0.35
x = np.arange(len(RECALL_CUTOFFS))

fig, ax = plt.subplots(figsize=(11, 6))

bars_b = ax.bar(
    x - bar_width / 2,
    [bm25_recall[k] for k in RECALL_CUTOFFS],
    width=bar_width, color=COLORS["BM25"], alpha=0.88,
    label=f"BM25 — Lucene (full docs, no truncation)",
)
bars_m = ax.bar(
    x + bar_width / 2,
    [medcpt_recall[k] for k in RECALL_CUTOFFS],
    width=bar_width, color=COLORS["MedCPT"], alpha=0.88,
    label=f"MedCPT (512-token truncation)",
)

for bars, color in [(bars_b, COLORS["BM25"]), (bars_m, COLORS["MedCPT"])]:
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.007,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none", alpha=0.85),
        )

ax.set_xticks(x)
ax.set_xticklabels([f"@{k}" for k in RECALL_CUTOFFS], fontsize=12)
ax.set_ylabel("Mean Recall@K", fontsize=12)
ax.set_title(
    "BM25 (Lucene, full docs) vs MedCPT (512-token truncation)\n"
    f"BioASQ Training Set — Full Corpus ({len(corpus_ids):,} docs)",
    fontsize=13, pad=12,
)
ax.set_ylim(0, 1.08)
ax.legend(fontsize=11, loc="upper left")

plt.tight_layout()
out_img = IMG_DIR / "bm25_vs_medcpt_train_recall.png"
plt.savefig(out_img, dpi=150)
plt.close()
print(f"Plot  saved → {out_img}")
print("\nDone.")
