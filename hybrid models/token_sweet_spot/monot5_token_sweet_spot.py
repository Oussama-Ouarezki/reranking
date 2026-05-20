"""
monoT5 token-length sweet spot analysis on BioASQ Task13BGoldenEnriched test set.

Evaluates monoT5 at max_length ∈ {150, 200, 250, 300, 350, 400, 512} and reports
nDCG@1, nDCG@5, nDCG@10. Produces a line plot and a formatted table.

Pipeline:
  1. BM25 retrieval via pre-built Lucene index (no rebuild)
  2. Retrieve top-50 candidates per query
  3. Rerank with monoT5 at each token budget
  4. Evaluate nDCG@1, nDCG@5, nDCG@10

Usage:
    cd /home/oussama/Desktop/reranking_project
    /home/oussama/miniconda3/envs/pyml/bin/python token_sweet_spot/monot5_token_sweet_spot.py
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
os.environ["PATH"] = "/usr/lib/jvm/java-21-openjdk-amd64/bin:" + os.environ.get("PATH", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import ir_measures
from ir_measures import nDCG, ScoredDoc, Qrel
from pyserini.search.lucene import LuceneSearcher
from transformers import T5ForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

LUCENE_INDEX = "data/bm25_indexing_full/corpus_full/lucene_index"
QUERIES_FILE = Path("data/bioasq/raw/Task13BGoldenEnriched/queries_full.jsonl")
QRELS_FILE   = Path("data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv")

CHECKPOINT  = Path("checkpoints/monot5-base-msmarco-100k")
OUT_DIR     = Path("token_sweet_spot")

TOKEN_LENGTHS = [150, 200, 250, 300, 350, 400, 512]
METRICS       = [nDCG @ 1, nDCG @ 5, nDCG @ 10]
METRIC_LABELS = {str(nDCG @ 1): "nDCG@1", str(nDCG @ 5): "nDCG@5", str(nDCG @ 10): "nDCG@10"}

BM25_TOP_K = 50
BATCH_SIZE = 32

TOKEN_TRUE  = "▁true"
TOKEN_FALSE = "▁false"

# ── Setup ─────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Load queries and qrels ────────────────────────────────────────────────────

print("\nLoading queries …")
queries = {}
with QUERIES_FILE.open() as f:
    for line in f:
        q = json.loads(line)
        queries[q["_id"]] = q["text"]
print(f"  {len(queries):,} queries")

print("Loading qrels …")
qrels = []
with QRELS_FILE.open() as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 3:
            qid, doc_id, score = parts
        elif len(parts) == 4:
            qid, _, doc_id, score = parts
        else:
            continue
        qrels.append(Qrel(qid, doc_id, int(score)))
print(f"  {len(qrels):,} qrel entries")

# ── BM25 retrieval via Lucene index ───────────────────────────────────────────

print(f"\nLoading Lucene index from {LUCENE_INDEX} …")
searcher = LuceneSearcher(LUCENE_INDEX)
searcher.set_bm25(k1=0.9, b=0.4)
print("  Searcher ready.")

print(f"\nRetrieving BM25 top-{BM25_TOP_K} for {len(queries):,} queries …")
bm25_candidates = {}
for qid, qtext in tqdm(queries.items(), desc="BM25"):
    hits = searcher.search(qtext, k=BM25_TOP_K)
    candidates = []
    for hit in hits:
        raw = json.loads(searcher.doc(hit.docid).raw())
        candidates.append((hit.docid, raw["contents"]))
    bm25_candidates[qid] = candidates

# ── Load monoT5 ───────────────────────────────────────────────────────────────

print(f"\nLoading monoT5 from {CHECKPOINT} …")
tokenizer = AutoTokenizer.from_pretrained(str(CHECKPOINT))
model = T5ForConditionalGeneration.from_pretrained(
    str(CHECKPOINT),
    torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
).to(device).eval()

true_id  = tokenizer.convert_tokens_to_ids(TOKEN_TRUE)
false_id = tokenizer.convert_tokens_to_ids(TOKEN_FALSE)
print(f"  true_id={true_id}  false_id={false_id}")


# ── Reranking helpers ─────────────────────────────────────────────────────────

def score_batch(query: str, passages: list[str], max_length: int) -> list[float]:
    inputs = [f"Query: {query} Document: {p} Relevant:" for p in passages]
    enc = tokenizer(
        inputs, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt",
    ).to(device)
    decoder_input = torch.zeros((len(inputs), 1), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input,
        ).logits  # (batch, 1, vocab)
    probs = torch.softmax(logits[:, 0, [true_id, false_id]], dim=-1)
    return probs[:, 0].cpu().tolist()


def rerank(query: str, candidates: list, max_length: int) -> list:
    all_scores = []
    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i : i + BATCH_SIZE]
        all_scores.extend(score_batch(query, [p for _, p in batch], max_length))
    ranked = sorted(zip(candidates, all_scores), key=lambda x: x[1], reverse=True)
    return [(doc_id, score) for (doc_id, _), score in ranked]


# ── Sweep over token lengths ──────────────────────────────────────────────────

results   = {}  # max_length → {metric_label: score}
timings   = {}  # max_length → seconds (reranking only)

for max_len in TOKEN_LENGTHS:
    print(f"\n{'='*55}")
    print(f"  max_length = {max_len}")
    print(f"{'='*55}")

    run = []
    t0  = time.perf_counter()
    for qid, qtext in tqdm(queries.items(), desc=f"monoT5 [{max_len}]"):
        ranked = rerank(qtext, bm25_candidates[qid], max_len)
        for doc_id, score in ranked:
            run.append(ScoredDoc(qid, doc_id, score=score))
    elapsed = time.perf_counter() - t0
    timings[max_len] = elapsed

    agg = ir_measures.calc_aggregate(METRICS, qrels, run)
    scores = {METRIC_LABELS[str(m)]: round(v, 4) for m, v in agg.items()}
    results[max_len] = scores

    print(f"  nDCG@1={scores['nDCG@1']}  nDCG@5={scores['nDCG@5']}  nDCG@10={scores['nDCG@10']}  time={elapsed:.1f}s")

# ── Print table ───────────────────────────────────────────────────────────────

col_w = 10
sep = "─" * (col_w + col_w * 3 + col_w + 2)

print(f"\n{sep}")
print(f"  monoT5 Token-Length Sweet Spot  —  BioASQ Task13BGoldenEnriched")
print(f"{sep}")
print(f"  {'Tokens':<{col_w}}{'nDCG@1':>{col_w}}{'nDCG@5':>{col_w}}{'nDCG@10':>{col_w}}{'Time (s)':>{col_w}}")
print(f"  {'------':<{col_w}}{'------':>{col_w}}{'------':>{col_w}}{'-------':>{col_w}}{'--------':>{col_w}}")
for max_len, scores in results.items():
    print(
        f"  {max_len:<{col_w}}"
        f"{scores['nDCG@1']:>{col_w}.4f}"
        f"{scores['nDCG@5']:>{col_w}.4f}"
        f"{scores['nDCG@10']:>{col_w}.4f}"
        f"{timings[max_len]:>{col_w}.1f}"
    )
print(f"{sep}\n")

# ── Save table to TSV ─────────────────────────────────────────────────────────

tsv_path = OUT_DIR / "monot5_token_sweet_spot.tsv"
with tsv_path.open("w") as f:
    f.write("max_length\tnDCG@1\tnDCG@5\tnDCG@10\ttime_s\n")
    for max_len, scores in results.items():
        f.write(
            f"{max_len}\t{scores['nDCG@1']}\t{scores['nDCG@5']}\t{scores['nDCG@10']}"
            f"\t{timings[max_len]:.1f}\n"
        )
print(f"Table saved → {tsv_path}")

# ── Plot ──────────────────────────────────────────────────────────────────────

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

token_lens  = list(results.keys())
ndcg1_vals  = [results[t]["nDCG@1"]  for t in token_lens]
ndcg5_vals  = [results[t]["nDCG@5"]  for t in token_lens]
ndcg10_vals = [results[t]["nDCG@10"] for t in token_lens]

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(token_lens, ndcg1_vals,  marker="o", linewidth=2, label="nDCG@1",  color="#e74c3c")
ax.plot(token_lens, ndcg5_vals,  marker="s", linewidth=2, label="nDCG@5",  color="#2980b9")
ax.plot(token_lens, ndcg10_vals, marker="^", linewidth=2, label="nDCG@10", color="#27ae60")

for x, y1, y5, y10 in zip(token_lens, ndcg1_vals, ndcg5_vals, ndcg10_vals):
    ax.annotate(f"{y1:.4f}",  (x, y1),  textcoords="offset points", xytext=(0,  8), ha="center", fontsize=8, color="#e74c3c")
    ax.annotate(f"{y5:.4f}",  (x, y5),  textcoords="offset points", xytext=(0,  8), ha="center", fontsize=8, color="#2980b9")
    ax.annotate(f"{y10:.4f}", (x, y10), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#27ae60")

ax.set_title(
    "monoT5 Token-Length Sweet Spot — BioASQ Task13BGoldenEnriched\n"
    f"(BM25 top-{BM25_TOP_K} → monoT5 rerank, checkpoint: monot5-base-msmarco-100k)",
    fontsize=13, pad=14,
)
ax.set_xlabel("max_length (tokens)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(token_lens)
ax.legend(fontsize=11, loc="lower right")
ax.set_ylim(
    min(ndcg1_vals + ndcg5_vals + ndcg10_vals) - 0.04,
    max(ndcg1_vals + ndcg5_vals + ndcg10_vals) + 0.06,
)

plt.tight_layout()
plot_path = OUT_DIR / "monot5_token_sweet_spot.png"
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"Plot  saved → {plot_path}")

# ── Timing bar chart ──────────────────────────────────────────────────────────

time_vals = [timings[t] for t in token_lens]

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(
    [str(t) for t in token_lens],
    time_vals,
    color="#8e44ad",
    alpha=0.85,
    width=0.55,
)

for bar, val in zip(bars, time_vals):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(time_vals) * 0.01,
        f"{val:.1f}s",
        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#8e44ad",
    )

ax.set_xlabel("max_length (tokens)", fontsize=12)
ax.set_ylabel("Reranking time (seconds)", fontsize=12)
ax.set_title(
    "monoT5 Reranking Time per Token Budget — BioASQ Task13BGoldenEnriched\n"
    f"({len(queries):,} queries × BM25 top-{BM25_TOP_K} candidates, batch_size={BATCH_SIZE})",
    fontsize=13, pad=12,
)
ax.set_ylim(0, max(time_vals) * 1.15)

plt.tight_layout()
time_plot_path = OUT_DIR / "monot5_token_sweet_spot_time.png"
plt.savefig(time_plot_path, dpi=150)
plt.close()
print(f"Plot  saved → {time_plot_path}")
print("\nDone.")
