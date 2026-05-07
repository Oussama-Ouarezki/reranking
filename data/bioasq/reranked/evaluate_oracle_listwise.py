"""
Evaluate the oracle_listwise.jsonl ranked list against BioASQ gold qrels.

Compares three runs:
  BM25      : bm25_order field  (first-stage baseline)
  Oracle    : permutation field (gold-sorted, upper bound)
  DeepSeek  : permutation from deepseek_sliding_reranked_prompt_2.jsonl (teacher)

Metrics: nDCG@5/10/20, MRR, Recall@5/10/20, P@5/10/20

Usage:
    python data/bioasq/reranked/evaluate_oracle_listwise.py
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import ir_measures
from ir_measures import nDCG, RR, Recall, P, ScoredDoc, Qrel

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

ROOT        = Path(__file__).resolve().parents[3]
ORACLE_FILE = ROOT / "data/bioasq/reranked/oracle_listwise.jsonl"
DEEPSEEK_FILE = ROOT / "data/bioasq/reranked/deepseek_sliding_reranked_prompt_2.jsonl"
QRELS_FILE  = ROOT / "data/bioasq/processed/qrels.tsv"
OUT_IMG     = Path(__file__).parent / "images" / "oracle_vs_bm25_vs_deepseek.png"
OUT_IMG.parent.mkdir(parents=True, exist_ok=True)

METRICS = [
    nDCG @ 5,  nDCG @ 10,  nDCG @ 20,
    RR,
    Recall @ 5, Recall @ 10, Recall @ 20,
    P @ 5,      P @ 10,      P @ 20,
]
LABELS = {
    str(nDCG @ 5):    "nDCG@5",
    str(nDCG @ 10):   "nDCG@10",
    str(nDCG @ 20):   "nDCG@20",
    str(RR):          "MRR",
    str(Recall @ 5):  "Recall@5",
    str(Recall @ 10): "Recall@10",
    str(Recall @ 20): "Recall@20",
    str(P @ 5):       "P@5",
    str(P @ 10):      "P@10",
    str(P @ 20):      "P@20",
}


# ── Load data ─────────────────────────────────────────────────────────────────

def load_qrels(path: Path) -> list[Qrel]:
    qrels = []
    with path.open() as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                qrels.append(Qrel(parts[0], parts[1], int(parts[2])))
    return qrels


def scored_docs_from_order(qid: str, order: list[str]) -> list[ScoredDoc]:
    n = len(order)
    return [ScoredDoc(qid, did, float(n - rank)) for rank, did in enumerate(order)]


print("Loading …")
qrels = load_qrels(QRELS_FILE)

oracle_entries   = {}
bm25_entries     = {}
deepseek_entries = {}

with ORACLE_FILE.open() as f:
    for line in f:
        d = json.loads(line)
        oracle_entries[d["qid"]]  = d["permutation"]
        bm25_entries[d["qid"]]    = d["bm25_order"]

with DEEPSEEK_FILE.open() as f:
    for line in f:
        d = json.loads(line)
        deepseek_entries[d["qid"]] = d["permutation"]

qids = list(oracle_entries.keys())
print(f"  {len(qids):,} queries  |  {len(qrels):,} qrel entries")

# Build scored-doc lists
bm25_run     = []
oracle_run   = []
deepseek_run = []

for qid in qids:
    bm25_run.extend(scored_docs_from_order(qid, bm25_entries[qid]))
    oracle_run.extend(scored_docs_from_order(qid, oracle_entries[qid]))
    if qid in deepseek_entries:
        deepseek_run.extend(scored_docs_from_order(qid, deepseek_entries[qid]))

# ── Evaluate ──────────────────────────────────────────────────────────────────

def calc(run):
    return {str(m): round(v, 4)
            for m, v in ir_measures.calc_aggregate(METRICS, qrels, run).items()}

print("Evaluating …")
bm25_scores     = calc(bm25_run)
oracle_scores   = calc(oracle_run)
deepseek_scores = calc(deepseek_run)

# ── Print table ───────────────────────────────────────────────────────────────

sep = "─" * 72
print(f"\n{sep}")
print(f"  {'Metric':<12} {'BM25':>10} {'DeepSeek':>12} {'Oracle':>10}  {'Δ BM25→Oracle':>14}")
print(f"  {'──────':<12} {'────':>10} {'────────':>12} {'──────':>10}  {'─────────────':>14}")

for m_key, label in LABELS.items():
    bv = bm25_scores.get(m_key, float("nan"))
    dv = deepseek_scores.get(m_key, float("nan"))
    ov = oracle_scores.get(m_key, float("nan"))
    delta = round(ov - bv, 4)
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:<12} {bv:>10.4f} {dv:>12.4f} {ov:>10.4f}  {sign}{delta:.4f}")

print(f"{sep}\n")

# ── Per-query nDCG@10 distribution ───────────────────────────────────────────

def per_query_ndcg10(run_list: list[ScoredDoc]) -> dict[str, float]:
    by_qid: dict[str, list[ScoredDoc]] = defaultdict(list)
    for s in run_list:
        by_qid[s.query_id].append(s)
    qrel_map: dict[str, list[Qrel]] = defaultdict(list)
    for q in qrels:
        qrel_map[q.query_id].append(q)
    scores = {}
    metric = nDCG @ 10
    for qid, docs in by_qid.items():
        if qrel_map[qid]:
            res = ir_measures.calc_aggregate([metric], qrel_map[qid], docs)
            scores[qid] = float(dict(res).get(metric, 0.0))
    return scores

print("Computing per-query nDCG@10 …")
bm25_pq     = per_query_ndcg10(bm25_run)
oracle_pq   = per_query_ndcg10(oracle_run)
deepseek_pq = per_query_ndcg10(deepseek_run)

common_qids = sorted(set(bm25_pq) & set(oracle_pq) & set(deepseek_pq))
bm25_vals     = [bm25_pq[q]     for q in common_qids]
oracle_vals   = [oracle_pq[q]   for q in common_qids]
deepseek_vals = [deepseek_pq[q] for q in common_qids]

gains_oracle   = [o - b for o, b in zip(oracle_vals, bm25_vals)]
gains_deepseek = [d - b for d, b in zip(deepseek_vals, bm25_vals)]

print(f"  Queries where oracle > BM25   : {sum(1 for g in gains_oracle if g > 0):,} / {len(common_qids):,}")
print(f"  Queries where deepseek > BM25 : {sum(1 for g in gains_deepseek if g > 0):,} / {len(common_qids):,}")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. Bar chart of key metrics
key_metrics = ["nDCG@5", "nDCG@10", "nDCG@20", "MRR", "Recall@10"]
key_keys    = [k for k, v in LABELS.items() if v in key_metrics]

x = range(len(key_metrics))
width = 0.26
bm25_vals_bar     = [bm25_scores.get(k, 0)     for k in key_keys]
deepseek_vals_bar = [deepseek_scores.get(k, 0) for k in key_keys]
oracle_vals_bar   = [oracle_scores.get(k, 0)   for k in key_keys]

ax = axes[0]
ax.bar([i - width for i in x], bm25_vals_bar,     width, label="BM25",     color="#5B9BD5")
ax.bar([i         for i in x], deepseek_vals_bar, width, label="DeepSeek", color="#ED7D31")
ax.bar([i + width for i in x], oracle_vals_bar,   width, label="Oracle",   color="#70AD47")
ax.set_xticks(list(x))
ax.set_xticklabels(key_metrics, rotation=20, ha="right")
ax.set_ylabel("Score")
ax.set_title("Key Metrics: BM25 vs DeepSeek vs Oracle")
ax.legend()
ax.set_ylim(0, 1.05)

# 2. Per-query nDCG@10 gain over BM25 (oracle)
ax2 = axes[1]
sorted_gains = sorted(gains_oracle)
ax2.bar(range(len(sorted_gains)), sorted_gains,
        color=["#70AD47" if g >= 0 else "#FF5050" for g in sorted_gains],
        width=1.0, linewidth=0)
ax2.axhline(0, color="grey", linewidth=0.8)
ax2.set_xlabel("Queries (sorted by gain)")
ax2.set_ylabel("nDCG@10 gain over BM25")
ax2.set_title("Per-Query Oracle Gain over BM25")

# 3. Scatter: BM25 nDCG@10 vs Oracle nDCG@10
ax3 = axes[2]
ax3.scatter(bm25_vals, oracle_vals, alpha=0.3, s=10, color="#70AD47", label="Oracle")
ax3.scatter(bm25_vals, deepseek_vals, alpha=0.3, s=10, color="#ED7D31", label="DeepSeek")
lims = [0, 1.05]
ax3.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
ax3.set_xlabel("BM25 nDCG@10")
ax3.set_ylabel("Reranker nDCG@10")
ax3.set_title("BM25 vs Reranker (per query nDCG@10)")
ax3.legend()

plt.tight_layout()
plt.savefig(OUT_IMG, dpi=150)
print(f"\nPlot saved → {OUT_IMG}")
plt.close()
