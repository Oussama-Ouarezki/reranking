"""Entropy-driven BM25 + monoT5 linear fusion with fine alpha sweep, fixed tau.

Pipeline (per query)
--------------------
1. Take BM25 top-50, score each doc with monoT5 → cached softmax prob p_true.
2. Compute H@20 entropy from monoT5: take top-20 probs, renormalise via softmax
   (over their logit margins) → distribution → normalised entropy.
3. If H ≤ tau → confident → use pure monoT5 scores.
4. Else, sweep alpha from 0.995 down to 0.000 (step 0.005). For each alpha:
       fused = alpha * monot5_norm + (1-alpha) * bm25_norm
       (always the ORIGINAL monoT5 signal — never cumulative).
   Recompute H@20 entropy on fused. Stop at the first alpha that yields H ≤ tau.
5. If no alpha satisfies tau → fall back to pure monoT5.

Normalisation (scores are fused as probability distributions over the 50 docs,
so entropy stays well-defined at every alpha):
  - p_t5 = softmax(monoT5 margin logits) across the 50 docs.
  - p_bm = minmax(bm25) then renormalised to sum to 1.
  - fused distribution = alpha * p_t5 + (1 - alpha) * p_bm.
  - Entropy: take top-20 of the current distribution, renormalise, compute
    normalised Shannon entropy.

For now, tau is fixed at 0.8 (will be tuned later).

Reads:  qwen4b_uncertainty/data/monot5_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/34_entropy_adaptive_alpha_fixed_tau.tsv
        qwen4b_uncertainty/data/34_entropy_adaptive_alpha_fixed_tau.json
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

import ir_measures
from ir_measures import nDCG, Qrel, ScoredDoc

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE     = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/monot5_scores_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV  = BASE / "qwen4b_uncertainty/data/34_entropy_adaptive_alpha_fixed_tau.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/34_entropy_adaptive_alpha_fixed_tau.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES        = ["summary", "factoid", "list", "yesno"]
TAU          = 0.8
ALPHAS       = np.round(np.arange(0.995, -0.0001, -0.005), 4)
ENTROPY_K    = 20

METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]


# ── helpers ───────────────────────────────────────────────────────────────────

def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def topk_entropy_dist(dist: np.ndarray, k: int = ENTROPY_K) -> float:
    """Normalised Shannon entropy of the top-k mass of a distribution."""
    top = np.sort(dist)[::-1][:k]
    if len(top) < 2 or top.sum() <= 0:
        return 0.0
    p = top / top.sum()
    p = np.clip(p, 1e-15, 1.0)
    h = -float((p * np.log(p)).sum())
    return h / math.log(len(p))


def prob_to_margin(prob: np.ndarray) -> np.ndarray:
    """logit_true − logit_false = log(p / (1−p))."""
    p = np.clip(prob, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def to_distribution(x: np.ndarray) -> np.ndarray:
    """Renormalise a non-negative vector to sum to 1; uniform if all zero."""
    s = x.sum()
    if s <= 1e-12:
        return np.full_like(x, 1.0 / len(x))
    return x / s


def evaluate(run, qrels, qid_set):
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q   = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


# ── load data ─────────────────────────────────────────────────────────────────

rows = []
with SCORES_F.open() as f:
    for line in f:
        rows.append(json.loads(line))
print(f"{len(rows)} queries loaded")

all_qrels: list[Qrel] = []
with QRELS_F.open() as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            try:
                all_qrels.append(Qrel(p[0], p[1], int(p[2])))
            except ValueError:
                pass
print(f"{len(all_qrels)} qrel rows")

# Per-query arrays
margin_map:  dict[str, np.ndarray] = {}
p_t5_map:    dict[str, np.ndarray] = {}   # softmax distribution over 50 docs
p_bm_map:    dict[str, np.ndarray] = {}   # minmax→normalised distribution
docids_map:  dict[str, list[str]]  = {}
qtypes:      dict[str, str]        = {}

for r in rows:
    qid   = r["qid"]
    items = r["scores"]
    qtypes[qid] = r["type"]
    prob = np.array([s["monot5_prob"] for s in items], dtype=float)
    bm25 = np.array([s["bm25_score"]  for s in items], dtype=float)
    margins = prob_to_margin(prob)
    margin_map[qid] = margins
    p_t5_map[qid]   = softmax(margins)
    p_bm_map[qid]   = to_distribution(minmax(bm25))
    docids_map[qid] = [s["docid"] for s in items]

qids = list(p_t5_map.keys())
type_qids: dict[str, set] = defaultdict(set)
for qid, t in qtypes.items():
    type_qids[t].add(qid)
scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]


# ── adaptive-alpha resolution per query ───────────────────────────────────────

def resolve(qid: str, tau: float):
    """Return (final_score_vector, decision_tag, chosen_alpha)."""
    p_t5 = p_t5_map[qid]
    h0   = topk_entropy_dist(p_t5)
    if h0 <= tau:
        return margin_map[qid].copy(), "confident", None

    p_bm = p_bm_map[qid]
    for alpha in ALPHAS:
        # Original monoT5 distribution every iteration — never cumulative.
        fused = alpha * p_t5 + (1.0 - alpha) * p_bm
        if topk_entropy_dist(fused) <= tau:
            return fused, "fused", float(alpha)
    return margin_map[qid].copy(), "fallback", None


# ── baseline ──────────────────────────────────────────────────────────────────

baseline_run = [
    ScoredDoc(qid, docids_map[qid][i], float(margin_map[qid][i]))
    for qid in qids
    for i in range(len(docids_map[qid]))
]
baseline = {scope: evaluate(baseline_run, all_qrels, qset) for scope, qset in scopes}
print("\nBaseline (pure monoT5):")
for scope, met in baseline.items():
    print(f"  {scope:<10} ndcg@10={met['ndcg@10']:.4f}")


# ── run with fixed tau ────────────────────────────────────────────────────────

print(f"\nResolving {len(qids)} queries at tau={TAU} …")

run: list[ScoredDoc] = []
n_confident = n_fused = n_fallback = 0
alpha_hist = []
per_query_decisions = []

for qid in tqdm(qids):
    scores, tag, alpha = resolve(qid, TAU)
    if tag == "confident":
        n_confident += 1
    elif tag == "fused":
        n_fused += 1
        alpha_hist.append(alpha)
    else:
        n_fallback += 1
    per_query_decisions.append({"qid": qid, "type": qtypes[qid],
                                "decision": tag, "alpha": alpha})
    for i, docid in enumerate(docids_map[qid]):
        run.append(ScoredDoc(qid, docid, float(scores[i])))

metrics_by_scope = {scope: evaluate(run, all_qrels, qset) for scope, qset in scopes}

# ── results table ─────────────────────────────────────────────────────────────

print("\n" + "="*78)
print(f"FIXED tau = {TAU}  |  alpha sweep {ALPHAS[0]:.3f} → {ALPHAS[-1]:.3f} step 0.005")
print("="*78)
print(f"  confident: {n_confident:4d}   fused: {n_fused:4d}   fallback: {n_fallback:4d}")
if alpha_hist:
    arr = np.array(alpha_hist)
    print(f"  alpha used — mean={arr.mean():.3f} median={np.median(arr):.3f} "
          f"min={arr.min():.3f} max={arr.max():.3f}")

print(f"\n{'Scope':<10} {'nDCG@1':>8} {'nDCG@3':>8} {'nDCG@5':>8} {'nDCG@10':>9}  "
      f"{'Δ@10':>8}")
print("-"*78)
for scope, _ in scopes:
    m  = metrics_by_scope[scope]
    bl = baseline[scope]
    d  = m["ndcg@10"] - bl["ndcg@10"]
    print(f"{scope:<10} {m['ndcg@1']:>8.4f} {m['ndcg@3']:>8.4f} "
          f"{m['ndcg@5']:>8.4f} {m['ndcg@10']:>9.4f}  {d:>+8.4f}")

print("\nBaseline ndcg@10:")
for scope, _ in scopes:
    print(f"  {scope:<10} {baseline[scope]['ndcg@10']:.4f}")


# ── write outputs ─────────────────────────────────────────────────────────────

with OUT_TSV.open("w") as f:
    f.write("qid\ttype\tdecision\talpha\n")
    for d in per_query_decisions:
        f.write(f"{d['qid']}\t{d['type']}\t{d['decision']}\t"
                f"{'' if d['alpha'] is None else d['alpha']}\n")
print(f"\nPer-query decisions → {OUT_TSV}")

summary = {
    "tau": TAU,
    "alpha_grid": [float(a) for a in ALPHAS],
    "counts": {"confident": n_confident, "fused": n_fused, "fallback": n_fallback},
    "alpha_stats": (
        {"mean": float(np.mean(alpha_hist)),
         "median": float(np.median(alpha_hist)),
         "min": float(np.min(alpha_hist)),
         "max": float(np.max(alpha_hist))}
        if alpha_hist else None
    ),
    "metrics": metrics_by_scope,
    "baseline": baseline,
    "delta_ndcg10": {s: metrics_by_scope[s]["ndcg@10"] - baseline[s]["ndcg@10"]
                     for s, _ in scopes},
}
with OUT_JSON.open("w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary → {OUT_JSON}")


# ── plot: alpha histogram ─────────────────────────────────────────────────────

if alpha_hist:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(alpha_hist, bins=40, edgecolor="black")
    ax.set_xlabel("alpha (chosen)")
    ax.set_ylabel("# queries")
    ax.set_title(f"Alpha chosen by entropy-adaptive fusion (tau={TAU})")
    plt.tight_layout()
    out_png = PLOTS / "34_entropy_adaptive_alpha_hist.png"
    plt.savefig(out_png, dpi=150)
    print(f"Plot → {out_png}")
