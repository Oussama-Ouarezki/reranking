"""monoT5 entropy-driven BM25 fusion — corrected algorithm, test set.

Algorithm
---------
Step 1  Recover margin logit = logit_true − logit_false = log(p/(1−p))
        from the cached monot5_prob values.
        Apply softmax across the top-20 margin logits → probability distribution.
        Compute H@20 entropy.  If H ≤ tau → model is certain → use pure monoT5.

Step 2  If H > tau → uncertain → try linear fusion with BM25.
        For each alpha in [0.95, 0.90, …, 0.50]:
          fused = alpha * minmax(margins) + (1−alpha) * minmax(bm25)
          Apply softmax across top-20 fused scores → H@20_fused.
          If H@20_fused ≤ tau → use fused scores and stop.
        Each attempt always uses the original margin logits, never re-fuses.

Step 3  If no alpha satisfies tau → fall back to pure monoT5.

Key fix vs earlier scripts
  - Margin logits (unbounded real) instead of softmax probs as the score signal.
  - One softmax per entropy computation (cross-document, top-k).  No double-softmax.
  - Fallback is pure monoT5, not the minimum-alpha fused scores.
  - Same tau threshold for both Step 1 and Step 2.

Reads:  qwen4b_uncertainty/data/monot5_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/33_entropy_bm25_fusion_test.tsv
        qwen4b_uncertainty/data/33_entropy_bm25_fusion_best_test.json
        qwen4b_uncertainty/plots/33_entropy_bm25_fusion_test.png
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
OUT_TSV  = BASE / "qwen4b_uncertainty/data/33_entropy_bm25_fusion_test.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/33_entropy_bm25_fusion_best_test.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES        = ["summary", "factoid", "list", "yesno"]
TAUS         = np.round(np.linspace(0.0, 1.0, 51), 4)   # 0.00 → 1.00, step 0.02
ALPHAS       = np.round(np.arange(0.95, 0.45, -0.05), 2) # 0.95 → 0.50
ENTROPY_K    = 20

METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]

TYPE_TARGETS = {
    "global":  "ndcg@10",
    "summary": "ndcg@10",
    "factoid": "ndcg@5",
    "list":    "ndcg@3",
    "yesno":   "ndcg@1",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def softmax_entropy_topk(scores: np.ndarray, k: int = ENTROPY_K) -> float:
    """Normalised entropy of softmax over top-k scores (any real values)."""
    top = np.sort(scores)[::-1][:k]
    if len(top) < 2:
        return 0.0
    top = top - top.max()          # numerical stability
    exp_ = np.exp(top)
    probs = exp_ / exp_.sum()
    probs = np.clip(probs, 1e-15, 1.0)
    h = -float((probs * np.log(probs)).sum())
    return h / math.log(len(probs))


def recover_margin(prob: np.ndarray) -> np.ndarray:
    """logit_true − logit_false = log(p / (1−p)) from cached softmax prob."""
    p = np.clip(prob, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def resolve_query(margins: np.ndarray, bm25_norm: np.ndarray, tau: float) -> np.ndarray:
    """Apply the 3-step entropy algorithm; return final score vector."""
    # Step 1
    h0 = softmax_entropy_topk(margins)
    if h0 <= tau:
        return margins.copy()   # certain → pure monoT5

    # Step 2
    t5_norm = minmax(margins)
    for alpha in ALPHAS:
        fused  = alpha * t5_norm + (1.0 - alpha) * bm25_norm
        h_f    = softmax_entropy_topk(fused)
        if h_f <= tau:
            return fused        # first alpha that satisfies tau

    # Step 3 — fall back to pure monoT5
    return margins.copy()


def evaluate(run: list[ScoredDoc], qrels: list[Qrel], qid_set: set) -> dict:
    sub_run  = [r for r in run if r.query_id in qid_set]
    sub_q    = [q for q in qrels if q.query_id in qid_set]
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
                q = Qrel(p[0], p[1], int(p[2]))
                all_qrels.append(q)
            except ValueError:
                pass
print(f"{len(all_qrels)} qrel rows")

# Build per-query arrays
margins_map:  dict[str, np.ndarray] = {}
bm25_norm_map: dict[str, np.ndarray] = {}
docids_map:   dict[str, list[str]]  = {}
qtypes:       dict[str, str]        = {}

for r in rows:
    qid   = r["qid"]
    items = r["scores"]
    qtypes[qid] = r["type"]
    prob  = np.array([s["monot5_prob"] for s in items], dtype=float)
    bm25  = np.array([s["bm25_score"]  for s in items], dtype=float)
    margins_map[qid]   = recover_margin(prob)
    bm25_norm_map[qid] = minmax(bm25)
    docids_map[qid]    = [s["docid"] for s in items]

qids = list(margins_map.keys())
type_qids: dict[str, set] = defaultdict(set)
for qid, t in qtypes.items():
    type_qids[t].add(qid)
scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

# ── tau sweep ─────────────────────────────────────────────────────────────────

print(f"\nSweeping {len(TAUS)} tau values …")

# Baseline: pure monoT5 (tau=1.0 → always step 1 triggers since H ≤ 1.0 always)
baseline_run = [
    ScoredDoc(qid, docids_map[qid][i], float(margins_map[qid][i]))
    for qid in qids
    for i in range(len(docids_map[qid]))
]
baseline = {scope: evaluate(baseline_run, all_qrels, qset) for scope, qset in scopes}
print("\nBaseline (pure monoT5 via margin logits):")
for scope, met in baseline.items():
    print(f"  {scope:<10} ndcg@10={met['ndcg@10']:.4f}")

tsv_rows = []
best: dict[str, dict] = {scope: {} for scope, _ in scopes}

for tau in tqdm(TAUS, desc="tau"):
    # Build run for this tau
    run: list[ScoredDoc] = []
    n_fused = 0
    n_fallback = 0
    for qid in qids:
        margins  = margins_map[qid]
        bm25_norm = bm25_norm_map[qid]
        h0 = softmax_entropy_topk(margins)

        if h0 <= tau:
            scores = margins.copy()
        else:
            t5_norm = minmax(margins)
            found = False
            for alpha in ALPHAS:
                fused = alpha * t5_norm + (1.0 - alpha) * bm25_norm
                if softmax_entropy_topk(fused) <= tau:
                    scores = fused
                    found  = True
                    n_fused += 1
                    break
            if not found:
                scores = margins.copy()
                n_fallback += 1

        for i, docid in enumerate(docids_map[qid]):
            run.append(ScoredDoc(qid, docid, float(scores[i])))

    metrics_by_scope = {scope: evaluate(run, all_qrels, qset) for scope, qset in scopes}

    for scope, met in metrics_by_scope.items():
        target = TYPE_TARGETS[scope]
        row = {
            "tau": float(tau),
            "scope": scope,
            "n_fused": n_fused,
            "n_fallback": n_fallback,
            **met,
        }
        tsv_rows.append(row)

        if not best[scope] or met[target] > best[scope].get(target, -1):
            best[scope] = {"tau": float(tau), **met,
                           "n_fused": n_fused, "n_fallback": n_fallback}

# ── write outputs ─────────────────────────────────────────────────────────────

header = ["tau", "scope", "n_fused", "n_fallback"] + METRIC_NAMES
with OUT_TSV.open("w") as f:
    f.write("\t".join(header) + "\n")
    for row in tsv_rows:
        f.write("\t".join(str(row[k]) for k in header) + "\n")
print(f"\nTSV → {OUT_TSV}")

with OUT_JSON.open("w") as f:
    json.dump(best, f, indent=2)
print(f"JSON → {OUT_JSON}")

# ── results table ─────────────────────────────────────────────────────────────

print("\n" + "="*72)
print("BEST RESULTS PER SCOPE")
print("="*72)
print(f"{'Scope':<10} {'tau*':>6} {'nDCG@1':>8} {'nDCG@3':>8} {'nDCG@5':>8} "
      f"{'nDCG@10':>8} {'fused':>7} {'fallbk':>7}")
print("-"*72)
for scope, _ in scopes:
    b = best[scope]
    bl = baseline[scope]
    target = TYPE_TARGETS[scope]
    delta  = b[target] - bl[target]
    print(f"{scope:<10} {b['tau']:>6.2f} {b['ndcg@1']:>8.4f} {b['ndcg@3']:>8.4f} "
          f"{b['ndcg@5']:>8.4f} {b['ndcg@10']:>8.4f} "
          f"{b['n_fused']:>7d} {b['n_fallback']:>7d}  "
          f"Δ{target}={delta:+.4f}")

print("\nBaseline (pure monoT5):")
for scope, _ in scopes:
    bl = baseline[scope]
    print(f"  {scope:<10} ndcg@10={bl['ndcg@10']:.4f}")

# ── plot ──────────────────────────────────────────────────────────────────────

import pandas as pd
df = pd.DataFrame(tsv_rows)

fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
axes = axes.flatten()

for ax, (scope, _) in zip(axes, scopes):
    sub = df[df["scope"] == scope]
    target = TYPE_TARGETS[scope]

    ax.plot(sub["tau"], sub[target], linewidth=2, label="entropy-BM25 fusion")
    ax.axhline(baseline[scope][target], color="gray", linestyle="--",
               linewidth=1.2, label="pure monoT5")
    b = best[scope]
    ax.axvline(b["tau"], color="red", linestyle=":", linewidth=1, alpha=0.8)
    ax.scatter([b["tau"]], [b[target]], color="red", zorder=5, s=50)

    ax.set_title(f"{scope}  (target: {target})", fontsize=11)
    ax.set_xlabel("tau")
    ax.set_ylabel(target)
    ax.legend(fontsize=8)

# hide unused subplot
for ax in axes[len(scopes):]:
    ax.set_visible(False)

fig.suptitle("monoT5 Entropy-Driven BM25 Fusion — tau sweep (test set)", fontsize=13)
plt.tight_layout()
out_png = PLOTS / "33_entropy_bm25_fusion_test.png"
plt.savefig(out_png, dpi=150)
print(f"\nPlot → {out_png}")
plt.show()
