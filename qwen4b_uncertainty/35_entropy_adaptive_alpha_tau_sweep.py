"""Grid-search the entropy threshold tau for the entropy-adaptive alpha pipeline.

Same pipeline as 34, but sweeps tau over a fine grid and records nDCG@1/3/5/10
per scope (global + per-question-type). Picks tau* per scope using each
scope's preferred target metric.

Reads:  qwen4b_uncertainty/data/monot5_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/35_tau_sweep.tsv
        qwen4b_uncertainty/data/35_tau_sweep_best.json
        qwen4b_uncertainty/plots/35_tau_sweep.png
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import ir_measures
from ir_measures import nDCG, Qrel, ScoredDoc

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE     = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/monot5_scores_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV  = BASE / "qwen4b_uncertainty/data/35_tau_sweep.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/35_tau_sweep_best.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES        = ["summary", "factoid", "list", "yesno"]
TAUS         = np.round(np.arange(0.0, 1.0001, 0.01), 4)
ALPHAS       = np.round(np.arange(0.995, -0.0001, -0.005), 4)
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

def minmax(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def softmax(x):
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def topk_entropy_dist(dist, k=ENTROPY_K):
    top = np.sort(dist)[::-1][:k]
    if len(top) < 2 or top.sum() <= 0:
        return 0.0
    p = top / top.sum()
    p = np.clip(p, 1e-15, 1.0)
    h = -float((p * np.log(p)).sum())
    return h / math.log(len(p))


def prob_to_margin(prob):
    p = np.clip(prob, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def to_distribution(x):
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

rows = [json.loads(l) for l in SCORES_F.open()]
print(f"{len(rows)} queries loaded")

all_qrels = []
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

margin_map, p_t5_map, p_bm_map, docids_map, qtypes = {}, {}, {}, {}, {}

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
type_qids = defaultdict(set)
for qid, t in qtypes.items():
    type_qids[t].add(qid)
scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]


# Pre-compute monoT5 baseline entropy per query (independent of tau)
h0_map = {qid: topk_entropy_dist(p_t5_map[qid]) for qid in qids}

# Pre-compute, for each query, the curve (alpha → fused entropy) so the
# tau sweep is a fast lookup instead of recomputing α entropies 100×.
fused_entropy = {}   # qid → np.ndarray of len(ALPHAS) with H@20 of fused dist
fused_dists   = {}   # qid → np.ndarray (len(ALPHAS), 50) with full fused dist
for qid in qids:
    p_t5 = p_t5_map[qid]
    p_bm = p_bm_map[qid]
    # broadcast: shape (len(ALPHAS), 50)
    fused = ALPHAS[:, None] * p_t5[None, :] + (1.0 - ALPHAS[:, None]) * p_bm[None, :]
    fused_dists[qid]   = fused
    fused_entropy[qid] = np.array([topk_entropy_dist(fused[i]) for i in range(len(ALPHAS))])


# ── tau sweep ─────────────────────────────────────────────────────────────────

baseline_run = [
    ScoredDoc(qid, docids_map[qid][i], float(margin_map[qid][i]))
    for qid in qids
    for i in range(len(docids_map[qid]))
]
baseline = {scope: evaluate(baseline_run, all_qrels, qset) for scope, qset in scopes}
print("\nBaseline (pure monoT5):")
for scope, met in baseline.items():
    print(f"  {scope:<10} ndcg@10={met['ndcg@10']:.4f}")

tsv_rows = []
best = {scope: {} for scope, _ in scopes}

print(f"\nSweeping {len(TAUS)} tau values …")
for tau in tqdm(TAUS, desc="tau"):
    run = []
    n_conf = n_fused = n_fb = 0
    alphas_used = []

    for qid in qids:
        if h0_map[qid] <= tau:
            scores = margin_map[qid]
            n_conf += 1
        else:
            ent = fused_entropy[qid]
            ok  = np.where(ent <= tau)[0]
            if ok.size > 0:
                idx = ok[0]                 # first alpha in 0.995→0 order
                scores = fused_dists[qid][idx]
                alphas_used.append(float(ALPHAS[idx]))
                n_fused += 1
            else:
                scores = margin_map[qid]
                n_fb += 1
        for i, docid in enumerate(docids_map[qid]):
            run.append(ScoredDoc(qid, docid, float(scores[i])))

    by_scope = {scope: evaluate(run, all_qrels, qset) for scope, qset in scopes}
    alpha_mean = float(np.mean(alphas_used)) if alphas_used else float("nan")

    for scope, met in by_scope.items():
        target = TYPE_TARGETS[scope]
        row = {"tau": float(tau), "scope": scope,
               "n_confident": n_conf, "n_fused": n_fused, "n_fallback": n_fb,
               "alpha_mean": alpha_mean, **met}
        tsv_rows.append(row)
        if not best[scope] or met[target] > best[scope].get(target, -1):
            best[scope] = {"tau": float(tau), **met,
                           "n_confident": n_conf, "n_fused": n_fused,
                           "n_fallback": n_fb, "alpha_mean": alpha_mean}


# ── write TSV / JSON ──────────────────────────────────────────────────────────

header = ["tau", "scope", "n_confident", "n_fused", "n_fallback",
          "alpha_mean"] + METRIC_NAMES
with OUT_TSV.open("w") as f:
    f.write("\t".join(header) + "\n")
    for r in tsv_rows:
        f.write("\t".join(str(r[k]) for k in header) + "\n")
print(f"\nTSV  → {OUT_TSV}")

with OUT_JSON.open("w") as f:
    json.dump({"best": best, "baseline": baseline}, f, indent=2)
print(f"JSON → {OUT_JSON}")


# ── results table ─────────────────────────────────────────────────────────────

print("\n" + "="*92)
print("BEST tau PER SCOPE")
print("="*92)
print(f"{'Scope':<10} {'tau*':>5} {'nDCG@1':>8} {'nDCG@3':>8} {'nDCG@5':>8} "
      f"{'nDCG@10':>9} {'conf':>5} {'fuse':>5} {'fb':>5} {'ᾱ':>6} {'Δtgt':>8}")
print("-"*92)
for scope, _ in scopes:
    b = best[scope]
    bl = baseline[scope]
    target = TYPE_TARGETS[scope]
    d = b[target] - bl[target]
    a = "—" if math.isnan(b["alpha_mean"]) else f"{b['alpha_mean']:.3f}"
    print(f"{scope:<10} {b['tau']:>5.2f} {b['ndcg@1']:>8.4f} {b['ndcg@3']:>8.4f} "
          f"{b['ndcg@5']:>8.4f} {b['ndcg@10']:>9.4f} "
          f"{b['n_confident']:>5d} {b['n_fused']:>5d} {b['n_fallback']:>5d} "
          f"{a:>6} {d:>+8.4f}")

print("\nBaseline ndcg@10:")
for scope, _ in scopes:
    print(f"  {scope:<10} {baseline[scope]['ndcg@10']:.4f}")


# ── plot ──────────────────────────────────────────────────────────────────────

df = pd.DataFrame(tsv_rows)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (scope, _) in zip(axes, scopes):
    sub = df[df["scope"] == scope]
    target = TYPE_TARGETS[scope]
    ax.plot(sub["tau"], sub[target], linewidth=2, label="entropy-adaptive fusion")
    ax.axhline(baseline[scope][target], color="gray", linestyle="--",
               linewidth=1.2, label="pure monoT5")
    b = best[scope]
    ax.axvline(b["tau"], color="red", linestyle=":", linewidth=1, alpha=0.8)
    ax.scatter([b["tau"]], [b[target]], color="red", zorder=5, s=50,
               label=f"τ*={b['tau']:.2f}")
    ax.set_title(f"{scope}  (target: {target})", fontsize=11)
    ax.set_xlabel("tau")
    ax.set_ylabel(target)
    ax.legend(fontsize=8)

for ax in axes[len(scopes):]:
    ax.set_visible(False)

fig.suptitle("Entropy-adaptive α + BM25 fusion — τ sweep (test set)", fontsize=13)
plt.tight_layout()
out_png = PLOTS / "35_tau_sweep.png"
plt.savefig(out_png, dpi=150)
print(f"Plot → {out_png}")
