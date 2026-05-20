"""Same as 38 but uses MEDIAN α* per type from the train oracle (not mean).

The train oracle's α* distribution is heavily skewed (long tail near 1.0),
so the median is a more robust central tendency than the mean. This script
applies median-per-type α as a fixed-α-by-question-type policy on the test
set and compares to monoT5/BM25 baselines and the global-median policy.

Reads:  qwen4b_uncertainty/data/monot5_scores_test.jsonl
        qwen4b_uncertainty/data/37_oracle_per_query_alpha_monot5_summary.json
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/39_per_type_alpha_median_generalization_test.tsv
        qwen4b_uncertainty/data/39_per_type_alpha_median_generalization_test.json
        qwen4b_uncertainty/plots/39_per_type_alpha_median_generalization_test.png
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import ir_measures
from ir_measures import nDCG, Qrel, ScoredDoc

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE       = Path(__file__).resolve().parents[1]
SCORES_F   = BASE / "qwen4b_uncertainty/data/monot5_scores_test.jsonl"
TRAIN_JSON = BASE / "qwen4b_uncertainty/data/37_oracle_per_query_alpha_monot5_summary.json"
QRELS_F    = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV    = BASE / "qwen4b_uncertainty/data/39_per_type_alpha_median_generalization_test.tsv"
OUT_JSON   = BASE / "qwen4b_uncertainty/data/39_per_type_alpha_median_generalization_test.json"
PLOTS      = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES        = ["summary", "factoid", "list", "yesno"]
METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]


def minmax(x):
    lo, hi = x.min(), x.max()
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

def softmax(x):
    z = x - x.max(); e = np.exp(z); return e / e.sum()

def prob_to_margin(prob):
    p = np.clip(prob, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))

def to_distribution(x):
    s = x.sum()
    return np.full_like(x, 1.0 / len(x)) if s <= 1e-12 else x / s

def evaluate(run, qrels, qid_set):
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q   = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


# ── train α (median) ──────────────────────────────────────────────────────────

train_summary = json.loads(TRAIN_JSON.read_text())
alpha_stats   = train_summary["alpha_stats"]

ALPHA_GLOBAL  = alpha_stats["global"]["median"]
ALPHA_BY_TYPE = {t: alpha_stats[t]["median"] for t in TYPES if t in alpha_stats}

print("Train oracle α* (MEDIAN) — applied as fixed α on test:")
print(f"  global : {ALPHA_GLOBAL:.4f}")
for t, a in ALPHA_BY_TYPE.items():
    print(f"  {t:<8}: {a:.4f}")


# ── test data ─────────────────────────────────────────────────────────────────

rows = [json.loads(l) for l in SCORES_F.open()]
print(f"\n{len(rows)} test queries loaded")

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

p_t5_map, p_bm_map, margin_map, docids_map, qtypes = {}, {}, {}, {}, {}
for r in rows:
    qid = r["qid"]; items = r["scores"]; qtypes[qid] = r["type"]
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


def build_run(alpha_for_qid):
    run = []
    for qid in qids:
        a = alpha_for_qid(qid)
        scores = a * p_t5_map[qid] + (1.0 - a) * p_bm_map[qid]
        for i, docid in enumerate(docids_map[qid]):
            run.append(ScoredDoc(qid, docid, float(scores[i])))
    return run

def build_run_monot5_margins():
    run = []
    for qid in qids:
        for i, docid in enumerate(docids_map[qid]):
            run.append(ScoredDoc(qid, docid, float(margin_map[qid][i])))
    return run

strategies = {
    "pure_monot5":           build_run_monot5_margins(),
    "pure_bm25":             build_run(lambda q: 0.0),
    "train_median_global":   build_run(lambda q: ALPHA_GLOBAL),
    "train_median_per_type": build_run(
        lambda q: ALPHA_BY_TYPE.get(qtypes[q], ALPHA_GLOBAL)),
}

results = {name: {scope: evaluate(run, all_qrels, qset)
                  for scope, qset in scopes}
           for name, run in strategies.items()}


# ── print ─────────────────────────────────────────────────────────────────────

print("\n" + "="*94)
print("Test-set generalisation — fixed α from train oracle MEDIAN per type")
print("="*94)
print(f"{'scope':<10} {'strategy':<24} " + " ".join(f"{m:>9}" for m in METRIC_NAMES))
print("-"*94)
for scope, _ in scopes:
    base = results["pure_monot5"][scope]
    for name, res in results.items():
        m = res[scope]
        delta = "" if name == "pure_monot5" else f"  Δ@10={m['ndcg@10']-base['ndcg@10']:+.4f}"
        print(f"{scope:<10} {name:<24} " +
              " ".join(f"{m[k]:>9.4f}" for k in METRIC_NAMES) + delta)
    print()


# ── outputs ───────────────────────────────────────────────────────────────────

with OUT_TSV.open("w") as f:
    f.write("scope\tstrategy\t" + "\t".join(METRIC_NAMES) + "\n")
    for scope, _ in scopes:
        for name, res in results.items():
            m = res[scope]
            f.write(f"{scope}\t{name}\t" +
                    "\t".join(f"{m[k]:.6f}" for k in METRIC_NAMES) + "\n")
print(f"TSV  → {OUT_TSV}")

with OUT_JSON.open("w") as f:
    json.dump({
        "alpha_global_train_median": ALPHA_GLOBAL,
        "alpha_per_type_train_median": ALPHA_BY_TYPE,
        "results": results,
    }, f, indent=2)
print(f"JSON → {OUT_JSON}")


# ── bar plot ──────────────────────────────────────────────────────────────────

scope_names = ["global"] + TYPES
strategy_order = ["pure_bm25", "pure_monot5",
                  "train_median_global", "train_median_per_type"]
strategy_labels = {
    "pure_bm25":              "BM25 only",
    "pure_monot5":            "monoT5 only",
    "train_median_global":    f"α=median global ({ALPHA_GLOBAL:.3f})",
    "train_median_per_type":  "α median per type (train)",
}
palette = sns.color_palette("deep", len(strategy_order))

fig, ax = plt.subplots(figsize=(11, 5.5))
xs = np.arange(len(scope_names))
width = 0.20
for i, name in enumerate(strategy_order):
    vals = [results[name][s]["ndcg@10"] for s in scope_names]
    offset = (i - (len(strategy_order)-1)/2) * width
    ax.bar(xs + offset, vals, width, label=strategy_labels[name],
           color=palette[i], edgecolor="black", linewidth=0.5)

ax.set_xticks(xs)
ax.set_xticklabels(scope_names)
ax.set_ylabel("nDCG@10")
ax.set_title("Generalisation of train-oracle MEDIAN α to test set — nDCG@10 per scope")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0.0, 1.0)
plt.tight_layout()
out_png = PLOTS / "39_per_type_alpha_median_generalization_test.png"
plt.savefig(out_png, dpi=150)
print(f"Plot → {out_png}")
