"""Per-type α grid search on train (α in [0.80, 1.00]) — apply on test.

Step 1 (train, 500 queries): for each question type, sweep α in [0.80, 1.00]
   step 0.005 and pick the single α that maximises aggregate nDCG@10 across
   queries of that type. Same fusion as 34/35/37/38:
       fused = α * softmax(monoT5 margins) + (1-α) * minmax(bm25)→sum-to-1
   Also pick a global α* (best aggregate over all 500 train queries).

Step 2 (test, 340 queries): apply the per-type α* (each query uses α for its
   type) and the global α*; report nDCG@1/3/5/10 per scope and Δ vs pure
   monoT5.

Reads:  qwen4b_uncertainty/data/monot5_scores.jsonl              (train)
        data/bioasq/processed/qrels.tsv                          (train qrels)
        qwen4b_uncertainty/data/monot5_scores_test.jsonl         (test)
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv          (test qrels)
Writes: qwen4b_uncertainty/data/40_alpha_grid_narrow_train.tsv
        qwen4b_uncertainty/data/40_alpha_narrow_test_results.tsv
        qwen4b_uncertainty/data/40_alpha_narrow_summary.json
        qwen4b_uncertainty/plots/40_alpha_narrow_train_curves.png
        qwen4b_uncertainty/plots/40_alpha_narrow_test_bars.png
"""

import json
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

BASE = Path(__file__).resolve().parents[1]

# TRAIN_SCORES = BASE / "qwen4b_uncertainty/data/monot5_scores_2000.jsonl"
# TRAIN_QRELS  = BASE / "data/bioasq/processed/qrels.tsv"

# TEST_SCORES  = BASE / "qwen4b_uncertainty/data/monot5_scores_test.jsonl"
# TEST_QRELS   = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"

TRAIN_SCORES = BASE / "qwen4b_uncertainty/data/qwen_scores.jsonl"
TRAIN_QRELS  = BASE / "data/bioasq/processed/qrels.tsv"

TEST_SCORES  = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
TEST_QRELS   = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"

OUT_TRAIN_TSV = BASE / "qwen4b_uncertainty/data/40_alpha_grid_narrow_train.tsv"
OUT_TEST_TSV  = BASE / "qwen4b_uncertainty/data/40_alpha_narrow_test_results.tsv"
OUT_JSON      = BASE / "qwen4b_uncertainty/data/40_alpha_narrow_summary.json"
PLOTS         = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES        = ["summary", "factoid", "list", "yesno"]
ALPHAS       = np.round(np.arange(0.99, 1.0001, 0.005), 4)
TARGET       = nDCG @ 10
TARGET_NAME  = "ndcg@10"
METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]


# ── helpers ───────────────────────────────────────────────────────────────────

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


def load_split(scores_path, qrels_path):
    rows = [json.loads(l) for l in scores_path.open()]
    qrels = []
    with qrels_path.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                try:
                    qrels.append(Qrel(p[0], p[1], int(p[2])))
                except ValueError:
                    pass

    p_t5, p_bm, margins, docids, qtypes = {}, {}, {}, {}, {}
    for r in rows:
        qid = r["qid"]; items = r["scores"]; qtypes[qid] = r["type"]
        prob = np.array([s["monot5_prob"] for s in items], dtype=float)
        bm   = np.array([s["bm25_score"]  for s in items], dtype=float)
        m    = prob_to_margin(prob)
        margins[qid] = m
        p_t5[qid]    = softmax(m)
        p_bm[qid]    = to_distribution(minmax(bm))
        docids[qid]  = [s["docid"] for s in items]
    qids = list(p_t5.keys())
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    return {"qids": qids, "qtypes": qtypes, "type_qids": type_qids,
            "p_t5": p_t5, "p_bm": p_bm, "margins": margins,
            "docids": docids, "qrels": qrels}


def fused_run(split, alpha_for_qid):
    run = []
    for qid in split["qids"]:
        a = alpha_for_qid(qid)
        s = a * split["p_t5"][qid] + (1.0 - a) * split["p_bm"][qid]
        for i, d in enumerate(split["docids"][qid]):
            run.append(ScoredDoc(qid, d, float(s[i])))
    return run

def monot5_baseline_run(split):
    run = []
    for qid in split["qids"]:
        for i, d in enumerate(split["docids"][qid]):
            run.append(ScoredDoc(qid, d, float(split["margins"][qid][i])))
    return run

def evaluate(run, qrels, qid_set):
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q   = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


# ── load both splits ──────────────────────────────────────────────────────────

print("Loading train …")
train = load_split(TRAIN_SCORES, TRAIN_QRELS)
print(f"  {len(train['qids'])} train queries")

print("Loading test …")
test  = load_split(TEST_SCORES, TEST_QRELS)
print(f"  {len(test['qids'])} test queries")


# ── train: per-type and global α grid sweep ───────────────────────────────────

print(f"\nTrain α grid sweep ({len(ALPHAS)} alphas in [{ALPHAS[0]}, {ALPHAS[-1]}]) …")

train_curves = {scope: {a: None for a in ALPHAS}
                for scope in ["global"] + TYPES}
train_qid_sets = {"global": set(train["qids"]),
                  **{t: train["type_qids"][t] for t in TYPES}}

for a in tqdm(ALPHAS, desc="alpha"):
    run = fused_run(train, lambda q, a=a: float(a))
    for scope, qset in train_qid_sets.items():
        m = evaluate(run, train["qrels"], qset)
        train_curves[scope][float(a)] = m

# Also baseline = pure monoT5 (margin logit ranking)
train_baseline_run = monot5_baseline_run(train)
train_baseline = {scope: evaluate(train_baseline_run, train["qrels"], qset)
                  for scope, qset in train_qid_sets.items()}

# Pick best α per scope (target ndcg@10)
best_alpha = {}
for scope in ["global"] + TYPES:
    best_a, best_v = None, -1.0
    for a, m in train_curves[scope].items():
        v = m[TARGET_NAME]
        if v > best_v:
            best_v, best_a = v, a
    best_alpha[scope] = {"alpha": best_a, **train_curves[scope][best_a]}

print("\nTrain best α per scope (target nDCG@10):")
for scope in ["global"] + TYPES:
    a = best_alpha[scope]
    bl = train_baseline[scope]
    print(f"  {scope:<10} α*={a['alpha']:.3f}  "
          f"ndcg@10={a['ndcg@10']:.4f}  "
          f"(monoT5 baseline {bl['ndcg@10']:.4f}, Δ={a['ndcg@10']-bl['ndcg@10']:+.4f})")


# ── train curves TSV ──────────────────────────────────────────────────────────

with OUT_TRAIN_TSV.open("w") as f:
    f.write("scope\talpha\t" + "\t".join(METRIC_NAMES) + "\n")
    for scope in ["global"] + TYPES:
        for a, m in train_curves[scope].items():
            f.write(f"{scope}\t{a:.4f}\t" +
                    "\t".join(f"{m[k]:.6f}" for k in METRIC_NAMES) + "\n")
print(f"\nTrain curves → {OUT_TRAIN_TSV}")


# ── apply best α* on test ─────────────────────────────────────────────────────

ALPHA_GLOBAL  = best_alpha["global"]["alpha"]
ALPHA_PER_TYPE = {t: best_alpha[t]["alpha"] for t in TYPES}

print(f"\nApplying α* on test:")
print(f"  global   = {ALPHA_GLOBAL:.3f}")
for t in TYPES:
    print(f"  {t:<8} = {ALPHA_PER_TYPE[t]:.3f}")

test_qid_sets = {"global": set(test["qids"]),
                 **{t: test["type_qids"][t] for t in TYPES}}

strategies = {
    "pure_monot5":      monot5_baseline_run(test),
    "pure_bm25":        fused_run(test, lambda q: 0.0),
    "alpha_global":     fused_run(test, lambda q: ALPHA_GLOBAL),
    "alpha_per_type":   fused_run(test,
                                  lambda q: ALPHA_PER_TYPE[test["qtypes"][q]]),
}

test_results = {name: {scope: evaluate(run, test["qrels"], qset)
                       for scope, qset in test_qid_sets.items()}
                for name, run in strategies.items()}


# ── print test ────────────────────────────────────────────────────────────────

print("\n" + "="*94)
print("TEST results — α* learned from train (α in [0.80, 1.00])")
print("="*94)
print(f"{'scope':<10} {'strategy':<20} " +
      " ".join(f"{m:>9}" for m in METRIC_NAMES))
print("-"*94)
for scope in ["global"] + TYPES:
    base = test_results["pure_monot5"][scope]
    for name, res in test_results.items():
        m = res[scope]
        delta = ("" if name == "pure_monot5"
                 else f"  Δ@10={m['ndcg@10']-base['ndcg@10']:+.4f}")
        print(f"{scope:<10} {name:<20} " +
              " ".join(f"{m[k]:>9.4f}" for k in METRIC_NAMES) + delta)
    print()


# ── outputs ───────────────────────────────────────────────────────────────────

with OUT_TEST_TSV.open("w") as f:
    f.write("scope\tstrategy\talpha\t" + "\t".join(METRIC_NAMES) + "\n")
    for scope in ["global"] + TYPES:
        for name, res in test_results.items():
            m = res[scope]
            if name == "alpha_global":
                a_str = f"{ALPHA_GLOBAL:.3f}"
            elif name == "alpha_per_type":
                a_str = f"{ALPHA_PER_TYPE.get(scope, ALPHA_GLOBAL):.3f}" \
                        if scope in TYPES else "per-type"
            else:
                a_str = "—"
            f.write(f"{scope}\t{name}\t{a_str}\t" +
                    "\t".join(f"{m[k]:.6f}" for k in METRIC_NAMES) + "\n")
print(f"Test results → {OUT_TEST_TSV}")

with OUT_JSON.open("w") as f:
    json.dump({
        "alpha_grid": [float(a) for a in ALPHAS],
        "alpha_global": ALPHA_GLOBAL,
        "alpha_per_type": ALPHA_PER_TYPE,
        "train_best": best_alpha,
        "train_baseline_monot5": train_baseline,
        "test_results": test_results,
    }, f, indent=2)
print(f"Summary      → {OUT_JSON}")


# ── plots ─────────────────────────────────────────────────────────────────────

# (a) train α-vs-nDCG@10 curves
fig, ax = plt.subplots(figsize=(9, 5.5))
palette = sns.color_palette("deep", len(TYPES) + 1)
for color, scope in zip(palette, ["global"] + TYPES):
    xs = list(train_curves[scope].keys())
    ys = [train_curves[scope][a][TARGET_NAME] for a in xs]
    ax.plot(xs, ys, marker="o", markersize=3, label=scope, color=color,
            linewidth=1.5)
    bl = train_baseline[scope][TARGET_NAME]
    ax.axhline(bl, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
    a_star = best_alpha[scope]["alpha"]
    ax.scatter([a_star], [best_alpha[scope][TARGET_NAME]],
               color=color, s=70, edgecolor="black", zorder=5)
ax.set_xlabel("α  (1 = pure monoT5)")
ax.set_ylabel("nDCG@10 (train)")
ax.set_title("Train α grid in [0.80, 1.00] — per-type curves\n"
             "(dashed = pure-monoT5 baseline; dot = best α)")
ax.legend(fontsize=9, loc="lower left")
plt.tight_layout()
out_png = PLOTS / "40_alpha_narrow_train_curves.png"
plt.savefig(out_png, dpi=150)
print(f"Train curves → {out_png}")

# (b) test ndcg@10 bar plot
scope_names = ["global"] + TYPES
strategy_order = ["pure_bm25", "pure_monot5", "alpha_global", "alpha_per_type"]
strategy_labels = {
    "pure_bm25":      "BM25 only",
    "pure_monot5":    "monoT5 only",
    "alpha_global":   f"α=global ({ALPHA_GLOBAL:.3f})",
    "alpha_per_type": "α per type (train)",
}
palette = sns.color_palette("deep", len(strategy_order))

fig, ax = plt.subplots(figsize=(11, 5.5))
xs = np.arange(len(scope_names))
width = 0.20
for i, name in enumerate(strategy_order):
    vals = [test_results[name][s]["ndcg@10"] for s in scope_names]
    offset = (i - (len(strategy_order)-1)/2) * width
    ax.bar(xs + offset, vals, width, label=strategy_labels[name],
           color=palette[i], edgecolor="black", linewidth=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(scope_names)
ax.set_ylabel("nDCG@10 (test)")
ax.set_title("Test nDCG@10 — α* trained on narrow grid [0.80, 1.00]")
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(0.0, 1.0)
plt.tight_layout()
out_png = PLOTS / "40_alpha_narrow_test_bars.png"
plt.savefig(out_png, dpi=150)
print(f"Test bars    → {out_png}")
