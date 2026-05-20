"""Oracle per-query alpha for monoT5 + BM25 linear fusion (BioASQ train set).

For each query, sweep alpha and pick the alpha that maximises nDCG@10 of the
linear-fused score, evaluated against gold qrels. Then aggregate alpha*
mean / std globally and per question type, and bar-plot the per-type means.

Score form (matches scripts 34/35):
    p_t5 = softmax(monoT5 margin logits)            # distribution over 50 docs
    p_bm = minmax(bm25) renormalised to sum to 1    # distribution over 50 docs
    fused(alpha) = alpha * p_t5 + (1 - alpha) * p_bm

Reads:  qwen4b_uncertainty/data/monot5_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/37_oracle_per_query_alpha_monot5.tsv
        qwen4b_uncertainty/data/37_oracle_per_query_alpha_monot5_summary.json
        qwen4b_uncertainty/plots/37_oracle_per_query_alpha_monot5.png
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
SCORES_F = BASE / "qwen4b_uncertainty/data/monot5_scores_2000.jsonl"
QRELS_F  = BASE / "data/bioasq/processed/qrels.tsv"
OUT_TSV  = BASE / "qwen4b_uncertainty/data/37_oracle_per_query_alpha_monot5.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/37_oracle_per_query_alpha_monot5_summary.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES   = ["summary", "factoid", "list", "yesno"]
ALPHAS  = np.round(np.arange(0.0, 1.0001, 0.005), 4)
TARGET  = nDCG @ 10
TARGET_NAME = "ndcg@10"


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


def prob_to_margin(prob):
    p = np.clip(prob, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def to_distribution(x):
    s = x.sum()
    if s <= 1e-12:
        return np.full_like(x, 1.0 / len(x))
    return x / s


def ndcg10(scores, docids, qid, qrels_for_qid):
    run = [ScoredDoc(qid, d, float(s)) for d, s in zip(docids, scores)]
    res = ir_measures.calc_aggregate([TARGET], qrels_for_qid, run)
    return float(res[TARGET])


# ── load data ─────────────────────────────────────────────────────────────────

if not SCORES_F.exists():
    raise SystemExit(f"missing {SCORES_F} — run 36_cache_monot5_train.py first")

rows = [json.loads(l) for l in SCORES_F.open()]
print(f"{len(rows)} queries loaded from {SCORES_F.name}")

qrels_by_qid = defaultdict(list)
with QRELS_F.open() as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3:
            try:
                qrels_by_qid[p[0]].append(Qrel(p[0], p[1], int(p[2])))
            except ValueError:
                pass
total_qrels = sum(len(v) for v in qrels_by_qid.values())
print(f"{total_qrels} qrel rows for {len(qrels_by_qid)} qids")


# ── per-query sweep ───────────────────────────────────────────────────────────

records = []
skipped_no_qrels = 0
skipped_no_relevant_in_top50 = 0

for r in tqdm(rows, desc="queries"):
    qid    = r["qid"]
    qtype  = r["type"]
    items  = r["scores"]
    qrels  = qrels_by_qid.get(qid, [])
    if not qrels:
        skipped_no_qrels += 1
        continue

    docids = [s["docid"] for s in items]
    prob   = np.array([s["monot5_prob"] for s in items], dtype=float)
    bm25   = np.array([s["bm25_score"]  for s in items], dtype=float)

    p_t5 = softmax(prob_to_margin(prob))
    p_bm = to_distribution(minmax(bm25))

    # Skip queries where no relevant doc is recallable in BM25 top-50:
    # nDCG@10 is then 0 for every alpha and the oracle is undefined.
    rel_ids = {q.doc_id for q in qrels if q.relevance > 0}
    if not (rel_ids & set(docids)):
        skipped_no_relevant_in_top50 += 1
        continue

    monot5_only_ndcg = ndcg10(p_t5, docids, qid, qrels)
    bm25_only_ndcg   = ndcg10(p_bm, docids, qid, qrels)

    best_alpha = None
    best_ndcg  = -1.0
    curve = []
    for a in ALPHAS:
        fused = a * p_t5 + (1.0 - a) * p_bm
        n = ndcg10(fused, docids, qid, qrels)
        curve.append(n)
        if n > best_ndcg:
            best_ndcg  = n
            best_alpha = float(a)

    records.append({
        "qid": qid,
        "type": qtype,
        "alpha_star": best_alpha,
        "ndcg10_star": best_ndcg,
        "ndcg10_monot5": monot5_only_ndcg,
        "ndcg10_bm25":   bm25_only_ndcg,
        "gain_over_monot5": best_ndcg - monot5_only_ndcg,
    })

print(f"\n{len(records)} queries with oracle alpha "
      f"(skipped: {skipped_no_qrels} no-qrels, "
      f"{skipped_no_relevant_in_top50} no-relevant-in-top50)")


# ── aggregates ────────────────────────────────────────────────────────────────

def stats(values):
    arr = np.array(values, dtype=float)
    return {"n": int(len(arr)),
            "mean": float(arr.mean()),
            "std":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max())}


by_type = defaultdict(list)
for r in records:
    by_type[r["type"]].append(r["alpha_star"])

global_alphas = [r["alpha_star"] for r in records]
agg = {"global": stats(global_alphas)}
for t in TYPES:
    if by_type[t]:
        agg[t] = stats(by_type[t])

# Mean nDCG gain summary too
def gain_stats(records, key="gain_over_monot5"):
    return stats([r[key] for r in records])

gains = {
    "global": gain_stats(records),
    **{t: gain_stats([r for r in records if r["type"] == t]) for t in TYPES
       if any(r["type"] == t for r in records)}
}


# ── print ─────────────────────────────────────────────────────────────────────

print("\n" + "="*78)
print("Oracle per-query α* — monoT5 + BM25 linear fusion (target: nDCG@10)")
print("="*78)
print(f"{'scope':<10} {'n':>4} {'mean α*':>9} {'std':>7} {'median':>8} "
      f"{'min':>6} {'max':>6}   {'gain mean':>10} {'gain std':>9}")
print("-"*78)
for scope in ["global"] + TYPES:
    if scope not in agg:
        continue
    a = agg[scope]
    g = gains[scope]
    print(f"{scope:<10} {a['n']:>4} {a['mean']:>9.4f} {a['std']:>7.4f} "
          f"{a['median']:>8.4f} {a['min']:>6.3f} {a['max']:>6.3f}   "
          f"{g['mean']:>+10.4f} {g['std']:>9.4f}")


# ── write outputs ─────────────────────────────────────────────────────────────

with OUT_TSV.open("w") as f:
    f.write("qid\ttype\talpha_star\tndcg10_star\tndcg10_monot5\tndcg10_bm25\t"
            "gain_over_monot5\n")
    for r in records:
        f.write(f"{r['qid']}\t{r['type']}\t{r['alpha_star']:.4f}\t"
                f"{r['ndcg10_star']:.6f}\t{r['ndcg10_monot5']:.6f}\t"
                f"{r['ndcg10_bm25']:.6f}\t{r['gain_over_monot5']:.6f}\n")
print(f"\nPer-query → {OUT_TSV}")

with OUT_JSON.open("w") as f:
    json.dump({"alpha_stats": agg, "gain_stats": gains,
               "n_skipped_no_qrels": skipped_no_qrels,
               "n_skipped_no_relevant_in_top50": skipped_no_relevant_in_top50},
              f, indent=2)
print(f"Summary  → {OUT_JSON}")


# ── bar plot ──────────────────────────────────────────────────────────────────

scopes_plot = ["global"] + [t for t in TYPES if t in agg]
means = [agg[s]["mean"] for s in scopes_plot]
stds  = [agg[s]["std"]  for s in scopes_plot]
ns    = [agg[s]["n"]    for s in scopes_plot]

fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(len(scopes_plot))
bars = ax.bar(xs, means, yerr=stds, capsize=6,
              color=["#444"] + sns.color_palette("deep", len(scopes_plot)-1),
              edgecolor="black", linewidth=0.6)
ax.set_xticks(xs)
ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(scopes_plot, ns)])
ax.set_ylabel("oracle α*  (1 = pure monoT5, 0 = pure BM25)")
ax.set_ylim(0.0, 1.05)
ax.set_title("Oracle per-query α* — monoT5 + BM25 linear fusion (BioASQ train)\n"
             "bars = mean across queries, error bars = std")
for b, m, s in zip(bars, means, stds):
    ax.text(b.get_x() + b.get_width()/2, m + s + 0.02,
            f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
plt.tight_layout()
out_png = PLOTS / "37_oracle_per_query_alpha_monot5.png"
plt.savefig(out_png, dpi=150)
print(f"Plot     → {out_png}")
