"""Can we predict the 43 queries where BM25 actually beats LiT5?

For each query we compute:
  - per-query nDCG@10 deltas (BM25 − LiT5)  → win / loss / tie label
  - candidate routing signals derived from BM25 top-50 scores:
      H@20, H@50  (normalised entropy — high = uncertain)
      gap1_2     = (s1 − s2) / s1     (relative top1−top2 margin)
      gap1_5     = (s1 − s5) / s1
      top1       = s1                 (raw BM25 top score)
      mean_top10
  - check signal AUROC at separating BM25-wins from BM25-loses
  - simulate: route to BM25 only when signal triggers, LiT5 otherwise,
    sweep the threshold, report best nDCG@1/@5/@10 per signal
  - report oracle ceiling (per-query max of the two)
"""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

ROOT = Path(__file__).resolve().parent.parent
BM25_CACHE  = ROOT / "application/cache/runs/bm25/20260427T130448Z.json"
LIT5_CACHE  = ROOT / "application/cache/runs/lit5/20260429T115008Z.json"
SCORE_CACHE = Path(__file__).resolve().parent / "cache" / "bm25_top50_scores.json"

OUT_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = OUT_DIR / "results"
PLOTS_DIR   = OUT_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(p): return json.load(open(p))


def get_metric(entry, key, k):
    return float(entry.get("metrics", {}).get(key, {}).get(k, 0.0))


def has_metrics(entry):
    return "metrics" in entry


def norm_entropy(vals):
    if not vals or sum(vals) == 0:
        return None
    s = sum(vals)
    probs = [v / s for v in vals]
    h = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
    return h / math.log(len(probs))


# ── load ─────────────────────────────────────────────────────────────────────
bm25_run = load_json(BM25_CACHE)
lit5_run = load_json(LIT5_CACHE)
bm25_pq, lit5_pq = bm25_run["per_query"], lit5_run["per_query"]
bm25_scores = load_json(SCORE_CACHE)

qids_all = sorted(set(bm25_pq) & set(lit5_pq) & set(bm25_scores))
qids = [q for q in qids_all if has_metrics(bm25_pq[q]) and has_metrics(lit5_pq[q])]
print(f"qids with both metrics: {len(qids)} / {len(qids_all)}")


# ── per-query labels & features ──────────────────────────────────────────────
records = []
for q in qids:
    b10 = get_metric(bm25_pq[q], "ndcg_at", "10")
    l10 = get_metric(lit5_pq[q], "ndcg_at", "10")
    label = 1 if b10 > l10 else (-1 if b10 < l10 else 0)

    s = sorted(bm25_scores[q], reverse=True)
    s = [v for v in s if v > 0]
    if len(s) < 5:
        continue
    h20 = norm_entropy(s[:20]) or 0.0
    h50 = norm_entropy(s[:50]) or 0.0
    gap1_2 = (s[0] - s[1]) / s[0] if s[0] > 0 else 0.0
    gap1_5 = (s[0] - s[4]) / s[0] if s[0] > 0 else 0.0
    top1   = s[0]
    mean10 = float(np.mean(s[:10]))
    records.append({
        "qid": q, "label": label, "delta": b10 - l10,
        "bm25_n10": b10, "lit5_n10": l10,
        "h20": h20, "h50": h50,
        "gap1_2": gap1_2, "gap1_5": gap1_5,
        "top1": top1, "mean_top10": mean10,
    })

n_win  = sum(r["label"] ==  1 for r in records)
n_lose = sum(r["label"] == -1 for r in records)
n_tie  = sum(r["label"] ==  0 for r in records)
print(f"BM25 wins:{n_win}  loses:{n_lose}  ties:{n_tie}  total:{len(records)}")


# ── oracle ───────────────────────────────────────────────────────────────────
def metric(entry, key, k): return get_metric(entry, key, k)

ks = ["1", "5", "10"]
bm25_avg = {k: np.mean([metric(bm25_pq[q], "ndcg_at", k) for q in qids]) for k in ks}
lit5_avg = {k: np.mean([metric(lit5_pq[q], "ndcg_at", k) for q in qids]) for k in ks}
oracle   = {k: np.mean([max(metric(bm25_pq[q], "ndcg_at", k),
                            metric(lit5_pq[q], "ndcg_at", k)) for q in qids]) for k in ks}

print(f"\nBaselines (n={len(qids)}):")
print(f"  BM25            nDCG@1={bm25_avg['1']:.4f}  @5={bm25_avg['5']:.4f}  @10={bm25_avg['10']:.4f}")
print(f"  LiT5            nDCG@1={lit5_avg['1']:.4f}  @5={lit5_avg['5']:.4f}  @10={lit5_avg['10']:.4f}")
print(f"  Oracle (perfect routing per @10)  "
      f"@1={oracle['1']:.4f}  @5={oracle['5']:.4f}  @10={oracle['10']:.4f}")


# ── AUROC of each signal at separating wins (label=+1) vs losses (label=−1) ──
def auroc(scores_pos, scores_neg):
    """Mann-Whitney U / |pos|*|neg|.  Higher score → 'more likely pos'."""
    if not scores_pos or not scores_neg:
        return float("nan")
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    all_scores = [(s, 1) for s in scores_pos] + [(s, 0) for s in scores_neg]
    all_scores.sort()
    # rank-sum
    ranks = {}
    i = 0
    while i < len(all_scores):
        j = i
        while j + 1 < len(all_scores) and all_scores[j + 1][0] == all_scores[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(ranks[k] for k, (_, lbl) in enumerate(all_scores) if lbl == 1)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


win_recs  = [r for r in records if r["label"] ==  1]
lose_recs = [r for r in records if r["label"] == -1]

SIGNALS = ["h20", "h50", "gap1_2", "gap1_5", "top1", "mean_top10"]
print(f"\nAUROC of each signal at separating BM25-wins ({len(win_recs)}) "
      f"from BM25-loses ({len(lose_recs)}):")
print("(0.5 = random; >0.5 = high signal → more likely BM25-win; "
      "<0.5 = low signal → more likely BM25-win)")
auc_table = {}
for sig in SIGNALS:
    pos = [r[sig] for r in win_recs]
    neg = [r[sig] for r in lose_recs]
    a = auroc(pos, neg)
    auc_table[sig] = a
    direction = "↑" if a > 0.5 else "↓"
    print(f"  {sig:<12} AUROC = {a:.4f}   ({direction} signal favours BM25)")


# ── sweep: route to BM25 when signal predicts BM25 will win ──────────────────
# For "↑" signals (AUROC > 0.5): route to BM25 when signal >= τ
# For "↓" signals (AUROC < 0.5): route to BM25 when signal <= τ
# Sweep over the empirical signal grid.

def simulate_route(records, signal, direction, tau):
    """direction='ge' → BM25 when sig>=tau; 'le' → BM25 when sig<=tau."""
    n1, n5, n10 = [], [], []
    n_bm25 = 0
    for r in records:
        if direction == "ge":
            use_bm25 = r[signal] >= tau
        else:
            use_bm25 = r[signal] <= tau
        if use_bm25:
            n_bm25 += 1
            n1.append(metric(bm25_pq[r["qid"]],  "ndcg_at", "1"))
            n5.append(metric(bm25_pq[r["qid"]],  "ndcg_at", "5"))
            n10.append(metric(bm25_pq[r["qid"]], "ndcg_at", "10"))
        else:
            n1.append(metric(lit5_pq[r["qid"]],  "ndcg_at", "1"))
            n5.append(metric(lit5_pq[r["qid"]],  "ndcg_at", "5"))
            n10.append(metric(lit5_pq[r["qid"]], "ndcg_at", "10"))
    return {
        "tau": tau,
        "ndcg1":  float(np.mean(n1)),
        "ndcg5":  float(np.mean(n5)),
        "ndcg10": float(np.mean(n10)),
        "n_bm25": n_bm25,
        "pct_bm25": 100 * n_bm25 / len(records),
    }


print(f"\n{'='*80}")
print("Routing sweep — pick BM25 when signal predicts BM25-win, else LiT5")
print("=" * 80)

best_per_signal = {}
sweep_results = {}
for sig in SIGNALS:
    direction = "ge" if auc_table[sig] >= 0.5 else "le"
    vals = sorted({r[sig] for r in records})
    grid = vals + [math.inf if direction == "ge" else -math.inf]
    sweep = [simulate_route(records, sig, direction, t) for t in grid]
    sweep_results[sig] = sweep
    # filter τ that route at least one to BM25 but not all
    nontrivial = [r for r in sweep if 0 < r["n_bm25"] < len(records)]
    pool = nontrivial if nontrivial else sweep
    best = max(pool, key=lambda r: r["ndcg10"])
    best_per_signal[sig] = (best, direction)
    print(f"\n  {sig:<12} dir={direction}  best τ={best['tau']:.5f}  "
          f"nDCG@1={best['ndcg1']:.4f}  @5={best['ndcg5']:.4f}  @10={best['ndcg10']:.4f}  "
          f"({best['n_bm25']} → BM25, {len(records)-best['n_bm25']} → LiT5)")


# ── save CSVs ────────────────────────────────────────────────────────────────
# per-query records
with open(RESULTS_DIR / "win_per_query.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader()
    for r in records:
        w.writerow(r)

# best-per-signal summary
with open(RESULTS_DIR / "win_signal_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["signal", "auroc", "direction", "best_tau",
                "ndcg1", "ndcg5", "ndcg10", "n_bm25", "pct_bm25"])
    for sig in SIGNALS:
        best, direction = best_per_signal[sig]
        w.writerow([sig, f"{auc_table[sig]:.4f}", direction,
                    f"{best['tau']:.6f}",
                    f"{best['ndcg1']:.4f}", f"{best['ndcg5']:.4f}",
                    f"{best['ndcg10']:.4f}", best["n_bm25"],
                    f"{best['pct_bm25']:.2f}"])

print(f"\nCSV → {RESULTS_DIR/'win_per_query.csv'}")
print(f"CSV → {RESULTS_DIR/'win_signal_summary.csv'}")


# ── plots ────────────────────────────────────────────────────────────────────
# 1) signal distribution: BM25-win vs BM25-lose, one panel per signal
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, sig in zip(axes.flat, SIGNALS):
    pos = [r[sig] for r in win_recs]
    neg = [r[sig] for r in lose_recs]
    bins = 25
    ax.hist(neg, bins=bins, alpha=0.55, color="#d62728",
            density=True, label=f"BM25 loses (n={len(neg)})")
    ax.hist(pos, bins=bins, alpha=0.55, color="#2ca02c",
            density=True, label=f"BM25 wins  (n={len(pos)})")
    ax.set_title(f"{sig}   AUROC={auc_table[sig]:.3f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel(sig, fontsize=9)
    ax.legend(fontsize=8)
fig.suptitle("Can each signal separate BM25-wins from BM25-losses?",
             fontsize=13, fontweight="bold")
fig.tight_layout()
p1 = PLOTS_DIR / "win_signal_distributions.png"
fig.savefig(p1, dpi=150); plt.close(fig)
print(f"Plot → {p1}")

# 2) Pareto routing curves per signal
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
metrics_plot = [("ndcg1", "nDCG@1", bm25_avg["1"],  lit5_avg["1"],  oracle["1"]),
                ("ndcg5", "nDCG@5", bm25_avg["5"],  lit5_avg["5"],  oracle["5"]),
                ("ndcg10","nDCG@10",bm25_avg["10"], lit5_avg["10"], oracle["10"])]
COLOURS = {"h20":"#bcbd22","h50":"#8c564b","gap1_2":"#1f77b4",
           "gap1_5":"#9467bd","top1":"#17becf","mean_top10":"#e377c2"}
for ax, (mkey, mlabel, b, l, o) in zip(axes2, metrics_plot):
    for sig in SIGNALS:
        pts = sweep_results[sig]
        xs = [r["pct_bm25"] for r in pts]
        ys = [r[mkey]       for r in pts]
        # sort by x for visual clarity
        order = np.argsort(xs)
        xs = [xs[i] for i in order]; ys = [ys[i] for i in order]
        ax.plot(xs, ys, color=COLOURS[sig], lw=1.6, alpha=0.85,
                label=f"{sig} (AUC={auc_table[sig]:.2f})")
    ax.axhline(b, color="black", ls=":",  lw=1.2, alpha=0.7, label=f"BM25 ({b:.4f})")
    ax.axhline(l, color="red",   ls=":",  lw=1.2, alpha=0.7, label=f"LiT5 ({l:.4f})")
    ax.axhline(o, color="green", ls="--", lw=1.4, alpha=0.85, label=f"Oracle ({o:.4f})")
    ax.set_xlabel("% queries routed to BM25", fontsize=10)
    ax.set_ylabel(mlabel, fontsize=10)
    ax.set_title(mlabel, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="lower left")
fig2.suptitle(f"Routing sweep — BM25 vs LiT5 per signal  (n={len(records)})",
              fontsize=13, fontweight="bold")
fig2.tight_layout()
p2 = PLOTS_DIR / "win_routing_pareto.png"
fig2.savefig(p2, dpi=150); plt.close(fig2)
print(f"Plot → {p2}")

print("\nDone.")
