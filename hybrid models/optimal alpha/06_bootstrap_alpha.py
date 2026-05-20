"""Bootstrap LF α sweep — random subsamples from the 500 train queries.

Repeats: B=100 random subsamples (size N=300), runs the α sweep on each,
records the best α per nDCG@10. Final best α = the one that wins most often
(mode), with a stability score (wins / B).

Reads:  optimal alpha/data/qwen06b_scores_train500.jsonl
        optimal alpha/data/qrels_train500.tsv
Writes: optimal alpha/results/bootstrap_alpha.tsv
        optimal alpha/results/bootstrap_alpha_best.json
        optimal alpha/plots/bootstrap_alpha_winrate.png
"""

import json
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import ir_measures
from ir_measures import nDCG, RR, Qrel, ScoredDoc

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "optimal alpha/data/qwen06b_scores_train500.jsonl"
QRELS_F  = BASE / "optimal alpha/data/qrels_train500.tsv"
OUT_TSV  = BASE / "optimal alpha/results/bootstrap_alpha.tsv"
OUT_JSON = BASE / "optimal alpha/results/bootstrap_alpha_best.json"
OUT_PLOT = BASE / "optimal alpha/plots/bootstrap_alpha_winrate.png"

ALPHAS = [1.0 - 10 ** (-k) for k in range(1, 11)] + [1.0]
B = 100         # bootstrap iterations
N_SUB = 300     # subsample size per iteration
SEED = 42

METRICS = [nDCG @ 1, nDCG @ 5, nDCG @ 10, RR @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@10"]


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fmt_alpha(a: float) -> str:
    if a >= 1.0:
        return "1.0 (pure Qwen)"
    s = f"{a:.10f}".rstrip("0").rstrip(".")
    return s


def main() -> None:
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    qwen_raw, bm25_norm, docids = {}, {}, {}
    for r in rows:
        qid = r["qid"]
        items = r["scores"]
        qwen_raw[qid] = np.array([s["qwen_prob"] for s in items], dtype=float)
        bm25_norm[qid] = minmax(np.array([s["bm25_score"] for s in items], dtype=float))
        docids[qid] = [s["docid"] for s in items]
    all_qids = list(qwen_raw.keys())

    qrels_by_qid: dict[str, list[Qrel]] = {}
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels_by_qid.setdefault(p[0], []).append(Qrel(p[0], p[1], int(p[2])))
    print(f"{len(all_qids)} queries available, B={B} bootstraps, subsample size={N_SUB}")

    # Pre-compute fused-score arrays for every α and every qid (saves work in the loop).
    runs_per_alpha: dict[float, dict[str, list[ScoredDoc]]] = {}
    for alpha in ALPHAS:
        d_run = {}
        for qid in all_qids:
            fused = alpha * qwen_raw[qid] + (1.0 - alpha) * bm25_norm[qid]
            d_run[qid] = [ScoredDoc(qid, d, float(s)) for d, s in zip(docids[qid], fused)]
        runs_per_alpha[alpha] = d_run

    rng = random.Random(SEED)
    rows_out = []
    winners_by_metric: dict[str, list[str]] = {m: [] for m in METRIC_NAMES}

    for b in tqdm(range(B), desc="bootstrap"):
        sub = rng.sample(all_qids, N_SUB)
        sub_set = set(sub)
        sub_qrels = [q for qid in sub for q in qrels_by_qid.get(qid, [])]

        for alpha in ALPHAS:
            run = [sd for qid in sub for sd in runs_per_alpha[alpha][qid]]
            res = ir_measures.calc_aggregate(METRICS, sub_qrels, run)
            row = {
                "iter": b, "alpha": float(alpha), "label": fmt_alpha(alpha),
                **{METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))},
            }
            rows_out.append(row)

        for m in METRIC_NAMES:
            best = max(
                (r for r in rows_out if r["iter"] == b),
                key=lambda r: r[m],
            )
            winners_by_metric[m].append(best["label"])

    df = pd.DataFrame(rows_out)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # Win counts per α per metric
    print("\n" + "=" * 88)
    print(f"  WIN COUNT (out of {B}) — α that maximised each metric per bootstrap iter")
    print("=" * 88)
    label_order = [fmt_alpha(a) for a in ALPHAS]
    print(f"  {'α':<22}  " + "  ".join(f"{m:>9}" for m in METRIC_NAMES))
    win_counts: dict[str, dict[str, int]] = {lab: {m: 0 for m in METRIC_NAMES} for lab in label_order}
    for m in METRIC_NAMES:
        c = Counter(winners_by_metric[m])
        for lab in label_order:
            win_counts[lab][m] = c.get(lab, 0)
    for lab in label_order:
        vals = "  ".join(f"{win_counts[lab][m]:>9d}" for m in METRIC_NAMES)
        print(f"  {lab:<22}  {vals}")

    # Mean metrics per α (across iterations)
    mean_by_alpha = df.groupby("label", sort=False)[METRIC_NAMES].mean().reindex(label_order)
    print("\n" + "=" * 88)
    print(f"  MEAN METRICS (across {B} bootstrap iters)")
    print("=" * 88)
    print(f"  {'α':<22}  " + "  ".join(f"{m:>9}" for m in METRIC_NAMES))
    for lab in label_order:
        vals = "  ".join(f"{mean_by_alpha.loc[lab, m]:>9.4f}" for m in METRIC_NAMES)
        print(f"  {lab:<22}  {vals}")

    best_by_mean = {m: mean_by_alpha[m].idxmax() for m in METRIC_NAMES}
    best_by_winrate = {m: max(label_order, key=lambda lab: win_counts[lab][m])
                       for m in METRIC_NAMES}

    print("\n=== Best α per metric ===")
    print(f"  {'metric':<10}  {'by mean':<22}  {'by win-rate':<22}")
    for m in METRIC_NAMES:
        print(f"  {m:<10}  {best_by_mean[m]:<22}  {best_by_winrate[m]:<22}")

    overall_label = mean_by_alpha["ndcg@10"].idxmax()
    overall_mean = mean_by_alpha.loc[overall_label].to_dict()
    overall_winrate = win_counts[overall_label]["ndcg@10"] / B

    print(f"\n=== OVERALL BEST (by mean nDCG@10) ===")
    print(f"  α={overall_label}  mean nDCG@10={overall_mean['ndcg@10']:.4f}  "
          f"won {overall_winrate:.0%} of bootstraps on nDCG@10")

    OUT_JSON.write_text(json.dumps({
        "n_queries": len(all_qids),
        "B": B, "subsample_size": N_SUB, "seed": SEED,
        "alphas_tested": [float(a) for a in ALPHAS],
        "win_counts": win_counts,
        "mean_metrics": {lab: {m: float(mean_by_alpha.loc[lab, m]) for m in METRIC_NAMES}
                         for lab in label_order},
        "best_by_mean": best_by_mean,
        "best_by_winrate": best_by_winrate,
        "best_overall_ndcg10": {
            "label": overall_label,
            "mean_metrics": {k: float(v) for k, v in overall_mean.items()},
            "ndcg10_winrate": overall_winrate,
        },
    }, indent=2))
    print(f"wrote {OUT_JSON}")

    # Plot — win-rate per α for each of nDCG@1/@5/@10
    plot_metrics = ["ndcg@1", "ndcg@5", "ndcg@10"]
    palette = {"ndcg@1": "#e377c2", "ndcg@5": "#9467bd", "ndcg@10": "#ff7f0e"}
    n_alpha = len(ALPHAS)
    bar_w = 0.27
    x = np.arange(n_alpha)

    fig, ax = plt.subplots(figsize=(max(12, n_alpha * 1.1), 6))
    for i, m in enumerate(plot_metrics):
        offsets = (i - (len(plot_metrics) - 1) / 2) * bar_w
        vals = [win_counts[lab][m] / B * 100 for lab in label_order]
        bars = ax.bar(x + offsets, vals, bar_w, label=m, color=palette[m],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=7)

    overall_idx = label_order.index(overall_label)
    ax.axvspan(overall_idx - 0.5, overall_idx + 0.5, color="yellow", alpha=0.2,
               zorder=0, label=f"best mean nDCG@10 (α={overall_label})")

    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=45, ha="right")
    ax.set_xlabel("α (Qwen weight)")
    ax.set_ylabel("win-rate (%)")
    ax.set_title(f"Bootstrap α stability — {B} random subsamples of {N_SUB} queries from 500 train\n"
                 f"bars = % of bootstraps where this α maximised the metric",
                 fontweight="bold")
    ax.set_ylim(0, max(105, max(win_counts[lab][m] / B * 100
                                 for lab in label_order for m in plot_metrics) + 5))
    ax.legend(loc="upper right")
    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PLOT}")


if __name__ == "__main__":
    main()
