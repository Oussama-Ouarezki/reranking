"""LF α sweep on 500 train queries — α from 0.999 up to 10 nines (plus pure Qwen).

Reads:  optimal alpha/data/qwen06b_scores_train500.jsonl
        optimal alpha/data/qrels_train500.tsv
Writes: optimal alpha/results/alpha_sweep_high.tsv
        optimal alpha/results/alpha_best_high.json
        optimal alpha/plots/alpha_bar_clusters_high.png
"""

import json
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
OUT_TSV  = BASE / "optimal alpha/results/alpha_sweep_high.tsv"
OUT_JSON = BASE / "optimal alpha/results/alpha_best_high.json"
OUT_PLOT = BASE / "optimal alpha/plots/alpha_bar_clusters_high.png"

# α ∈ {0.999, 0.9999, ..., 0.9999999999, 1.0}
ALPHAS = [1.0 - 10 ** (-k) for k in range(3, 11)]

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
    qids = list(qwen_raw.keys())

    qrels: list[Qrel] = []
    qid_set = set(qids)
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0] in qid_set:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    print(f"{len(qids)} queries, {len(qrels)} qrels")

    sweep_rows = []
    for alpha in tqdm(ALPHAS, desc="alpha"):
        run = []
        for qid in qids:
            fused = alpha * qwen_raw[qid] + (1.0 - alpha) * bm25_norm[qid]
            for d, s in zip(docids[qid], fused):
                run.append(ScoredDoc(qid, d, float(s)))
        res = ir_measures.calc_aggregate(METRICS, qrels, run)
        m = {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}
        sweep_rows.append({"alpha": float(alpha), "label": fmt_alpha(alpha), **m})

    df = pd.DataFrame(sweep_rows)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    print("\n" + "=" * 80)
    print(f"  {'α':<22}  " + "  ".join(f"{m:>9}" for m in METRIC_NAMES))
    print("=" * 80)
    for r in sweep_rows:
        vals = "  ".join(f"{r[m]:>9.4f}" for m in METRIC_NAMES)
        print(f"  {r['label']:<22}  {vals}")

    overall_idx = df["ndcg@10"].idxmax()
    overall = df.iloc[overall_idx]
    print(f"\n=== OVERALL BEST (by nDCG@10) ===")
    print(f"  α={overall['label']}  ndcg@1={overall['ndcg@1']:.4f}  "
          f"ndcg@5={overall['ndcg@5']:.4f}  ndcg@10={overall['ndcg@10']:.4f}  "
          f"mrr@10={overall['mrr@10']:.4f}")

    OUT_JSON.write_text(json.dumps({
        "n_queries": len(qids),
        "alphas_tested": [float(a) for a in ALPHAS],
        "sweep": sweep_rows,
        "best_overall_ndcg10": {
            "alpha": float(overall["alpha"]),
            "label": overall["label"],
            "metrics": {m: float(overall[m]) for m in METRIC_NAMES},
        },
    }, indent=2))
    print(f"wrote {OUT_JSON}")

    plot_metrics = ["ndcg@1", "ndcg@5", "ndcg@10"]
    palette = {"ndcg@1": "#e377c2", "ndcg@5": "#9467bd", "ndcg@10": "#ff7f0e"}
    n_alpha = len(ALPHAS)
    bar_w = 0.27
    x = np.arange(n_alpha)

    fig, ax = plt.subplots(figsize=(max(12, n_alpha * 1.2), 6))
    for i, m in enumerate(plot_metrics):
        offsets = (i - (len(plot_metrics) - 1) / 2) * bar_w
        vals = df[m].values
        bars = ax.bar(x + offsets, vals, bar_w, label=m, color=palette[m],
                      edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0005,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=90)

    ax.axvspan(overall_idx - 0.5, overall_idx + 0.5, color="yellow", alpha=0.15,
               zorder=0, label=f"best nDCG@10 (α={overall['label']})")

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_alpha(a) for a in ALPHAS], rotation=45, ha="right")
    ax.set_xlabel("α (Qwen weight)")
    ax.set_ylabel("score")
    ax.set_title(f"LF α sweep — Qwen3-0.6B on {len(qids)} BioASQ training queries\n"
                 f"high-α region (α ≥ 0.999); bars = nDCG@1 / @5 / @10",
                 fontweight="bold")
    ymin = max(0.0, float(df[plot_metrics].min().min()) - 0.005)
    ymax = float(df[plot_metrics].max().max()) + 0.01
    ax.set_ylim(ymin, ymax)
    ax.legend(loc="lower left")
    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PLOT}")


if __name__ == "__main__":
    main()
