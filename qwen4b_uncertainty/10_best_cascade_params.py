"""Find production parameters for two Qwen3-4B + BM25 *linear-fusion-only* cascades.

NO gating. Score every query with:
    score(d) = α · qwen_norm(d) + (1-α) · bm25_norm(d)
where qwen_norm and bm25_norm are per-query min-max normalised on the 50 BM25
candidates.

Configuration A — "qwen4b_linear_fusion"  (single global α)
    optimise α for GLOBAL nDCG@10.

Configuration B — "qwen4b_linear_fusion_dynamic"  (per-query-type α)
    list    → optimise α for nDCG@3
    summary → optimise α for nDCG@10
    yesno   → optimise α for nDCG@1
    factoid → optimise α for nDCG@5

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/cascade_params.json
        qwen4b_uncertainty/data/cascade_grid.tsv
        qwen4b_uncertainty/plots/cascade_alpha_curves.png
"""

import json
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

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores.jsonl"
QRELS_F = BASE / "data/bioasq/processed/qrels.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/cascade_params.json"
OUT_TSV = BASE / "qwen4b_uncertainty/data/cascade_grid.tsv"
PLOTS = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]

ALPHAS = np.round(np.linspace(0.0, 1.0, 41), 3)   # 0.000 .. 1.000 step 0.025

TYPE_TARGETS = {
    "list":    "ndcg@3",
    "summary": "ndcg@10",
    "yesno":   "ndcg@1",
    "factoid": "ndcg@5",
}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def load_scores() -> list[dict]:
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate(run: list[ScoredDoc], qrels: list[Qrel],
             qid_set: set[str]) -> dict[str, float]:
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main() -> None:
    rows = load_scores()
    qrels = load_qrels()
    print(f"{len(rows)} queries / {len(qrels)} qrel rows")

    qwen_arr: dict[str, np.ndarray] = {}
    qwen_norm: dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    docids: dict[str, list[str]] = {}
    qtypes: dict[str, str] = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = sorted(r["scores"], key=lambda s: s["qwen_prob"], reverse=True)
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_arr[qid] = q
        qwen_norm[qid] = minmax(q)
        bm25_norm[qid] = minmax(b)
        docids[qid] = [s["docid"] for s in items]

    qids = list(qwen_arr.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    base_run = [ScoredDoc(qid, d, float(s))
                for qid in qids
                for d, s in zip(docids[qid], qwen_arr[qid])]
    base = {sc: evaluate(base_run, qrels, qs) for sc, qs in scopes}
    print("\nPure Qwen baseline:")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    for sc in ["global"] + TYPES:
        print(f"  {sc:<8}  " + "  ".join(f"{base[sc][m]:>8.4f}" for m in METRIC_NAMES))

    print(f"\nSweeping α({len(ALPHAS)}) — pure linear fusion, no gate …")
    rows_out: list[dict] = []
    for alpha in tqdm(ALPHAS, desc="α"):
        run: list[ScoredDoc] = []
        for qid in qids:
            s = alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid]
            for d, sv in zip(docids[qid], s):
                run.append(ScoredDoc(qid, d, float(sv)))
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            rows_out.append({"scope": sc, "alpha": float(alpha), **m})

    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    def best(scope: str, metric: str) -> dict:
        sub = df[df["scope"] == scope]
        return sub.loc[sub[metric].idxmax()].to_dict()

    A = best("global", "ndcg@10")
    B: dict[str, dict] = {}
    for t in TYPES:
        target = TYPE_TARGETS[t]
        b = best(t, target)
        b["target_metric"] = target
        B[t] = b

    print("\n" + "=" * 90)
    print("CONFIG A — qwen4b_linear_fusion  (single global α, target = global nDCG@10)")
    print("=" * 90)
    print(f"  α* = {A['alpha']:.3f}")
    print(f"  scores: " + "  ".join(f"{m}={A[m]:.4f}" for m in METRIC_NAMES))
    print(f"  Δ vs Qwen: " +
          "  ".join(f"{m}={A[m]-base['global'][m]:+.4f}" for m in METRIC_NAMES))

    print("\n" + "=" * 90)
    print("CONFIG B — qwen4b_linear_fusion_dynamic  (per-query-type α)")
    print("=" * 90)
    for t in TYPES:
        b = B[t]
        target = b["target_metric"]
        bv = base[t][target]
        print(f"  {t:<8}  target={target:<7}  α*={b['alpha']:.3f}  "
              f"value={b[target]:.4f}  baseline={bv:.4f}  Δ={b[target]-bv:+.4f}")

    agg_B: dict[str, float] = {}
    for m in METRIC_NAMES:
        s = 0.0
        for t in TYPES:
            s += B[t][m] * len(type_qids[t])
        agg_B[m] = s / len(qids)
    print("\n  Config B weighted-mean across types vs Qwen baseline:")
    for m in METRIC_NAMES:
        d = agg_B[m] - base["global"][m]
        print(f"    {m}: {agg_B[m]:.4f}  (Qwen {base['global'][m]:.4f}  Δ={d:+.4f})")

    params = {
        "qwen4b_linear_fusion": {
            "alpha": float(A["alpha"]),
            "target": "global ndcg@10",
            "metrics_500q": {m: float(A[m]) for m in METRIC_NAMES},
            "notes": (
                "Pure linear fusion, no gating. For every query: "
                "score(d) = alpha * qwen_norm(d) + (1-alpha) * bm25_norm(d). "
                "qwen_norm and bm25_norm are min-max normalised per query."
            ),
        },
        "qwen4b_linear_fusion_dynamic": {
            "fusion_rule": "linear_minmax_no_gate",
            "per_type": {
                t: {
                    "alpha":         float(B[t]["alpha"]),
                    "target_metric": B[t]["target_metric"],
                    "metric_value":  float(B[t][B[t]["target_metric"]]),
                    "metrics_500q":  {m: float(B[t][m]) for m in METRIC_NAMES},
                }
                for t in TYPES
            },
            "fallback_alpha": float(A["alpha"]),
        },
        "baseline_pure_qwen": {sc: {m: float(base[sc][m]) for m in METRIC_NAMES}
                               for sc in ["global"] + TYPES},
    }
    OUT_JSON.write_text(json.dumps(params, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Plot — α vs metric, one panel per scope, all 4 metrics overlaid
    fig, axes = plt.subplots(1, 5, figsize=(28, 5), sharey=False)
    palette = {"ndcg@1": "#e377c2", "ndcg@3": "#4c72b0",
               "ndcg@5": "#2ca02c", "ndcg@10": "#1f77b4"}
    for ax, sc in zip(axes, ["global"] + TYPES):
        sub = df[df["scope"] == sc].sort_values("alpha")
        for m in METRIC_NAMES:
            ax.plot(sub["alpha"], sub[m], marker="o", ms=2.5, lw=1.6,
                    color=palette[m], label=m)
            ax.axhline(base[sc][m], color=palette[m], ls=":", lw=0.8, alpha=0.5)
        # Mark chosen α* with vertical line
        if sc == "global":
            ax.axvline(A["alpha"], color="red", lw=1.6, ls="-",
                       label=f"chosen α*={A['alpha']:.3f}")
        else:
            tm = TYPE_TARGETS[sc]
            ax.axvline(B[sc]["alpha"], color="red", lw=1.6, ls="-",
                       label=f"chosen α*={B[sc]['alpha']:.3f}\n(target {tm})")
        ax.set_xlabel("α (Qwen weight)")
        if ax is axes[0]:
            ax.set_ylabel("nDCG")
        title = sc if sc == "global" else f"{sc}  (target {TYPE_TARGETS[sc]})"
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Linear fusion (no gate) — α sweep per scope.  Red line = chosen α* "
        "(global → nDCG@10; per-type → that type's target metric).",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "cascade_alpha_curves.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
