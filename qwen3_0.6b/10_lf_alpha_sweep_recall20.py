"""Sweep linear-fusion alpha per query type to maximize Recall@20.

score(d) = alpha * qwen_prob(d) + (1 - alpha) * bm25_minmax(d)

Sweeps alpha ∈ [0, 1] (101 steps). For each scope (global + per type),
finds alpha* that maximizes Recall@20.

Reads:  qwen3_0.6b/data/qwen06b_scores_test.jsonl
Writes: qwen3_0.6b/results/lf_alpha_sweep_recall20.tsv
        qwen3_0.6b/results/lf_alpha_best_recall20.json
        qwen3_0.6b/plots/lf_alpha_recall20.png
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
from ir_measures import R, nDCG, Qrel, ScoredDoc

from _common import BASE, TYPES, load_qrels, load_qwen_scores, minmax

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

SCORES_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
OUT_TSV  = BASE / "qwen3_0.6b/results/lf_alpha_sweep_recall20.tsv"
OUT_JSON = BASE / "qwen3_0.6b/results/lf_alpha_best_recall20.json"
PLOTS    = BASE / "qwen3_0.6b/plots"

ALPHAS = np.round(np.linspace(0.0, 1.0, 101), 3)
TARGET = R @ 20
SECONDARY = [nDCG @ 1, nDCG @ 5, nDCG @ 10, R @ 10, R @ 20]
SECONDARY_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "recall@10", "recall@20"]


def evaluate_recall20(run: list[ScoredDoc], qrels: list[Qrel],
                      qid_set: set[str] | None = None) -> dict[str, float]:
    if qid_set is not None:
        run = [r for r in run if r.query_id in qid_set]
        qrels = [q for q in qrels if q.query_id in qid_set]
    if not run or not qrels:
        return {n: float("nan") for n in SECONDARY_NAMES}
    res = ir_measures.calc_aggregate(SECONDARY, qrels, run)
    return {SECONDARY_NAMES[i]: float(res[SECONDARY[i]]) for i in range(len(SECONDARY))}


def main() -> None:
    rows = load_qwen_scores(SCORES_F)
    qrels = load_qrels()
    qtypes = {r["qid"]: r["type"] for r in rows}

    qwen_raw: dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    docids: dict[str, list[str]] = {}
    for r in rows:
        qid = r["qid"]
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_raw[qid] = q
        bm25_norm[qid] = minmax(b)
        docids[qid] = [s["docid"] for s in items]

    qids = list(qwen_raw.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # Pure-Qwen baseline
    base_run = [ScoredDoc(qid, d, float(s))
                for qid in qids
                for d, s in zip(docids[qid], qwen_raw[qid])]
    base = {sc: evaluate_recall20(base_run, qrels, qs) for sc, qs in scopes}
    print("Pure Qwen baseline:")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>10}" for m in SECONDARY_NAMES))
    for sc, _ in scopes:
        vals = "  ".join(f"{base[sc][m]:>10.4f}" for m in SECONDARY_NAMES)
        print(f"  {sc:<8}  {vals}")

    # Sweep
    print(f"\nSweeping {len(ALPHAS)} alpha values …")
    sweep_rows = []
    for alpha in tqdm(ALPHAS, desc="alpha"):
        run = [ScoredDoc(qid, d, float(s))
               for qid in qids
               for d, s in zip(docids[qid],
                               alpha * qwen_raw[qid] + (1 - alpha) * bm25_norm[qid])]
        for sc, qs in scopes:
            m = evaluate_recall20(run, qrels, qs)
            sweep_rows.append({"scope": sc, "alpha": float(alpha), **m})

    df = pd.DataFrame(sweep_rows)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    # Best per scope (target = recall@20)
    print("\n" + "=" * 110)
    print("BEST α PER SCOPE (target: recall@20)")
    print("=" * 110)
    print(f"  {'scope':<8}  {'alpha*':>7}  "
          + "  ".join(f"{m:>10}" for m in SECONDARY_NAMES)
          + "  " + "  ".join(f"{'Δ'+m:>11}" for m in SECONDARY_NAMES))

    results: dict[str, dict] = {}
    for sc, _ in scopes:
        sub = df[df["scope"] == sc]
        best = sub.loc[sub["recall@20"].idxmax()]
        bv = base[sc]
        deltas = "  ".join(f"{best[m]-bv[m]:>+11.4f}" for m in SECONDARY_NAMES)
        vals   = "  ".join(f"{best[m]:>10.4f}" for m in SECONDARY_NAMES)
        print(f"  {sc:<8}  {best['alpha']:>7.3f}  {vals}  {deltas}")
        results[sc] = {
            "alpha_star": float(best["alpha"]),
            "metrics": {m: float(best[m]) for m in SECONDARY_NAMES},
            "baseline_pure_qwen": {m: float(bv[m]) for m in SECONDARY_NAMES},
            "delta_vs_qwen": {m: float(best[m] - bv[m]) for m in SECONDARY_NAMES},
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Apply per-type best alphas as a single fused run
    per_type_alpha = {sc: results[sc]["alpha_star"] for sc in TYPES}
    print(f"\nPer-type α*: {per_type_alpha}")
    fused_run = []
    for qid in qids:
        t = qtypes[qid]
        a = per_type_alpha[t]
        s_ = a * qwen_raw[qid] + (1 - a) * bm25_norm[qid]
        for d, sv in zip(docids[qid], s_):
            fused_run.append(ScoredDoc(qid, d, float(sv)))
    fused_metrics: dict = {}
    for sc, qs in scopes:
        fused_metrics[sc] = evaluate_recall20(fused_run, qrels, qs)
    print("\nApplying per-type α* simultaneously:")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>10}" for m in SECONDARY_NAMES))
    for sc, _ in scopes:
        vals = "  ".join(f"{fused_metrics[sc][m]:>10.4f}" for m in SECONDARY_NAMES)
        print(f"  {sc:<8}  {vals}")

    results["per_type_combined"] = {
        "alphas": per_type_alpha,
        "metrics": {sc: fused_metrics[sc] for sc, _ in scopes},
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))

    # Plot
    PLOTS.mkdir(parents=True, exist_ok=True)
    n_scopes = len(scopes)
    fig, axes = plt.subplots(1, n_scopes, figsize=(4.2 * n_scopes, 4.5), sharey=False)
    palette = {"recall@20": "#1f77b4", "recall@10": "#2ca02c",
               "ndcg@10": "#ff7f0e", "ndcg@5": "#9467bd", "ndcg@1": "#e377c2"}
    for ax, (sc, _) in zip(axes, scopes):
        sub = df[df["scope"] == sc].sort_values("alpha")
        for m in SECONDARY_NAMES:
            ax.plot(sub["alpha"], sub[m], lw=1.4, color=palette[m], label=m)
            ax.axhline(base[sc][m], color=palette[m], ls=":", lw=0.7, alpha=0.5)
        a_star = results[sc]["alpha_star"]
        ax.axvline(a_star, color="red", lw=1.4, ls="-", label=f"α*={a_star:.2f}")
        ax.set_xlabel("α (Qwen weight)")
        ax.set_ylabel("score")
        ax.set_title(sc, fontweight="bold")
        ax.legend(fontsize=7)
    fig.suptitle("LF α sweep — target Recall@20  (dotted = pure-Qwen baseline)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = PLOTS / "lf_alpha_recall20.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
