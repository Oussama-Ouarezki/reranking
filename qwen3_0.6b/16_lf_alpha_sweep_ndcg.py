"""Fine-grained LF alpha sweep targeting nDCG@10 (main metric).

score(d) = alpha * qwen_prob(d) + (1 - alpha) * bm25_minmax(d)

Two-pass grid:
  - Coarse: alpha in [0.0, 1.0] step 0.05 (21 pts)
  - Fine:   alpha in [0.95, 1.0] step 0.001 (51 pts)
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ir_measures import ScoredDoc

from _common import (
    BASE, METRIC_NAMES, TYPES, evaluate, load_qrels, load_qwen_scores, minmax,
)

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

SCORES_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
OUT_TSV  = BASE / "qwen3_0.6b/results/lf_alpha_sweep_ndcg10.tsv"
OUT_JSON = BASE / "qwen3_0.6b/results/lf_alpha_best_ndcg10.json"
PLOT     = BASE / "qwen3_0.6b/plots/lf_alpha_ndcg10.png"

COARSE = np.round(np.linspace(0.0, 1.0, 21), 4)
FINE   = np.round(np.linspace(0.95, 1.0, 51), 4)
ALPHAS = np.unique(np.concatenate([COARSE, FINE]))


def main() -> None:
    rows = load_qwen_scores(SCORES_F)
    qrels = load_qrels()
    qtypes = {r["qid"]: r["type"] for r in rows}

    qwen_raw, bm25_norm, docids = {}, {}, {}
    for r in rows:
        qid = r["qid"]
        items = r["scores"]
        qwen_raw[qid] = np.array([s["qwen_prob"] for s in items], dtype=float)
        bm25_norm[qid] = minmax(np.array([s["bm25_score"] for s in items], dtype=float))
        docids[qid] = [s["docid"] for s in items]

    qids = list(qwen_raw.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # Pure-Qwen baseline (alpha=1.0)
    base_run = [ScoredDoc(qid, d, float(s))
                for qid in qids
                for d, s in zip(docids[qid], qwen_raw[qid])]
    base = {sc: evaluate(base_run, qrels, qs) for sc, qs in scopes}

    print(f"Sweeping {len(ALPHAS)} alpha values (coarse + fine grid)…")
    sweep_rows = []
    for alpha in tqdm(ALPHAS, desc="alpha"):
        run = [ScoredDoc(qid, d, float(s))
               for qid in qids
               for d, s in zip(docids[qid],
                               alpha * qwen_raw[qid] + (1 - alpha) * bm25_norm[qid])]
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            sweep_rows.append({"scope": sc, "alpha": float(alpha), **m})

    df = pd.DataFrame(sweep_rows)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    # Best per scope (target = nDCG@10)
    print("\n" + "=" * 100)
    print("BEST α PER SCOPE (target: nDCG@10)")
    print("=" * 100)
    print(f"  {'scope':<8}  {'α*':>7}  "
          + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δ'+m:>9}" for m in METRIC_NAMES))

    results: dict[str, dict] = {}
    for sc, _ in scopes:
        sub = df[df["scope"] == sc]
        best = sub.loc[sub["ndcg@10"].idxmax()]
        bv = base[sc]
        vals = "  ".join(f"{best[m]:>8.4f}" for m in METRIC_NAMES)
        deltas = "  ".join(f"{best[m]-bv[m]:>+9.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {best['alpha']:>7.4f}  {vals}  {deltas}")
        results[sc] = {
            "alpha_star": float(best["alpha"]),
            "metrics": {m: float(best[m]) for m in METRIC_NAMES},
            "baseline_pure_qwen": {m: float(bv[m]) for m in METRIC_NAMES},
            "delta_vs_qwen": {m: float(best[m] - bv[m]) for m in METRIC_NAMES},
        }

    # Apply per-type best alphas
    per_type_alpha = {sc: results[sc]["alpha_star"] for sc in TYPES}
    fused_run = []
    for qid in qids:
        a = per_type_alpha[qtypes[qid]]
        s_ = a * qwen_raw[qid] + (1 - a) * bm25_norm[qid]
        for d, sv in zip(docids[qid], s_):
            fused_run.append(ScoredDoc(qid, d, float(sv)))
    print(f"\nPer-type α*: {per_type_alpha}")
    print("Combined per-type α run:")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    fused_metrics = {}
    for sc, qs in scopes:
        m = evaluate(fused_run, qrels, qs)
        fused_metrics[sc] = m
        print(f"  {sc:<8}  " + "  ".join(f"{m[k]:>8.4f}" for k in METRIC_NAMES))

    results["per_type_combined"] = {"alphas": per_type_alpha, "metrics": fused_metrics}
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Plot
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    n = len(scopes)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.5))
    palette = {"ndcg@1": "#e377c2", "ndcg@5": "#9467bd",
               "ndcg@10": "#ff7f0e", "mrr@10": "#1f77b4"}
    for ax, (sc, _) in zip(axes, scopes):
        sub = df[df["scope"] == sc].sort_values("alpha")
        for m in METRIC_NAMES:
            ax.plot(sub["alpha"], sub[m], lw=1.4, color=palette[m], label=m)
            ax.axhline(base[sc][m], color=palette[m], ls=":", lw=0.7, alpha=0.5)
        a_star = results[sc]["alpha_star"]
        ax.axvline(a_star, color="red", lw=1.2, ls="-", label=f"α*={a_star:.3f}")
        ax.set_xlabel("α (Qwen weight)")
        ax.set_ylabel("score")
        ax.set_title(sc, fontweight="bold")
        ax.set_xlim(0.9, 1.001)
        ax.legend(fontsize=7)
    fig.suptitle("LF α sweep — target nDCG@10  (zoom α∈[0.9,1.0]; dotted = pure Qwen)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOT, dpi=150)
    plt.close(fig)
    print(f"  → {PLOT}")


if __name__ == "__main__":
    main()
