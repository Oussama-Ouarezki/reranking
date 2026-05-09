"""Gated cascade: per-type LF → LiT5 top-20.

Per-query confidence margin over Qwen scores:
    conf = (p1 - p2) / (p1 + p2)
Gate (threshold τ):
    conf >= τ  → keep LF ranking (skip LiT5)
    conf <  τ  → use cached LF→LiT5 ranking

LF α* per type (tuned for Recall@20):
    summary 0.99, factoid 0.99, list 0.99, yesno 0.82

Sweeps τ; reports global + per-type metrics; finds best τ overall.

Reads:  qwen3_0.6b/data/qwen06b_scores_test.jsonl
        qwen3_0.6b/data/lit5_scores_lf.jsonl
Writes: qwen3_0.6b/results/gated_lf_lit5_sweep.tsv
        qwen3_0.6b/results/gated_lf_lit5_best.json
        qwen3_0.6b/plots/gated_lf_lit5_global.png
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ir_measures import ScoredDoc

from _common import (
    BASE, METRIC_NAMES, TYPES, evaluate, load_qrels, load_qwen_scores, minmax,
)

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

LF_ALPHA = {"summary": 0.99, "factoid": 0.99, "list": 0.99, "yesno": 0.82}

QWEN_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
LIT5_F = BASE / "qwen3_0.6b/data/lit5_scores_lf.jsonl"
OUT_TSV = BASE / "qwen3_0.6b/results/gated_lf_lit5_sweep.tsv"
OUT_JSON = BASE / "qwen3_0.6b/results/gated_lf_lit5_best.json"
PLOTS = BASE / "qwen3_0.6b/plots"

TAUS = np.round(np.concatenate([
    np.linspace(0.000, 0.020, 21),
    np.linspace(0.025, 0.200, 36),
    np.linspace(0.22, 1.00, 40),
]), 4)


def confidence(probs: np.ndarray) -> float:
    p = np.sort(probs)[::-1]
    p1 = float(p[0])
    p2 = float(p[1]) if len(p) > 1 else 0.0
    denom = p1 + p2
    return (p1 - p2) / denom if denom > 1e-12 else 0.0


def main() -> None:
    qwen_rows = load_qwen_scores(QWEN_F)
    lit5_rows = {r["qid"]: r for r in load_qwen_scores(LIT5_F)}
    qrels = load_qrels()
    qtypes = {r["qid"]: r["type"] for r in qwen_rows}

    lf_run: dict[str, list[ScoredDoc]] = {}
    lit5_run: dict[str, list[ScoredDoc]] = {}
    confs: dict[str, float] = {}

    for r in qwen_rows:
        qid = r["qid"]
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        a = LF_ALPHA.get(r["type"], 1.0)
        fused = a * q + (1.0 - a) * minmax(b)
        confs[qid] = confidence(q)
        lf_run[qid] = [ScoredDoc(qid, s["docid"], float(fv))
                       for s, fv in zip(items, fused)]

        d = lit5_rows.get(qid)
        if d is None:
            lit5_run[qid] = lf_run[qid]
            continue
        lit5_run[qid] = [ScoredDoc(qid, x["docid"], float(x["score"])) for x in d["ranked"]]

    qids = list(lf_run.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)

    print(f"Confidence stats: min={min(confs.values()):.4f}  "
          f"max={max(confs.values()):.4f}  "
          f"median={float(np.median(list(confs.values()))):.4f}")

    # References
    lf_only = [r for qid in qids for r in lf_run[qid]]
    lit5_all = [r for qid in qids for r in lit5_run[qid]]
    ref_lf = evaluate(lf_only, qrels)
    ref_lit5 = evaluate(lit5_all, qrels)
    print("\nReference:")
    print(f"  LF only          : {ref_lf}")
    print(f"  always LF→LiT5   : {ref_lit5}")

    # Sweep
    rows = []
    for tau in TAUS:
        run: list[ScoredDoc] = []
        n_lit5 = 0
        for qid in qids:
            if confs[qid] < tau:
                run.extend(lit5_run[qid])
                n_lit5 += 1
            else:
                run.extend(lf_run[qid])
        global_m = evaluate(run, qrels)
        rows.append({"tau": float(tau), "n_lit5": n_lit5,
                     "frac_lit5": n_lit5 / len(qids), "scope": "global", **global_m})
        for t in TYPES:
            type_m = evaluate(run, qrels, type_qids[t])
            n_t = sum(1 for qid in type_qids[t] if confs[qid] < tau)
            rows.append({"tau": float(tau), "n_lit5": n_t,
                         "frac_lit5": n_t / max(len(type_qids[t]), 1),
                         "scope": t, **type_m})

    df = pd.DataFrame(rows)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # Best per scope × metric
    summary: dict = {"reference": {"lf_only": ref_lf, "always_lf_lit5": ref_lit5},
                     "lf_alpha": LF_ALPHA}
    print("\n" + "=" * 100)
    print("BEST τ PER SCOPE × METRIC")
    print("=" * 100)
    print(f"  {'scope':<8} {'metric':<8}  {'τ*':>6}  {'frac_LiT5':>10}  {'value':>8}  "
          f"{'Δ vs LF':>9}  {'Δ vs allLiT5':>14}")
    for sc in ["global"] + TYPES:
        sub = df[df["scope"] == sc]
        scope_summary = {}
        for met in METRIC_NAMES:
            best = sub.loc[sub[met].idxmax()]
            d_lf = best[met] - ref_lf[met]
            d_lit5 = best[met] - ref_lit5[met]
            print(f"  {sc:<8} {met:<8}  {best['tau']:>6.4f}  {best['frac_lit5']:>10.3f}  "
                  f"{best[met]:>8.4f}  {d_lf:>+9.4f}  {d_lit5:>+14.4f}")
            scope_summary[met] = {
                "tau": float(best["tau"]),
                "frac_lit5": float(best["frac_lit5"]),
                "value": float(best[met]),
                "delta_vs_lf": float(d_lf),
                "delta_vs_alwaysLiT5": float(d_lit5),
            }
        summary[sc] = scope_summary

    # Single best τ across all four global metrics (mean rank)
    g = df[df["scope"] == "global"].sort_values("tau").reset_index(drop=True)
    g_norm = g[METRIC_NAMES].apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-12))
    g["avg_norm"] = g_norm.mean(axis=1)
    best_overall = g.loc[g["avg_norm"].idxmax()]
    print("\nBest τ across all global metrics (avg-norm):")
    print(f"  τ={best_overall['tau']:.4f}  frac_LiT5={best_overall['frac_lit5']:.3f}")
    for m in METRIC_NAMES:
        print(f"    {m:<8} {best_overall[m]:.4f}  "
              f"(LF {ref_lf[m]:.4f}, allLiT5 {ref_lit5[m]:.4f})")
    summary["best_overall_global"] = {
        "tau": float(best_overall["tau"]),
        "frac_lit5": float(best_overall["frac_lit5"]),
        **{m: float(best_overall[m]) for m in METRIC_NAMES},
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Plot
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    palette = {"ndcg@1": "#e377c2", "ndcg@5": "#2ca02c",
               "ndcg@10": "#1f77b4", "mrr@10": "#ff7f0e"}
    for m in METRIC_NAMES:
        ax1.plot(g["tau"], g[m], marker=".", lw=1.4, color=palette[m], label=m)
        ax1.axhline(ref_lf[m], color=palette[m], ls=":", lw=0.8, alpha=0.5)
        ax1.axhline(ref_lit5[m], color=palette[m], ls="--", lw=0.8, alpha=0.5)
    ax1.axvline(best_overall["tau"], color="red", lw=1.2, ls="-",
                label=f"τ*={best_overall['tau']:.3f}")
    ax1.set_xlabel("τ  (gate threshold on Qwen confidence)")
    ax1.set_ylabel("score")
    ax1.set_title("Gated LF→LiT5 cascade — global\n"
                  "Dotted = LF only.  Dashed = always LF→LiT5.", fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(g["tau"], g["frac_lit5"], color="grey", ls="-.", lw=1.0, label="frac→LiT5")
    ax2.set_ylabel("fraction routed to LiT5", color="grey")
    ax2.set_ylim(0, 1)
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = PLOTS / "gated_lf_lit5_global.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
