"""Gated cascade: gate AFTER LF.

For each query:
    confidence = (p1 - p2) / (p1 + p2) on Qwen probs
    conf >= τ  → keep LF ranking (cheap)
    conf <  τ  → use Exp 8 ranking (LF + duoT5(15-25) + LiT5 top20)

Sweeps τ over a fine grid; reports best τ globally + per type.
α: summary=0.99, factoid=0.99, list=0.99, yesno=0.85.

Reads:  qwen3_0.6b/data/qwen06b_scores_test.jsonl
        qwen3_0.6b/data/lf_duot5unc_top20.jsonl   (Exp 8 ranked)
Writes: qwen3_0.6b/results/gated_lf_exp8_sweep.tsv
        qwen3_0.6b/results/gated_lf_exp8_best.json
        qwen3_0.6b/plots/gated_lf_exp8_global.png
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

LF_ALPHA = {"summary": 0.99, "factoid": 0.99, "list": 0.99, "yesno": 0.85}

QWEN_F = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
EXP8_F = BASE / "qwen3_0.6b/data/lf_duot5unc_top20.jsonl"
OUT_TSV = BASE / "qwen3_0.6b/results/gated_lf_exp8_sweep.tsv"
OUT_JSON = BASE / "qwen3_0.6b/results/gated_lf_exp8_best.json"
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
    exp8_rows = {r["qid"]: r for r in load_qwen_scores(EXP8_F)}
    qrels = load_qrels()
    qtypes = {r["qid"]: r["type"] for r in qwen_rows}

    lf_run: dict[str, list[ScoredDoc]] = {}
    exp8_run: dict[str, list[ScoredDoc]] = {}
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

        d = exp8_rows.get(qid)
        if d is None:
            exp8_run[qid] = lf_run[qid]
            continue
        exp8_run[qid] = [ScoredDoc(qid, x["docid"], float(x["score"])) for x in d["ranked"]]

    qids = list(lf_run.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)

    print(f"Confidence stats: min={min(confs.values()):.4f}  "
          f"max={max(confs.values()):.4f}  "
          f"median={float(np.median(list(confs.values()))):.4f}")

    lf_only = [r for qid in qids for r in lf_run[qid]]
    exp8_all = [r for qid in qids for r in exp8_run[qid]]
    ref_lf = evaluate(lf_only, qrels)
    ref_exp8 = evaluate(exp8_all, qrels)
    print("\nReference:")
    print(f"  LF only            : {ref_lf}")
    print(f"  always Exp 8       : {ref_exp8}")

    rows = []
    for tau in TAUS:
        run: list[ScoredDoc] = []
        n_exp8 = 0
        for qid in qids:
            if confs[qid] < tau:
                run.extend(exp8_run[qid])
                n_exp8 += 1
            else:
                run.extend(lf_run[qid])
        global_m = evaluate(run, qrels)
        rows.append({"tau": float(tau), "n_exp8": n_exp8,
                     "frac_exp8": n_exp8 / len(qids), "scope": "global", **global_m})
        for t in TYPES:
            type_m = evaluate(run, qrels, type_qids[t])
            n_t = sum(1 for qid in type_qids[t] if confs[qid] < tau)
            rows.append({"tau": float(tau), "n_exp8": n_t,
                         "frac_exp8": n_t / max(len(type_qids[t]), 1),
                         "scope": t, **type_m})

    df = pd.DataFrame(rows)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    summary: dict = {"reference": {"lf_only": ref_lf, "always_exp8": ref_exp8},
                     "lf_alpha": LF_ALPHA}
    print("\n" + "=" * 110)
    print("BEST τ PER SCOPE × METRIC")
    print("=" * 110)
    print(f"  {'scope':<8} {'metric':<8}  {'τ*':>7}  {'frac_Exp8':>10}  {'value':>8}  "
          f"{'Δ vs LF':>9}  {'Δ vs allExp8':>14}")
    for sc in ["global"] + TYPES:
        sub = df[df["scope"] == sc]
        scope_summary = {}
        for met in METRIC_NAMES:
            best = sub.loc[sub[met].idxmax()]
            d_lf = best[met] - ref_lf[met]
            d_e8 = best[met] - ref_exp8[met]
            print(f"  {sc:<8} {met:<8}  {best['tau']:>7.4f}  {best['frac_exp8']:>10.3f}  "
                  f"{best[met]:>8.4f}  {d_lf:>+9.4f}  {d_e8:>+14.4f}")
            scope_summary[met] = {
                "tau": float(best["tau"]),
                "frac_exp8": float(best["frac_exp8"]),
                "value": float(best[met]),
                "delta_vs_lf": float(d_lf),
                "delta_vs_alwaysExp8": float(d_e8),
            }
        summary[sc] = scope_summary

    g = df[df["scope"] == "global"].sort_values("tau").reset_index(drop=True)
    g_norm = g[METRIC_NAMES].apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-12))
    g["avg_norm"] = g_norm.mean(axis=1)
    best_overall = g.loc[g["avg_norm"].idxmax()]
    print("\nBest τ across all global metrics (avg-norm):")
    print(f"  τ={best_overall['tau']:.4f}  frac_Exp8={best_overall['frac_exp8']:.3f}")
    for m in METRIC_NAMES:
        print(f"    {m:<8} {best_overall[m]:.4f}  "
              f"(LF {ref_lf[m]:.4f}, allExp8 {ref_exp8[m]:.4f})")
    summary["best_overall_global"] = {
        "tau": float(best_overall["tau"]),
        "frac_exp8": float(best_overall["frac_exp8"]),
        **{m: float(best_overall[m]) for m in METRIC_NAMES},
    }

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_JSON}")

    PLOTS.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    palette = {"ndcg@1": "#e377c2", "ndcg@5": "#2ca02c",
               "ndcg@10": "#1f77b4", "mrr@10": "#ff7f0e"}
    for m in METRIC_NAMES:
        ax1.plot(g["tau"], g[m], marker=".", lw=1.4, color=palette[m], label=m)
        ax1.axhline(ref_lf[m], color=palette[m], ls=":", lw=0.8, alpha=0.5)
        ax1.axhline(ref_exp8[m], color=palette[m], ls="--", lw=0.8, alpha=0.5)
    ax1.axvline(best_overall["tau"], color="red", lw=1.2, ls="-",
                label=f"τ*={best_overall['tau']:.3f}")
    ax1.set_xlabel("τ  (gate threshold on Qwen confidence)")
    ax1.set_ylabel("score")
    ax1.set_title("Gated LF→Exp8 cascade — global\n"
                  "Dotted = LF only.  Dashed = always Exp 8.", fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(g["tau"], g["frac_exp8"], color="grey", ls="-.", lw=1.0, label="frac→Exp8")
    ax2.set_ylabel("fraction routed to Exp 8", color="grey")
    ax2.set_ylim(0, 1)
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = PLOTS / "gated_lf_exp8_global.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
