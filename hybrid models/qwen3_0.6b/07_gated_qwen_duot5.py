"""Gated cascade: Qwen3-0.6B → duoT5.

Per-query confidence margin over Qwen scores:
    conf = (p1 - p2) / (p1 + p2 + 1e-12)
where p1, p2 are the top-1 and top-2 Qwen P(yes) over the 50 BM25 candidates.

Gate (threshold τ):
    conf >= τ  → confident → keep Qwen-only ranking (skip duoT5)
    conf <  τ  → uncertain → use cached duoT5 top-20 ranking

Sweeps τ in [0, 1] in fine steps. For each τ, reports global + per-type metrics
and the fraction of queries routed to duoT5.

Reads:  qwen3_0.6b/data/qwen06b_scores_test.jsonl
        qwen3_0.6b/data/duot5_scores_qwen.jsonl  (cached Exp 3 ranking)
Writes: qwen3_0.6b/results/gated_qwen_duot5_sweep.tsv
        qwen3_0.6b/results/gated_qwen_duot5_best.json
        qwen3_0.6b/plots/gated_qwen_duot5_global.png
"""

import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ir_measures import ScoredDoc

from _common import (
    BASE, METRIC_NAMES, TYPES, evaluate, load_qrels, load_qwen_scores,
)

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

QWEN_F  = BASE / "qwen3_0.6b/data/qwen06b_scores_test.jsonl"
DUOT5_F = BASE / "qwen3_0.6b/data/duot5_scores_qwen.jsonl"
OUT_TSV = BASE / "qwen3_0.6b/results/gated_qwen_duot5_sweep.tsv"
OUT_JSON = BASE / "qwen3_0.6b/results/gated_qwen_duot5_best.json"
PLOTS = BASE / "qwen3_0.6b/plots"

TAUS = np.round(np.concatenate([
    np.linspace(0.00, 0.20, 21),
    np.linspace(0.21, 1.00, 80),
]), 4)


def confidence(probs: np.ndarray) -> float:
    p = np.sort(probs)[::-1]
    p1 = float(p[0])
    p2 = float(p[1]) if len(p) > 1 else 0.0
    denom = p1 + p2
    return (p1 - p2) / denom if denom > 1e-12 else 0.0


def main() -> None:
    qwen_rows = load_qwen_scores(QWEN_F)
    duot5_rows = {r["qid"]: r for r in load_qwen_scores(DUOT5_F)}
    qrels = load_qrels()
    qtypes = {r["qid"]: r["type"] for r in qwen_rows}

    # Per-query data
    qwen_run: dict[str, list[ScoredDoc]] = {}
    duot5_run: dict[str, list[ScoredDoc]] = {}
    confs: dict[str, float] = {}

    for r in qwen_rows:
        qid = r["qid"]
        items = r["scores"]
        probs = np.array([s["qwen_prob"] for s in items], dtype=float)
        confs[qid] = confidence(probs)
        qwen_run[qid] = [ScoredDoc(qid, s["docid"], float(s["qwen_prob"])) for s in items]

        d = duot5_rows.get(qid)
        if d is None:
            duot5_run[qid] = qwen_run[qid]
            continue
        duot5_run[qid] = [ScoredDoc(qid, x["docid"], float(x["duo_score"])) for x in d["ranked"]]

    qids = list(qwen_run.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)

    print(f"Confidence stats: min={min(confs.values()):.4f}  "
          f"max={max(confs.values()):.4f}  "
          f"median={float(np.median(list(confs.values()))):.4f}")

    # Reference points
    qwen_only = [r for qid in qids for r in qwen_run[qid]]
    duot5_all = [r for qid in qids for r in duot5_run[qid]]
    ref_qwen = evaluate(qwen_only, qrels)
    ref_duot5 = evaluate(duot5_all, qrels)
    print("\nReference:")
    print(f"  pure Qwen        : {ref_qwen}")
    print(f"  always duoT5     : {ref_duot5}")

    # Sweep τ
    rows = []
    for tau in TAUS:
        # conf >= tau → Qwen; conf < tau → duoT5
        run: list[ScoredDoc] = []
        n_duot5 = 0
        for qid in qids:
            if confs[qid] < tau:
                run.extend(duot5_run[qid])
                n_duot5 += 1
            else:
                run.extend(qwen_run[qid])

        global_m = evaluate(run, qrels)
        row = {"tau": float(tau), "n_duot5": n_duot5,
               "frac_duot5": n_duot5 / len(qids), "scope": "global", **global_m}
        rows.append(row)
        for t in TYPES:
            type_m = evaluate(run, qrels, type_qids[t])
            n_t = sum(1 for qid in type_qids[t] if confs[qid] < tau)
            rows.append({"tau": float(tau), "n_duot5": n_t,
                         "frac_duot5": n_t / max(len(type_qids[t]), 1),
                         "scope": t, **type_m})

    df = pd.DataFrame(rows)
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"\nwrote {OUT_TSV}")

    # Best per scope/metric
    summary: dict = {"reference": {"pure_qwen": ref_qwen, "always_duot5": ref_duot5}}
    print("\n" + "=" * 100)
    print("BEST τ PER SCOPE × METRIC")
    print("=" * 100)
    print(f"  {'scope':<8} {'metric':<8}  {'τ*':>6}  {'frac_duoT5':>10}  {'value':>8}  "
          f"{'Δ vs Qwen':>9}  {'Δ vs alwaysDuoT5':>17}")
    for sc in ["global"] + TYPES:
        sub = df[df["scope"] == sc]
        scope_summary = {}
        for met in METRIC_NAMES:
            best = sub.loc[sub[met].idxmax()]
            d_qwen = best[met] - ref_qwen[met]
            d_duot5 = best[met] - ref_duot5[met]
            print(f"  {sc:<8} {met:<8}  {best['tau']:>6.4f}  {best['frac_duot5']:>10.3f}  "
                  f"{best[met]:>8.4f}  {d_qwen:>+9.4f}  {d_duot5:>+17.4f}")
            scope_summary[met] = {
                "tau": float(best["tau"]),
                "frac_duot5": float(best["frac_duot5"]),
                "value": float(best[met]),
                "delta_vs_qwen": float(d_qwen),
                "delta_vs_alwaysDuoT5": float(d_duot5),
            }
        summary[sc] = scope_summary

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Plot global metrics vs τ
    PLOTS.mkdir(parents=True, exist_ok=True)
    g = df[df["scope"] == "global"].sort_values("tau")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    palette = {"ndcg@1": "#e377c2", "ndcg@5": "#2ca02c",
               "ndcg@10": "#1f77b4", "mrr@10": "#ff7f0e"}
    for m in METRIC_NAMES:
        ax1.plot(g["tau"], g[m], marker=".", lw=1.4, color=palette[m], label=m)
        ax1.axhline(ref_qwen[m], color=palette[m], ls=":", lw=0.8, alpha=0.5)
        ax1.axhline(ref_duot5[m], color=palette[m], ls="--", lw=0.8, alpha=0.5)
    ax1.set_xlabel("τ  (gate threshold on Qwen confidence)")
    ax1.set_ylabel("score")
    ax1.set_title("Gated Qwen→duoT5 cascade — global\n"
                  "Dotted = pure Qwen.  Dashed = always duoT5.", fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(g["tau"], g["frac_duot5"], color="grey", ls="-.", lw=1.0, label="frac→duoT5")
    ax2.set_ylabel("fraction of queries routed to duoT5", color="grey")
    ax2.set_ylim(0, 1)
    ax1.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    p = PLOTS / "gated_qwen_duot5_global.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
