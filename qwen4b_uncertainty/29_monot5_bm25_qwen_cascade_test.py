"""monoT5+BM25 → Qwen4B+BM25 cascade via H@20 entropy gate.

Pipeline per query:
  1. Fuse monoT5 + BM25:  score = alpha_t5[type] * monot5_norm + (1-alpha_t5[type]) * bm25_norm
  2. Compute H@20 on raw monoT5 probs (uncertainty signal)
  3. If H@20 > tau  →  escalate: use Qwen+BM25 instead
                        score = alpha_qw[type] * qwen_norm + (1-alpha_qw[type]) * bm25_norm
  4. Else           →  keep monoT5+BM25 score

Alphas are fixed to the per-type Stage-1 optima from scripts 27 (monoT5) and 23 (Qwen).
Sweeps tau in [0, 1] (101 steps) per scope.

Reads:  qwen4b_uncertainty/data/monot5_scores_test.jsonl
        qwen4b_uncertainty/data/qwen_scores_test.jsonl
        qwen4b_uncertainty/data/monot5_alpha_best_test.json
        qwen4b_uncertainty/data/alpha_best_test.json
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/cascade_monot5_qwen_test.tsv
        qwen4b_uncertainty/data/cascade_monot5_qwen_best_test.json
        qwen4b_uncertainty/plots/cascade_monot5_qwen_test.png
"""

import json
import math
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
T5_SCORES_F  = BASE / "qwen4b_uncertainty/data/monot5_scores_test.jsonl"
QW_SCORES_F  = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
T5_ALPHA_F   = BASE / "qwen4b_uncertainty/data/monot5_alpha_best_test.json"
QW_ALPHA_F   = BASE / "qwen4b_uncertainty/data/alpha_best_test.json"
QRELS_F      = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV      = BASE / "qwen4b_uncertainty/data/cascade_monot5_qwen_test.tsv"
OUT_JSON     = BASE / "qwen4b_uncertainty/data/cascade_monot5_qwen_best_test.json"
PLOTS        = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES  = ["summary", "factoid", "list", "yesno"]
TAUS   = np.round(np.linspace(0.0, 1.0, 101), 3)

METRICS      = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]

TYPE_TARGETS = {
    "global":  "ndcg@10",
    "summary": "ndcg@10",
    "factoid": "ndcg@5",
    "list":    "ndcg@3",
    "yesno":   "ndcg@1",
}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def norm_entropy_top20(vals: np.ndarray) -> float:
    top = np.sort(vals)[::-1][:20]
    s = top.sum()
    if s <= 0 or len(top) < 2:
        return 0.0
    p = top / s
    p = np.clip(p, 1e-15, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(len(top)))


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def evaluate(run: list[ScoredDoc], qrels: list[Qrel], qid_set: set[str]) -> dict[str, float]:
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q   = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main() -> None:
    # Load monoT5 scores
    t5_rows: dict[str, dict] = {}
    with T5_SCORES_F.open() as f:
        for line in f:
            r = json.loads(line)
            t5_rows[r["qid"]] = r

    # Load Qwen scores
    qw_rows: dict[str, dict] = {}
    with QW_SCORES_F.open() as f:
        for line in f:
            r = json.loads(line)
            qw_rows[r["qid"]] = r

    qids = sorted(set(t5_rows) & set(qw_rows))
    print(f"{len(qids)} queries with both monoT5 and Qwen scores")

    qrels = load_qrels()
    print(f"{len(qrels)} qrel rows")

    t5_alpha_cfg = json.loads(T5_ALPHA_F.read_text())
    qw_alpha_cfg = json.loads(QW_ALPHA_F.read_text())

    # Per-type fixed alphas
    alpha_t5: dict[str, float] = {sc: t5_alpha_cfg[sc]["alpha_star"] for sc in ["global"] + TYPES}
    alpha_qw: dict[str, float] = {sc: qw_alpha_cfg[sc]["alpha_star"] for sc in ["global"] + TYPES}

    print("\nFixed alphas:")
    print(f"  {'scope':<8}  {'alpha_monot5':>13}  {'alpha_qwen':>11}")
    for sc in ["global"] + TYPES:
        print(f"  {sc:<8}  {alpha_t5[sc]:>13.3f}  {alpha_qw[sc]:>11.3f}")

    # Pre-compute arrays
    t5_raw:   dict[str, np.ndarray] = {}
    t5_norm:  dict[str, np.ndarray] = {}
    qw_raw:   dict[str, np.ndarray] = {}
    qw_norm:  dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    docids:   dict[str, list[str]]  = {}
    qtypes:   dict[str, str]        = {}
    H20_t5:   dict[str, float]      = {}

    for qid in qids:
        t5 = t5_rows[qid]
        qw = qw_rows[qid]
        qtypes[qid] = t5["type"]

        # align on docids from monoT5 (both come from same BM25 top-50)
        t5_items = {s["docid"]: s for s in t5["scores"]}
        qw_items = {s["docid"]: s for s in qw["scores"]}
        shared = [d for d in [s["docid"] for s in t5["scores"]] if d in qw_items]

        docids[qid] = shared
        t5_p  = np.array([t5_items[d]["monot5_prob"] for d in shared], dtype=float)
        qw_p  = np.array([qw_items[d]["qwen_prob"]   for d in shared], dtype=float)
        bm25  = np.array([t5_items[d]["bm25_score"]  for d in shared], dtype=float)

        t5_raw[qid]    = t5_p
        t5_norm[qid]   = minmax(t5_p)
        qw_raw[qid]    = qw_p
        qw_norm[qid]   = minmax(qw_p)
        bm25_norm[qid] = minmax(bm25)
        H20_t5[qid]    = norm_entropy_top20(t5_p)

    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    h_vals = list(H20_t5.values())
    print(f"\nmonoT5 H@20: min={min(h_vals):.3f}  max={max(h_vals):.3f}  mean={np.mean(h_vals):.3f}")
    for t in TYPES:
        ht = [H20_t5[q] for q in type_qids[t]]
        print(f"  {t:<8}  n={len(ht):3d}  mean={np.mean(ht):.3f}  std={np.std(ht):.3f}")

    # Reference baselines
    def make_run_t5_fused(scope: str) -> list[ScoredDoc]:
        a = alpha_t5[scope]
        return [ScoredDoc(qid, d, float(s))
                for qid in qids
                for d, s in zip(docids[qid],
                                a * t5_norm[qid] + (1 - a) * bm25_norm[qid])]

    def make_run_qw_fused(scope: str) -> list[ScoredDoc]:
        a = alpha_qw[scope]
        return [ScoredDoc(qid, d, float(s))
                for qid in qids
                for d, s in zip(docids[qid],
                                a * qw_norm[qid] + (1 - a) * bm25_norm[qid])]

    base_t5_pure = {sc: evaluate(
        [ScoredDoc(qid, d, float(s)) for qid in qids
         for d, s in zip(docids[qid], t5_raw[qid])], qrels, qs)
        for sc, qs in scopes}

    base_t5_fused = {sc: evaluate(make_run_t5_fused(sc), qrels, qs) for sc, qs in scopes}
    base_qw_fused = {sc: evaluate(make_run_qw_fused(sc), qrels, qs) for sc, qs in scopes}

    print("\nReference baselines (nDCG@10):")
    print(f"  {'scope':<8}  {'monoT5_pure':>12}  {'monoT5+BM25':>12}  {'Qwen+BM25':>10}")
    for sc, _ in scopes:
        print(f"  {sc:<8}  {base_t5_pure[sc]['ndcg@10']:>12.4f}  "
              f"{base_t5_fused[sc]['ndcg@10']:>12.4f}  "
              f"{base_qw_fused[sc]['ndcg@10']:>10.4f}")

    # Cascade tau sweep — each scope uses its own alpha pair
    print(f"\nSweeping {len(TAUS)} tau values …")
    sweep_rows = []
    for tau in tqdm(TAUS, desc="tau"):
        for sc, qs in scopes:
            a_t5 = alpha_t5[sc]
            a_qw = alpha_qw[sc]
            run = []
            n_escalated = 0
            for qid in qids:
                if H20_t5[qid] > tau:   # monoT5 uncertain → escalate to Qwen
                    scores = a_qw * qw_norm[qid] + (1 - a_qw) * bm25_norm[qid]
                    n_escalated += 1
                else:                    # monoT5 confident → keep monoT5+BM25
                    scores = a_t5 * t5_norm[qid] + (1 - a_t5) * bm25_norm[qid]
                for d, s in zip(docids[qid], scores):
                    run.append(ScoredDoc(qid, d, float(s)))
            m = evaluate(run, qrels, qs)
            n_esc_sc = sum(1 for q in qs if H20_t5[q] > tau)
            sweep_rows.append({
                "scope": sc,
                "tau": float(tau),
                "n_escalated": n_esc_sc,
                "pct_escalated": round(100.0 * n_esc_sc / len(qs), 1),
                **m,
            })

    df = pd.DataFrame(sweep_rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    def best(scope: str, metric: str) -> dict:
        sub = df[df["scope"] == scope]
        return sub.loc[sub[metric].idxmax()].to_dict()

    print("\n" + "=" * 120)
    print("BEST TAU — monoT5+BM25 → Qwen+BM25 cascade (escalate when monoT5 H@20 > tau)")
    print("=" * 120)
    print(f"  {'scope':<8}  {'tau*':>6}  {'%esc':>6}  "
          + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δt5+bm25_'+m:>13}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δqw+bm25_'+m:>13}" for m in METRIC_NAMES))

    results: dict[str, dict] = {}
    for sc, qs in scopes:
        target = TYPE_TARGETS[sc]
        b   = best(sc, target)
        bt5 = base_t5_fused[sc]
        bqw = base_qw_fused[sc]
        vals   = "  ".join(f"{b[m]:>8.4f}" for m in METRIC_NAMES)
        dt5    = "  ".join(f"{b[m]-bt5[m]:>+13.4f}" for m in METRIC_NAMES)
        dqw    = "  ".join(f"{b[m]-bqw[m]:>+13.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {b['tau']:>6.3f}  {b['pct_escalated']:>5.1f}%  "
              f"{vals}  {dt5}  {dqw}")
        results[sc] = {
            "alpha_monot5":    alpha_t5[sc],
            "alpha_qwen":      alpha_qw[sc],
            "tau_star":        float(b["tau"]),
            "pct_escalated":   float(b["pct_escalated"]),
            "target_metric":   target,
            "metrics":         {m: float(b[m]) for m in METRIC_NAMES},
            "baseline_t5_fused":  {m: float(bt5[m]) for m in METRIC_NAMES},
            "baseline_qw_fused":  {m: float(bqw[m]) for m in METRIC_NAMES},
            "delta_vs_t5_fused":  {m: float(b[m] - bt5[m]) for m in METRIC_NAMES},
            "delta_vs_qw_fused":  {m: float(b[m] - bqw[m]) for m in METRIC_NAMES},
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Plot: tau curves per scope
    n = len(scopes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    pal = {"ndcg@1": "#e377c2", "ndcg@3": "#4c72b0",
           "ndcg@5": "#2ca02c", "ndcg@10": "#1f77b4"}

    for ax, (sc, qs) in zip(axes, scopes):
        sub = df[df["scope"] == sc].sort_values("tau")
        for m in METRIC_NAMES:
            lw = 2.4 if m == TYPE_TARGETS[sc] else 1.2
            ax.plot(sub["tau"], sub[m], lw=lw, color=pal[m], label=m)
            # monoT5+BM25 reference (horizontal)
            ax.axhline(base_t5_fused[sc][m], color=pal[m], ls="--", lw=0.9, alpha=0.6)
            # Qwen+BM25 reference (horizontal)
            ax.axhline(base_qw_fused[sc][m], color=pal[m], ls=":",  lw=0.9, alpha=0.6)

        target   = TYPE_TARGETS[sc]
        tau_star = results[sc]["tau_star"]
        pct      = results[sc]["pct_escalated"]
        ax.axvline(tau_star, color="red", lw=1.8, ls="-",
                   label=f"τ*={tau_star:.3f}\n({pct:.0f}% → Qwen)")

        # secondary axis: % escalated
        ax2 = ax.twinx()
        ax2.fill_between(sub["tau"], sub["pct_escalated"], alpha=0.07, color="gray")
        ax2.plot(sub["tau"], sub["pct_escalated"], color="gray", lw=1.0, alpha=0.5)
        ax2.set_ylabel("% escalated to Qwen", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
        ax2.set_ylim(0, 110)

        ax.set_xlabel("τ  (monoT5 H@20 threshold)")
        ax.set_ylabel("nDCG")
        ax.set_title(
            f"{sc}  (target: {target})\n"
            f"Dashed=monoT5+BM25  Dotted=Qwen+BM25  Red=τ*",
            fontweight="bold", fontsize=9,
        )
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "Cascade: monoT5+BM25 → Qwen4B+BM25 (escalate when monoT5 H@20 > τ)\n"
        "Task13BGoldenEnriched test set",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "cascade_monot5_qwen_test.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
