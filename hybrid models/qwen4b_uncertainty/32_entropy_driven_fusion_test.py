"""Entropy-driven adaptive fusion — monoT5 + BM25, Task13BGoldenEnriched test set.

Algorithm per query:
  1. Compute H@20(monoT5 probs)
  2. If H@20 <= tau  →  certain, keep pure monoT5
  3. If H@20 >  tau  →  uncertain, start increasing BM25 weight:
       for alpha in [0.95, 0.90, 0.85, ..., 0.50]:
           fused = alpha * monot5_norm + (1-alpha) * bm25_norm
           H@20_fused = norm_entropy_top20(fused)
           if H@20_fused <= tau:
               stop  ← model is certain enough
       use the fused scores at the stopping alpha (or lowest alpha if never certain)

Tau is the single global parameter — it controls BOTH the gate AND the stopping criterion.
Sweep tau across [0.0 .. 1.0] (101 values) to find the best global threshold.

Reads:  qwen4b_uncertainty/data/monot5_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/entropy_fusion_test.tsv
        qwen4b_uncertainty/data/entropy_fusion_best_test.json
        qwen4b_uncertainty/plots/entropy_fusion_test.png
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

BASE     = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/monot5_scores_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV  = BASE / "qwen4b_uncertainty/data/entropy_fusion_test.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/entropy_fusion_best_test.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES        = ["summary", "factoid", "list", "yesno"]
TAUS         = np.round(np.linspace(0.0, 1.0, 101), 3)
ALPHAS       = np.round(np.arange(0.95, 0.45, -0.05), 2)   # 0.95, 0.90, …, 0.50

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


def resolve_query(monot5_raw: np.ndarray,
                  monot5_norm: np.ndarray,
                  bm25_norm: np.ndarray,
                  tau: float) -> tuple[np.ndarray, float, int]:
    """Return (scores, final_alpha, steps_taken).

    steps_taken = 0  → certain from the start (no fusion)
    steps_taken = k  → needed k alpha decreases before entropy dropped below tau
    steps_taken = -1 → never became certain, stopped at alpha_floor (0.50)
    """
    h0 = norm_entropy_top20(monot5_raw)
    if h0 <= tau:
        return monot5_raw, 1.0, 0

    # uncertain — iterate through alphas until H@20 drops below tau
    for step, alpha in enumerate(ALPHAS, start=1):
        fused = alpha * monot5_norm + (1 - alpha) * bm25_norm
        h_fused = norm_entropy_top20(fused)
        if h_fused <= tau:
            return fused, float(alpha), step
    # never reached certainty — return lowest alpha attempted
    return fused, float(ALPHAS[-1]), -1


def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def evaluate(run: list[ScoredDoc], qrels: list[Qrel],
             qid_set: set[str]) -> dict[str, float]:
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q   = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main() -> None:
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"{len(rows)} queries loaded")

    qrels = load_qrels()
    print(f"{len(qrels)} qrel rows")

    t5_raw:  dict[str, np.ndarray] = {}
    t5_norm: dict[str, np.ndarray] = {}
    bm25_n:  dict[str, np.ndarray] = {}
    docids:  dict[str, list[str]]  = {}
    qtypes:  dict[str, str]        = {}
    H20:     dict[str, float]      = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = r["scores"]
        q = np.array([s["monot5_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"]  for s in items], dtype=float)
        t5_raw[qid]  = q
        t5_norm[qid] = minmax(q)
        bm25_n[qid]  = minmax(b)
        docids[qid]  = [s["docid"] for s in items]
        H20[qid]     = norm_entropy_top20(q)

    qids = list(t5_raw.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # Baselines
    base_pure = {sc: evaluate(
        [ScoredDoc(qid, d, float(s)) for qid in qids
         for d, s in zip(docids[qid], t5_raw[qid])], qrels, qs)
        for sc, qs in scopes}

    base_fixed = {sc: evaluate(
        [ScoredDoc(qid, d, float(s)) for qid in qids
         for d, s in zip(docids[qid],
                         0.975 * t5_norm[qid] + 0.025 * bm25_n[qid])], qrels, qs)
        for sc, qs in scopes}

    print("\nBaselines (nDCG@10):")
    print(f"  {'scope':<8}  {'pure_monot5':>12}  {'fixed_α=0.975':>14}")
    for sc, _ in scopes:
        print(f"  {sc:<8}  {base_pure[sc]['ndcg@10']:>12.4f}  "
              f"{base_fixed[sc]['ndcg@10']:>14.4f}")

    # Pre-compute per-query resolution for each tau (alpha chosen by stopping rule)
    # This is fast: at most 10 alpha steps per query
    print(f"\nSweeping {len(TAUS)} tau values …")
    sweep_rows = []

    for tau in tqdm(TAUS, desc="tau"):
        run = []
        alpha_used:  dict[str, float] = {}
        steps_taken: dict[str, int]   = {}

        for qid in qids:
            scores, alpha, steps = resolve_query(
                t5_raw[qid], t5_norm[qid], bm25_n[qid], tau
            )
            alpha_used[qid]  = alpha
            steps_taken[qid] = steps
            for d, s in zip(docids[qid], scores):
                run.append(ScoredDoc(qid, d, float(s)))

        n_fused   = sum(1 for a in alpha_used.values() if a < 1.0)
        n_never   = sum(1 for s in steps_taken.values() if s == -1)
        mean_alpha = float(np.mean(list(alpha_used.values())))

        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            fused_sc  = sum(1 for q in qs if alpha_used[q] < 1.0)
            never_sc  = sum(1 for q in qs if steps_taken[q] == -1)
            malpha_sc = float(np.mean([alpha_used[q] for q in qs]))
            msteps_sc = float(np.mean([steps_taken[q] for q in qs
                                       if steps_taken[q] >= 0]))
            sweep_rows.append({
                "tau":          float(tau),
                "scope":        sc,
                "n_fused":      fused_sc,
                "pct_fused":    round(100.0 * fused_sc / len(qs), 1),
                "n_never_certain": never_sc,
                "mean_alpha":   round(malpha_sc, 4),
                "mean_steps":   round(msteps_sc, 2),
                **m,
            })

    df = pd.DataFrame(sweep_rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    def best_row(scope: str, metric: str) -> dict:
        sub = df[df["scope"] == scope]
        return sub.loc[sub[metric].idxmax()].to_dict()

    print("\n" + "=" * 120)
    print("BEST TAU — entropy-driven fusion (H@20 gate + stopping criterion)")
    print(f"Alphas tried: {list(ALPHAS)}")
    print("=" * 120)
    print(f"  {'scope':<8}  {'tau*':>6}  {'%fused':>7}  {'%never':>7}  "
          f"{'mean_α':>7}  {'mean_steps':>11}  "
          + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δpure_'+m:>11}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δfixed_'+m:>11}" for m in METRIC_NAMES))

    results: dict[str, dict] = {}
    for sc, qs in scopes:
        target = TYPE_TARGETS[sc]
        b  = best_row(sc, target)
        bv = base_pure[sc]
        bf = base_fixed[sc]
        vals   = "  ".join(f"{b[m]:>8.4f}" for m in METRIC_NAMES)
        dpure  = "  ".join(f"{b[m]-bv[m]:>+11.4f}" for m in METRIC_NAMES)
        dfixed = "  ".join(f"{b[m]-bf[m]:>+11.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {b['tau']:>6.3f}  {b['pct_fused']:>6.1f}%  "
              f"{b['n_never_certain']:>6.0f}q  "
              f"{b['mean_alpha']:>7.4f}  {b['mean_steps']:>11.2f}  "
              f"{vals}  {dpure}  {dfixed}")
        results[sc] = {
            "tau_star":       float(b["tau"]),
            "pct_fused":      float(b["pct_fused"]),
            "n_never_certain": int(b["n_never_certain"]),
            "mean_alpha":     float(b["mean_alpha"]),
            "mean_steps":     float(b["mean_steps"]),
            "target_metric":  target,
            "metrics":        {m: float(b[m]) for m in METRIC_NAMES},
            "baseline_pure":  {m: float(bv[m]) for m in METRIC_NAMES},
            "baseline_fixed": {m: float(bf[m]) for m in METRIC_NAMES},
            "delta_vs_pure":  {m: float(b[m] - bv[m]) for m in METRIC_NAMES},
            "delta_vs_fixed": {m: float(b[m] - bf[m]) for m in METRIC_NAMES},
        }

    # Global tau curve — show nDCG@10 + % fused + mean alpha at each tau
    print("\n--- Global nDCG@10 vs tau ---")
    sub_g = df[df["scope"] == "global"].sort_values("tau")
    best_g = results["global"]["tau_star"]
    for _, row in sub_g[sub_g["tau"].isin(
            np.round(np.linspace(0, 1, 21), 2))].iterrows():
        marker = " ◄ best" if abs(row["tau"] - best_g) < 0.001 else ""
        print(f"  tau={row['tau']:.2f}  {row['pct_fused']:>5.1f}% fused  "
              f"mean_α={row['mean_alpha']:.4f}  "
              f"ndcg@10={row['ndcg@10']:.4f}{marker}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # --- Plot 1: tau curves per scope ---
    n = len(scopes)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    pal = {"ndcg@1": "#e377c2", "ndcg@3": "#4c72b0",
           "ndcg@5": "#2ca02c", "ndcg@10": "#1f77b4"}

    for col, (sc, qs) in enumerate(scopes):
        sub  = df[df["scope"] == sc].sort_values("tau")
        target   = TYPE_TARGETS[sc]
        tau_star = results[sc]["tau_star"]
        pct      = results[sc]["pct_fused"]

        # top: nDCG curves
        ax = axes[0][col]
        for m in METRIC_NAMES:
            lw = 2.4 if m == target else 1.2
            ax.plot(sub["tau"], sub[m], lw=lw, color=pal[m], label=m)
            ax.axhline(base_pure[sc][m],  color=pal[m], ls=":",  lw=0.9, alpha=0.5)
            ax.axhline(base_fixed[sc][m], color=pal[m], ls="--", lw=0.9, alpha=0.5)
        ax.axvline(tau_star, color="red", lw=1.8, ls="-",
                   label=f"τ*={tau_star:.3f}  ({pct:.0f}% fused)")
        ax.set_xlabel("τ (entropy threshold)")
        ax.set_ylabel("nDCG")
        ax.set_title(f"{sc}  target={target}", fontweight="bold", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")

        # bottom: mean_alpha and % fused
        ax2 = axes[1][col]
        ax2.plot(sub["tau"], sub["mean_alpha"], color="#1f77b4", lw=2.0,
                 label="mean α used")
        ax2.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax2.set_ylabel("mean α", color="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#1f77b4")

        ax3 = ax2.twinx()
        ax3.fill_between(sub["tau"], sub["pct_fused"], alpha=0.15, color="#ff7f0e")
        ax3.plot(sub["tau"], sub["pct_fused"], color="#ff7f0e", lw=1.5,
                 label="% fused")
        ax3.set_ylabel("% queries fused", color="#ff7f0e")
        ax3.tick_params(axis="y", labelcolor="#ff7f0e")
        ax3.set_ylim(0, 110)

        ax2.axvline(tau_star, color="red", lw=1.8, ls="-")
        ax2.set_xlabel("τ")
        ax2.set_title(f"{sc} — mean α & % fused vs τ", fontsize=9)

    fig.suptitle(
        "Entropy-driven adaptive fusion (H@20 gate + stopping criterion)\n"
        "Gate: fuse if H@20 > τ.  Stop: decrease α until H@20(fused) ≤ τ.\n"
        "monoT5 + BM25 — Task13BGoldenEnriched test set",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "entropy_fusion_test.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
