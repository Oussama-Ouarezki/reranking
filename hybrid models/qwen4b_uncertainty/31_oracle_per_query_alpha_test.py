"""Oracle per-query best alpha conditioned on H@20 gate — monoT5, test set.

For each tau (H@20 threshold):
  - Certain queries (H@20 <= tau)  → pure monoT5 scores
  - Uncertain queries (H@20 > tau) → sweep alpha in {1.0, 0.95, 0.90, ..., 0.50}
                                      pick the alpha that maximises nDCG@10 per query

This is an oracle upper bound: it shows how much performance is left on the table
if we could select alpha optimally per uncertain query, as a function of tau.

Also reports:
  - which alpha each query chose (distribution)
  - global aggregate across all queries
  - per query-type breakdown

Reads:  qwen4b_uncertainty/data/monot5_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/oracle_per_query_alpha_test.tsv
        qwen4b_uncertainty/data/oracle_per_query_alpha_best_test.json
        qwen4b_uncertainty/plots/oracle_per_query_alpha_test.png
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
OUT_TSV  = BASE / "qwen4b_uncertainty/data/oracle_per_query_alpha_test.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/oracle_per_query_alpha_best_test.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES        = ["summary", "factoid", "list", "yesno"]
TAUS         = np.round(np.linspace(0.0, 1.0, 21), 2)          # 0.00, 0.05, …, 1.00
ALPHAS       = np.round(np.arange(1.0, 0.45, -0.05), 2)        # 1.00, 0.95, 0.90, …, 0.50

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


def ndcg10_single(docids: list[str], scores: np.ndarray,
                  qid: str, qrels: list[Qrel]) -> float:
    if not qrels:
        return 0.0
    run = [ScoredDoc(qid, d, float(s)) for d, s in zip(docids, scores)]
    try:
        res = ir_measures.calc_aggregate([nDCG @ 10], qrels, run)
        return float(res[nDCG @ 10])
    except Exception:
        return 0.0


def evaluate(run: list[ScoredDoc], qrels: list[Qrel], qid_set: set[str]) -> dict[str, float]:
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

    # Load qrels into per-query dict
    all_qrels: list[Qrel] = []
    qrels_map: dict[str, list[Qrel]] = defaultdict(list)
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                q = Qrel(p[0], p[1], int(p[2]))
                all_qrels.append(q)
                qrels_map[p[0]].append(q)
    print(f"{len(all_qrels)} qrel rows")

    monot5_raw:  dict[str, np.ndarray] = {}
    monot5_norm: dict[str, np.ndarray] = {}
    bm25_norm:   dict[str, np.ndarray] = {}
    docids:      dict[str, list[str]]  = {}
    qtypes:      dict[str, str]        = {}
    H20:         dict[str, float]      = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = r["scores"]
        q = np.array([s["monot5_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"]  for s in items], dtype=float)
        monot5_raw[qid]  = q
        monot5_norm[qid] = minmax(q)
        bm25_norm[qid]   = minmax(b)
        docids[qid]      = [s["docid"] for s in items]
        H20[qid]         = norm_entropy_top20(q)

    qids = list(monot5_raw.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # ── Pre-compute oracle best alpha per query (regardless of tau) ────────────
    # For each query, find which alpha maximises nDCG@10 individually
    print(f"\nPre-computing oracle best alpha per query "
          f"({len(qids)} queries × {len(ALPHAS)} alphas) …")
    oracle_alpha: dict[str, float] = {}   # best alpha per query
    oracle_ndcg:  dict[str, float] = {}   # nDCG@10 at oracle alpha
    pure_ndcg:    dict[str, float] = {}   # nDCG@10 at pure monoT5 (alpha=1.0)

    for qid in tqdm(qids, unit="q"):
        qr = qrels_map.get(qid, [])
        best_val   = -1.0
        best_alpha = 1.0
        for alpha in ALPHAS:
            if alpha == 1.0:
                scores = monot5_raw[qid]
            else:
                scores = alpha * monot5_norm[qid] + (1 - alpha) * bm25_norm[qid]
            v = ndcg10_single(docids[qid], scores, qid, qr)
            if v > best_val:
                best_val   = v
                best_alpha = float(alpha)
        oracle_alpha[qid] = best_alpha
        oracle_ndcg[qid]  = best_val
        pure_ndcg[qid]    = ndcg10_single(docids[qid], monot5_raw[qid], qid, qr)

    # Distribution of oracle alphas
    from collections import Counter
    alpha_dist = Counter(oracle_alpha.values())
    print("\nOracle best-alpha distribution (all queries):")
    for a in sorted(alpha_dist.keys(), reverse=True):
        bar = "█" * alpha_dist[a]
        pct = 100 * alpha_dist[a] / len(qids)
        print(f"  α={a:.2f}  {alpha_dist[a]:>3} queries ({pct:>5.1f}%)  {bar}")

    # Overall oracle ceiling vs pure monoT5
    oracle_mean   = np.mean(list(oracle_ndcg.values()))
    pure_mean     = np.mean(list(pure_ndcg.values()))
    print(f"\nOracle ceiling nDCG@10 (per-query best alpha): {oracle_mean:.4f}")
    print(f"Pure monoT5 nDCG@10:                           {pure_mean:.4f}")
    print(f"Gap (oracle - pure):                           {oracle_mean - pure_mean:+.4f}")

    # ── Sweep tau: gate + oracle alpha for uncertain queries ───────────────────
    print(f"\nSweeping {len(TAUS)} tau values …")
    sweep_rows = []
    for tau in tqdm(TAUS, desc="tau"):
        run = []
        for qid in qids:
            if H20[qid] > tau:
                # uncertain → use oracle best alpha for this query
                a = oracle_alpha[qid]
                if a == 1.0:
                    scores = monot5_raw[qid]
                else:
                    scores = a * monot5_norm[qid] + (1 - a) * bm25_norm[qid]
            else:
                # certain → pure monoT5
                scores = monot5_raw[qid]
            for d, s in zip(docids[qid], scores):
                run.append(ScoredDoc(qid, d, float(s)))

        n_uncertain = sum(1 for q in qids if H20[q] > tau)
        for sc, qs in scopes:
            m = evaluate(run, all_qrels, qs)
            n_unc_sc = sum(1 for q in qs if H20[q] > tau)
            mean_alpha_unc = (np.mean([oracle_alpha[q] for q in qs if H20[q] > tau])
                              if n_unc_sc > 0 else 1.0)
            sweep_rows.append({
                "tau":           float(tau),
                "scope":         sc,
                "n_uncertain":   n_unc_sc,
                "pct_uncertain": round(100.0 * n_unc_sc / len(qs), 1),
                "mean_oracle_alpha_uncertain": round(float(mean_alpha_unc), 4),
                **m,
            })

    df = pd.DataFrame(sweep_rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    # ── Reference baselines ────────────────────────────────────────────────────
    base_pure = {sc: evaluate(
        [ScoredDoc(qid, d, float(s)) for qid in qids
         for d, s in zip(docids[qid], monot5_raw[qid])], all_qrels, qs)
        for sc, qs in scopes}

    # Best fixed alpha from script 27
    best_fixed_alpha = {"global": 0.975, "summary": 0.875,
                        "factoid": 1.000, "list": 0.975, "yesno": 0.950}
    base_fixed = {}
    for sc, qs in scopes:
        a = best_fixed_alpha.get(sc, 0.975)
        run = [ScoredDoc(qid, d, float(s)) for qid in qids
               for d, s in zip(docids[qid],
                                a * monot5_norm[qid] + (1 - a) * bm25_norm[qid])]
        base_fixed[sc] = evaluate(run, all_qrels, qs)

    def best_row(scope: str, metric: str) -> dict:
        sub = df[df["scope"] == scope]
        return sub.loc[sub[metric].idxmax()].to_dict()

    print("\n" + "=" * 115)
    print("ORACLE GATE: best tau per scope")
    print(f"Alphas tried per uncertain query: {list(ALPHAS)}")
    print("=" * 115)
    print(f"  {'scope':<8}  {'tau*':>5}  {'%unc':>6}  {'mean_α_unc':>11}  "
          + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δpure_'+m:>11}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δfixed_'+m:>11}" for m in METRIC_NAMES))

    results: dict[str, dict] = {}
    for sc, _ in scopes:
        target = TYPE_TARGETS[sc]
        b  = best_row(sc, target)
        bv = base_pure[sc]
        bf = base_fixed[sc]
        vals   = "  ".join(f"{b[m]:>8.4f}" for m in METRIC_NAMES)
        dpure  = "  ".join(f"{b[m]-bv[m]:>+11.4f}" for m in METRIC_NAMES)
        dfixed = "  ".join(f"{b[m]-bf[m]:>+11.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {b['tau']:>5.2f}  {b['pct_uncertain']:>5.1f}%  "
              f"{b['mean_oracle_alpha_uncertain']:>11.4f}  {vals}  {dpure}  {dfixed}")
        results[sc] = {
            "tau_star":     float(b["tau"]),
            "pct_uncertain": float(b["pct_uncertain"]),
            "mean_oracle_alpha_uncertain": float(b["mean_oracle_alpha_uncertain"]),
            "target_metric": target,
            "metrics":      {m: float(b[m]) for m in METRIC_NAMES},
            "baseline_pure":  {m: float(bv[m]) for m in METRIC_NAMES},
            "baseline_fixed": {m: float(bf[m]) for m in METRIC_NAMES},
            "delta_vs_pure":  {m: float(b[m] - bv[m]) for m in METRIC_NAMES},
            "delta_vs_fixed": {m: float(b[m] - bf[m]) for m in METRIC_NAMES},
        }
        # Per-type oracle alpha distribution
        qs_set = type_qids.get(sc, set(qids))
        unc_qs = [q for q in qs_set if H20[q] > b["tau"]]
        if unc_qs:
            dist = Counter(oracle_alpha[q] for q in unc_qs)
            alpha_line = "  ".join(
                f"α={a:.2f}:{dist[a]}" for a in sorted(dist, reverse=True)
            )
            print(f"           └─ alpha dist for uncertain: {alpha_line}")

    # Global tau curve
    print("\n--- Global nDCG@10 at oracle gate vs tau ---")
    sub_g = df[df["scope"] == "global"].sort_values("tau")
    for _, row in sub_g.iterrows():
        bar = "█" * int(row["ndcg@10"] * 200 - 174)
        print(f"  tau={row['tau']:.2f}  {row['pct_uncertain']:>5.1f}% unc  "
              f"ndcg@10={row['ndcg@10']:.4f}  {bar}")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    n = len(scopes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    pal = {"ndcg@1": "#e377c2", "ndcg@3": "#4c72b0",
           "ndcg@5": "#2ca02c", "ndcg@10": "#1f77b4"}

    for ax, (sc, qs) in zip(axes, scopes):
        sub = df[df["scope"] == sc].sort_values("tau")
        target = TYPE_TARGETS[sc]
        for m in METRIC_NAMES:
            lw = 2.4 if m == target else 1.2
            ax.plot(sub["tau"], sub[m], lw=lw, color=pal[m], label=m)
            ax.axhline(base_pure[sc][m],  color=pal[m], ls=":",  lw=0.9, alpha=0.6)
            ax.axhline(base_fixed[sc][m], color=pal[m], ls="--", lw=0.9, alpha=0.6)

        tau_star = results[sc]["tau_star"]
        pct      = results[sc]["pct_uncertain"]
        ax.axvline(tau_star, color="red", lw=1.8, ls="-",
                   label=f"τ*={tau_star:.2f}  ({pct:.0f}% uncertain)")

        ax2 = ax.twinx()
        ax2.fill_between(sub["tau"], sub["pct_uncertain"], alpha=0.07, color="gray")
        ax2.plot(sub["tau"], sub["pct_uncertain"], color="gray", lw=1.0, alpha=0.5)
        ax2.set_ylabel("% uncertain", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
        ax2.set_ylim(0, 110)

        ax.set_xlabel("τ  (H@20 gate threshold)")
        ax.set_ylabel("nDCG")
        ax.set_title(
            f"{sc}  (target: {target})\n"
            f"Dotted=pure monoT5  Dashed=best fixed α  Red=τ*",
            fontweight="bold", fontsize=9,
        )
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        f"Oracle per-query alpha (gap=0.05, range {ALPHAS[-1]:.2f}–{ALPHAS[0]:.2f}) "
        f"with H@20 gate\n"
        "monoT5 + BM25 — Task13BGoldenEnriched test set",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "oracle_per_query_alpha_test.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
