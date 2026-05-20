"""Fixed-alpha + H@20 entropy gate tau sweep — Task13BGoldenEnriched test set.

Alpha is fixed to the per-type best from Stage 1 (script 23).
Only tau is swept to find the best entropy gate threshold per type.

    if H@20(query) > tau:
        score(d) = alpha_type * qwen_norm(d) + (1-alpha_type) * bm25_norm(d)
    else:
        score(d) = qwen_prob(d)

Reads:  qwen4b_uncertainty/data/qwen_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
        qwen4b_uncertainty/data/alpha_best_test.json
Writes: qwen4b_uncertainty/data/fixed_alpha_tau_test.tsv
        qwen4b_uncertainty/data/fixed_alpha_tau_best_test.json
        qwen4b_uncertainty/plots/fixed_alpha_tau_test.png
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
SCORES_F    = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
QRELS_F     = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
ALPHA_F     = BASE / "qwen4b_uncertainty/data/alpha_best_test.json"
OUT_TSV     = BASE / "qwen4b_uncertainty/data/fixed_alpha_tau_test.tsv"
OUT_JSON    = BASE / "qwen4b_uncertainty/data/fixed_alpha_tau_best_test.json"
PLOTS       = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES  = ["summary", "factoid", "list", "yesno"]
TAUS   = np.round(np.linspace(0.0, 1.0, 101), 3)   # finer grid: step 0.01

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
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"{len(rows)} queries loaded")

    qrels = load_qrels()
    print(f"{len(qrels)} qrel rows")

    alpha_cfg = json.loads(ALPHA_F.read_text())
    # per-type fixed alphas (use global alpha for global scope)
    fixed_alpha: dict[str, float] = {
        sc: alpha_cfg[sc]["alpha_star"] for sc in ["global"] + TYPES
    }
    print("\nFixed alphas (from Stage 1):")
    for sc, a in fixed_alpha.items():
        print(f"  {sc:<8}  alpha={a:.3f}  (target: {TYPE_TARGETS[sc]})")

    qwen_raw:  dict[str, np.ndarray] = {}
    qwen_norm: dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    docids:    dict[str, list[str]]  = {}
    qtypes:    dict[str, str]        = {}
    H20:       dict[str, float]      = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = r["scores"]
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_raw[qid]  = q
        qwen_norm[qid] = minmax(q)
        bm25_norm[qid] = minmax(b)
        docids[qid]    = [s["docid"] for s in items]
        H20[qid]       = norm_entropy_top20(q)

    qids = list(qwen_raw.keys())
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # Pure-Qwen baseline
    base_run = [ScoredDoc(qid, d, float(s))
                for qid in qids
                for d, s in zip(docids[qid], qwen_raw[qid])]
    base = {sc: evaluate(base_run, qrels, qs) for sc, qs in scopes}

    # Stage-1 reference (always fuse with fixed alpha)
    s1_run_by_scope: dict[str, list[ScoredDoc]] = {}
    for sc, qs in scopes:
        alpha = fixed_alpha[sc]
        run = []
        for qid in qids:
            scores = alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid]
            for d, s in zip(docids[qid], scores):
                run.append(ScoredDoc(qid, d, float(s)))
        s1_run_by_scope[sc] = run
    s1 = {sc: evaluate(s1_run_by_scope[sc], qrels, qs) for sc, qs in scopes}

    print("\nBaselines:")
    print(f"  {'scope':<8}  {'alpha':>6}  "
          + "  ".join(f"{'qwen_'+m:>12}" for m in METRIC_NAMES)
          + "  "
          + "  ".join(f"{'s1_'+m:>11}" for m in METRIC_NAMES))
    for sc, _ in scopes:
        qv = "  ".join(f"{base[sc][m]:>12.4f}" for m in METRIC_NAMES)
        sv = "  ".join(f"{s1[sc][m]:>11.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {fixed_alpha[sc]:>6.3f}  {qv}  {sv}")

    # Tau sweep — each scope uses its own fixed alpha
    print(f"\nSweeping {len(TAUS)} tau values …")
    sweep_rows = []
    for tau in tqdm(TAUS, desc="tau"):
        n_fused_global = sum(1 for q in qids if H20[q] > tau)
        for sc, qs in scopes:
            alpha = fixed_alpha[sc]
            run = []
            for qid in qids:
                if H20[qid] > tau:
                    scores = alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid]
                else:
                    scores = qwen_raw[qid]
                for d, s in zip(docids[qid], scores):
                    run.append(ScoredDoc(qid, d, float(s)))
            m = evaluate(run, qrels, qs)
            n_fused_sc = sum(1 for q in qs if H20[q] > tau)
            sweep_rows.append({
                "scope": sc,
                "alpha": alpha,
                "tau": float(tau),
                "n_fused": n_fused_sc,
                "pct_fused": round(100.0 * n_fused_sc / len(qs), 1),
                **m,
            })

    df = pd.DataFrame(sweep_rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    def best(scope: str, metric: str) -> dict:
        sub = df[df["scope"] == scope]
        return sub.loc[sub[metric].idxmax()].to_dict()

    print("\n" + "=" * 110)
    print("BEST TAU PER SCOPE (fixed alpha from Stage 1)")
    print("=" * 110)
    hdr = (f"  {'scope':<8}  {'alpha':>6}  {'tau*':>6}  {'%fused':>7}  "
           + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
           + "  " + "  ".join(f"{'Δqwen_'+m:>11}" for m in METRIC_NAMES)
           + "  " + "  ".join(f"{'Δs1_'+m:>9}" for m in METRIC_NAMES))
    print(hdr)

    results: dict[str, dict] = {}
    for sc, qs in scopes:
        target = TYPE_TARGETS[sc]
        b  = best(sc, target)
        bv = base[sc]
        sv = s1[sc]
        vals  = "  ".join(f"{b[m]:>8.4f}" for m in METRIC_NAMES)
        dqwen = "  ".join(f"{b[m]-bv[m]:>+11.4f}" for m in METRIC_NAMES)
        ds1   = "  ".join(f"{b[m]-sv[m]:>+9.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {fixed_alpha[sc]:>6.3f}  {b['tau']:>6.3f}  "
              f"{b['pct_fused']:>6.1f}%  {vals}  {dqwen}  {ds1}")
        results[sc] = {
            "alpha":         fixed_alpha[sc],
            "tau_star":      float(b["tau"]),
            "pct_fused":     float(b["pct_fused"]),
            "target_metric": target,
            "metrics":       {m: float(b[m]) for m in METRIC_NAMES},
            "baseline_qwen": {m: float(bv[m]) for m in METRIC_NAMES},
            "baseline_s1":   {m: float(sv[m]) for m in METRIC_NAMES},
            "delta_vs_qwen": {m: float(b[m] - bv[m]) for m in METRIC_NAMES},
            "delta_vs_s1":   {m: float(b[m] - sv[m]) for m in METRIC_NAMES},
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # --- Plot: tau curves per scope, target metric highlighted ---
    n = len(scopes)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=False)
    pal = {"ndcg@1": "#e377c2", "ndcg@3": "#4c72b0",
           "ndcg@5": "#2ca02c", "ndcg@10": "#1f77b4"}

    for ax, (sc, qs) in zip(axes, scopes):
        sub = df[df["scope"] == sc].sort_values("tau")
        for m in METRIC_NAMES:
            lw = 2.4 if m == TYPE_TARGETS[sc] else 1.2
            ax.plot(sub["tau"], sub[m], lw=lw, color=pal[m], label=m)
            ax.axhline(base[sc][m], color=pal[m], ls=":", lw=0.8, alpha=0.5)
            ax.axhline(s1[sc][m],   color=pal[m], ls="--", lw=0.8, alpha=0.7)

        target = TYPE_TARGETS[sc]
        tau_star = results[sc]["tau_star"]
        pct = results[sc]["pct_fused"]
        ax.axvline(tau_star, color="red", lw=1.8, ls="-",
                   label=f"τ*={tau_star:.3f}  ({pct:.0f}% fused)")

        # secondary axis: % fused
        ax2 = ax.twinx()
        ax2.fill_between(sub["tau"], sub["pct_fused"], alpha=0.08, color="gray")
        ax2.plot(sub["tau"], sub["pct_fused"], color="gray", lw=1.0, ls="-", alpha=0.5)
        ax2.set_ylabel("% queries fused", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
        ax2.set_ylim(0, 110)

        ax.set_xlabel("τ  (H@20 threshold)")
        ax.set_ylabel("nDCG")
        ax.set_title(
            f"{sc}  α={fixed_alpha[sc]:.3f}  (target: {target})\n"
            f"Dotted=pure Qwen  Dashed=always-fuse  Red=τ*",
            fontweight="bold", fontsize=9,
        )
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "Fixed-alpha entropy gate: tau sweep per query type\n"
        "Task13BGoldenEnriched test set — H@20 normalised entropy",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "fixed_alpha_tau_test.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
