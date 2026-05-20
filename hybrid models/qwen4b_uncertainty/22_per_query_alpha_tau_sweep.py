"""Oracle per-query alpha + tau sweep on the Task13BGoldenEnriched test set.

For each query independently, finds (alpha*, tau*) that maximise nDCG@10:

  If H50(query) > tau: score(d) = alpha*qwen_norm(d) + (1-alpha)*bm25_norm(d)
  Else:                score(d) = qwen_prob(d)

Sweeps:
  alpha in [0.0 .. 1.0], 41 steps
  tau   in [0.0 .. 1.0], 41 steps  (normalised entropy of top-50 probs)

Reads:  qwen4b_uncertainty/data/qwen_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/per_query_best_test.tsv
        qwen4b_uncertainty/data/per_query_summary_test.tsv
        qwen4b_uncertainty/plots/per_query_oracle_test.png
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
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_PER_Q  = BASE / "qwen4b_uncertainty/data/per_query_best_test.tsv"
OUT_SUMMARY = BASE / "qwen4b_uncertainty/data/per_query_summary_test.tsv"
PLOTS = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES = ["summary", "factoid", "list", "yesno"]
ALPHAS = np.round(np.linspace(0.0, 1.0, 41), 3)
TAUS   = np.round(np.linspace(0.0, 1.0, 41), 3)
METRIC = nDCG @ 10
METRIC_NAME = "ndcg@10"


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def norm_entropy(vals: np.ndarray) -> float:
    s = vals.sum()
    if s <= 0 or len(vals) < 2:
        return 0.0
    p = vals / s
    p = np.clip(p, 1e-15, 1.0)
    return float(-(p * np.log(p)).sum() / math.log(len(vals)))


def load_qrels() -> dict[str, list[Qrel]]:
    qrels: dict[str, list[Qrel]] = defaultdict(list)
    with QRELS_F.open() as f:
        next(f)  # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels[p[0]].append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def ndcg10_single(run_docs: list[tuple[str, float]], qrels: list[Qrel]) -> float:
    if not run_docs or not qrels:
        return 0.0
    qid = qrels[0].query_id
    scored = [ScoredDoc(qid, d, s) for d, s in run_docs]
    try:
        res = ir_measures.calc_aggregate([METRIC], qrels, scored)
        return float(res[METRIC])
    except Exception:
        return 0.0


def main() -> None:
    rows = []
    with SCORES_F.open() as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"{len(rows)} queries loaded")

    qrels_map = load_qrels()
    print(f"{sum(len(v) for v in qrels_map.values())} qrel rows")

    # Pre-compute per-query arrays
    qwen_arr:  dict[str, np.ndarray] = {}
    qwen_norm: dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    docids:    dict[str, list[str]]  = {}
    qtypes:    dict[str, str]        = {}
    H50:       dict[str, float]      = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = r["scores"]  # keep BM25-retrieval order
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_arr[qid]  = q
        qwen_norm[qid] = minmax(q)
        bm25_norm[qid] = minmax(b)
        docids[qid]    = [s["docid"] for s in items]
        H50[qid]       = norm_entropy(q[:50])

    qids = list(qwen_arr.keys())

    # Pure-Qwen baseline per query
    baseline: dict[str, float] = {}
    for qid in qids:
        run = list(zip(docids[qid], qwen_arr[qid].tolist()))
        baseline[qid] = ndcg10_single(run, qrels_map.get(qid, []))

    print(f"\nPure-Qwen mean nDCG@10: {np.mean(list(baseline.values())):.4f}")

    # Per-query oracle sweep
    print(f"\nSweeping {len(ALPHAS)} alphas x {len(TAUS)} taus for {len(qids)} queries …")
    records = []

    for qid in tqdm(qids, unit="q"):
        qrel = qrels_map.get(qid, [])
        h    = H50[qid]
        best_val  = -1.0
        best_alpha = 1.0
        best_tau   = float("inf")

        for tau in TAUS:
            fuse = h > tau
            for alpha in ALPHAS:
                if fuse:
                    scores = alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid]
                else:
                    scores = qwen_arr[qid]
                v = ndcg10_single(list(zip(docids[qid], scores.tolist())), qrel)
                if v > best_val:
                    best_val   = v
                    best_alpha = float(alpha)
                    best_tau   = float(tau)

        records.append({
            "qid":       qid,
            "type":      qtypes[qid],
            "H50":       round(h, 4),
            "best_alpha": best_alpha,
            "best_tau":   best_tau,
            "oracle_ndcg10": round(best_val, 6),
            "baseline_ndcg10": round(baseline[qid], 6),
            "delta": round(best_val - baseline[qid], 6),
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT_PER_Q, sep="\t", index=False)
    print(f"wrote {OUT_PER_Q}")

    # Summary by type + global
    type_qids: dict[str, set[str]] = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)

    summary_rows = []
    for scope in ["global"] + TYPES:
        mask = df["type"] == scope if scope != "global" else pd.Series([True] * len(df))
        sub = df[mask]
        if sub.empty:
            continue
        summary_rows.append({
            "scope": scope,
            "n_queries": len(sub),
            "baseline_ndcg10_mean": round(sub["baseline_ndcg10"].mean(), 4),
            "oracle_ndcg10_mean":   round(sub["oracle_ndcg10"].mean(), 4),
            "delta_mean":           round(sub["delta"].mean(), 4),
            "alpha_mean":           round(sub["best_alpha"].mean(), 4),
            "alpha_std":            round(sub["best_alpha"].std(), 4),
            "tau_mean":             round(sub["best_tau"].mean(), 4),
            "tau_std":              round(sub["best_tau"].std(), 4),
            "H50_mean":             round(sub["H50"].mean(), 4),
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT_SUMMARY, sep="\t", index=False)
    print(f"wrote {OUT_SUMMARY}")
    print()
    print(df_sum.to_string(index=False))

    # --- Plots ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Oracle vs baseline nDCG@10 per query type
    ax = axes[0][0]
    x = np.arange(len(df_sum))
    w = 0.35
    ax.bar(x - w/2, df_sum["baseline_ndcg10_mean"], w, label="Pure Qwen baseline")
    ax.bar(x + w/2, df_sum["oracle_ndcg10_mean"],   w, label="Oracle (best α,τ per query)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_sum["scope"], rotation=15)
    ax.set_ylabel("nDCG@10")
    ax.set_title("Oracle vs Baseline nDCG@10", fontweight="bold")
    ax.legend(fontsize=9)

    # 2. Distribution of best alpha per type
    ax = axes[0][1]
    for t in TYPES:
        sub = df[df["type"] == t]["best_alpha"]
        ax.hist(sub, bins=20, alpha=0.5, label=t, density=True)
    ax.set_xlabel("Best α")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Oracle α per type", fontweight="bold")
    ax.legend(fontsize=9)

    # 3. Distribution of best tau per type
    ax = axes[0][2]
    for t in TYPES:
        sub = df[df["type"] == t]["best_tau"]
        ax.hist(sub, bins=20, alpha=0.5, label=t, density=True)
    ax.set_xlabel("Best τ")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Oracle τ per type", fontweight="bold")
    ax.legend(fontsize=9)

    # 4. Delta nDCG@10 (oracle - baseline) scatter vs H50
    ax = axes[1][0]
    palette = {"summary": "#1f77b4", "factoid": "#ff7f0e", "list": "#2ca02c", "yesno": "#d62728"}
    for t in TYPES:
        sub = df[df["type"] == t]
        ax.scatter(sub["H50"], sub["delta"], alpha=0.4, s=20,
                   color=palette.get(t, "gray"), label=t)
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.set_xlabel("H50 (normalised entropy of Qwen probs)")
    ax.set_ylabel("Δ nDCG@10 (oracle − baseline)")
    ax.set_title("Oracle gain vs query entropy", fontweight="bold")
    ax.legend(fontsize=9)

    # 5. Best alpha vs H50
    ax = axes[1][1]
    for t in TYPES:
        sub = df[df["type"] == t]
        ax.scatter(sub["H50"], sub["best_alpha"], alpha=0.4, s=20,
                   color=palette.get(t, "gray"), label=t)
    ax.set_xlabel("H50")
    ax.set_ylabel("Oracle α")
    ax.set_title("Oracle α vs query entropy", fontweight="bold")
    ax.legend(fontsize=9)

    # 6. Best alpha vs best tau
    ax = axes[1][2]
    for t in TYPES:
        sub = df[df["type"] == t]
        ax.scatter(sub["best_alpha"], sub["best_tau"], alpha=0.4, s=20,
                   color=palette.get(t, "gray"), label=t)
    ax.set_xlabel("Oracle α")
    ax.set_ylabel("Oracle τ")
    ax.set_title("Oracle α vs Oracle τ per query", fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle(
        "Per-query oracle (α*, τ*) sweep — Task13BGoldenEnriched test set",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "per_query_oracle_test.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
