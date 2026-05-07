"""Best linear-fusion alpha per query type — Task13BGoldenEnriched test set.

No gating. For every query:
    score(d) = alpha * qwen_norm(d) + (1-alpha) * bm25_norm(d)

Sweeps alpha in [0.0 .. 1.0] (41 steps) and finds, for each scope
(global + per type), the alpha* that maximises the type's target metric:
    list    -> nDCG@3
    summary -> nDCG@10
    yesno   -> nDCG@1
    factoid -> nDCG@5
    global  -> nDCG@10

Reads:  qwen4b_uncertainty/data/qwen_scores_test.jsonl
        data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes: qwen4b_uncertainty/data/alpha_sweep_test.tsv
        qwen4b_uncertainty/data/alpha_best_test.json
        qwen4b_uncertainty/plots/alpha_sweep_test.png
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
from ir_measures import nDCG, Qrel, ScoredDoc

sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

BASE = Path(__file__).resolve().parents[1]
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
QRELS_F  = BASE / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV  = BASE / "qwen4b_uncertainty/data/alpha_sweep_test.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/alpha_best_test.json"
PLOTS    = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES  = ["summary", "factoid", "list", "yesno"]
ALPHAS = np.round(np.linspace(0.0, 1.0, 41), 3)

METRICS = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]

TYPE_TARGETS = {
    "list":    "ndcg@3",
    "summary": "ndcg@10",
    "yesno":   "ndcg@1",
    "factoid": "ndcg@5",
    "global":  "ndcg@10",
}


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


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

    qwen_norm: dict[str, np.ndarray] = {}
    bm25_norm: dict[str, np.ndarray] = {}
    qwen_raw:  dict[str, np.ndarray] = {}
    docids:    dict[str, list[str]]  = {}
    qtypes:    dict[str, str]        = {}

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

    print("\nPure Qwen baseline:")
    print(f"  {'scope':<8}  " + "  ".join(f"{m:>8}" for m in METRIC_NAMES))
    for sc, _ in scopes:
        vals = "  ".join(f"{base[sc][m]:>8.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {vals}")

    # Alpha sweep
    print(f"\nSweeping {len(ALPHAS)} alpha values …")
    sweep_rows = []
    for alpha in tqdm(ALPHAS, desc="alpha"):
        run = [ScoredDoc(qid, d, float(s))
               for qid in qids
               for d, s in zip(
                   docids[qid],
                   alpha * qwen_norm[qid] + (1 - alpha) * bm25_norm[qid],
               )]
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            sweep_rows.append({"scope": sc, "alpha": float(alpha), **m})

    df = pd.DataFrame(sweep_rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    def best(scope: str, metric: str) -> dict:
        sub = df[df["scope"] == scope]
        return sub.loc[sub[metric].idxmax()].to_dict()

    print("\n" + "=" * 90)
    print("BEST ALPHA PER QUERY TYPE")
    print("=" * 90)
    print(f"  {'scope':<8}  {'target':<9}  {'alpha*':>7}  "
          + "  ".join(f"{m:>8}" for m in METRIC_NAMES)
          + "  " + "  ".join(f"{'Δ'+m:>9}" for m in METRIC_NAMES))

    results: dict[str, dict] = {}
    for sc, _ in scopes:
        target = TYPE_TARGETS[sc]
        b = best(sc, target)
        bv = base[sc]
        deltas = "  ".join(f"{b[m]-bv[m]:>+9.4f}" for m in METRIC_NAMES)
        vals   = "  ".join(f"{b[m]:>8.4f}" for m in METRIC_NAMES)
        print(f"  {sc:<8}  {target:<9}  {b['alpha']:>7.3f}  {vals}  {deltas}")
        results[sc] = {
            "alpha_star":   float(b["alpha"]),
            "target_metric": target,
            "metrics": {m: float(b[m]) for m in METRIC_NAMES},
            "baseline": {m: float(bv[m]) for m in METRIC_NAMES},
            "delta":    {m: float(b[m] - bv[m]) for m in METRIC_NAMES},
        }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # Plot: alpha vs each metric, one panel per scope
    n_scopes = len(scopes)
    fig, axes = plt.subplots(1, n_scopes, figsize=(5 * n_scopes, 5), sharey=False)
    palette = {"ndcg@1": "#e377c2", "ndcg@3": "#4c72b0",
               "ndcg@5": "#2ca02c", "ndcg@10": "#1f77b4"}

    for ax, (sc, _) in zip(axes, scopes):
        sub = df[df["scope"] == sc].sort_values("alpha")
        for m in METRIC_NAMES:
            ax.plot(sub["alpha"], sub[m], marker="o", ms=2.5, lw=1.6,
                    color=palette[m], label=m)
            ax.axhline(base[sc][m], color=palette[m], ls=":", lw=0.8, alpha=0.5)
        target = TYPE_TARGETS[sc]
        alpha_star = results[sc]["alpha_star"]
        ax.axvline(alpha_star, color="red", lw=1.6, ls="-",
                   label=f"α*={alpha_star:.3f}\n(target {target})")
        ax.set_xlabel("α  (Qwen weight)")
        ax.set_ylabel("nDCG")
        title = sc if sc == "global" else f"{sc}  (target: {target})"
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=7)

    fig.suptitle(
        "Linear fusion α sweep — Task13BGoldenEnriched test set\n"
        "Dotted lines = pure-Qwen baseline.  Red = chosen α* (per scope target metric).",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "alpha_sweep_test.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
