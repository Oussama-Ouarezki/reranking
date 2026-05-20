"""Gated dynamic linear fusion — confidence-margin signal instead of entropy.

Per-query confidence signal:
    conf = (p1 − p2) / |p1 + p2|
where p1, p2 are the top-1 and top-2 Qwen P(yes) over the 50 BM25 candidates.
High conf → Qwen is sure about the top doc.  Low conf → top-1 ≈ top-2 → uncertain.

Gate: when conf < τ → uncertain → fuse (linear, per-type α);
      when conf ≥ τ → confident → keep pure Qwen.

α per type is the mixed-target dynamic config (each type's natural metric):
    list α=0.875   summary α=0.800   yesno α=0.750   factoid α=0.875

Stage 1 — Global τ (target = global nDCG@10).
Stage 2 — Per-type τ, each at its target metric (list→@3, summary→@10,
          yesno→@1, factoid→@5).

Reads:  qwen4b_uncertainty/data/qwen_scores.jsonl
        data/bioasq/processed/qrels.tsv
Writes: qwen4b_uncertainty/data/gated_conf_grid.tsv
        qwen4b_uncertainty/data/gated_conf_params.json
        qwen4b_uncertainty/plots/gated_conf_global.png
        qwen4b_uncertainty/plots/gated_conf_per_type.png
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
SCORES_F = BASE / "qwen4b_uncertainty/data/qwen_scores.jsonl"
QRELS_F = BASE / "data/bioasq/processed/qrels.tsv"
OUT_TSV = BASE / "qwen4b_uncertainty/data/gated_conf_grid.tsv"
OUT_JSON = BASE / "qwen4b_uncertainty/data/gated_conf_params.json"
PLOTS = BASE / "qwen4b_uncertainty/plots"
PLOTS.mkdir(parents=True, exist_ok=True)

TYPES = ["summary", "factoid", "list", "yesno"]
METRICS = [nDCG @ 1, nDCG @ 3, nDCG @ 5, nDCG @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"]

ALPHA_BY_TYPE = {
    "list":    0.875,
    "summary": 0.800,
    "yesno":   0.750,
    "factoid": 0.875,
}
TYPE_TARGETS = {
    "list":    "ndcg@3",
    "summary": "ndcg@10",
    "yesno":   "ndcg@1",
    "factoid": "ndcg@5",
}
GLOBAL_TARGET = "ndcg@10"

# Confidence ∈ [0, 1] for sorted-descending Qwen P(yes); sweep finely near 0.
TAUS = np.round(np.concatenate([
    np.linspace(0.0, 0.05, 11),       # fine near 0 (most queries cluster low)
    np.linspace(0.06, 0.30, 13),
    np.linspace(0.35, 1.0, 14),
]), 5)
TAUS = np.unique(TAUS)


def confidence_margin(probs_sorted_desc: np.ndarray) -> float:
    if len(probs_sorted_desc) < 2:
        return 1.0
    p1, p2 = float(probs_sorted_desc[0]), float(probs_sorted_desc[1])
    denom = abs(p1 + p2)
    if denom < 1e-15:
        return 0.0
    return (p1 - p2) / denom


def minmax(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_qrels():
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def evaluate(run, qrels, qid_set):
    sub_run = [r for r in run if r.query_id in qid_set]
    sub_q = [q for q in qrels if q.query_id in qid_set]
    if not sub_run or not sub_q:
        return {n: float("nan") for n in METRIC_NAMES}
    res = ir_measures.calc_aggregate(METRICS, sub_q, sub_run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def main():
    rows = [json.loads(l) for l in SCORES_F.open()]
    qrels = load_qrels()
    print(f"{len(rows)} queries / {len(qrels)} qrel rows")

    qwen_arr, qwen_norm, bm25_norm, docids, qtypes = {}, {}, {}, {}, {}
    CONF: dict[str, float] = {}

    for r in rows:
        qid = r["qid"]
        qtypes[qid] = r["type"]
        items = sorted(r["scores"], key=lambda s: s["qwen_prob"], reverse=True)
        q = np.array([s["qwen_prob"] for s in items], dtype=float)
        b = np.array([s["bm25_score"] for s in items], dtype=float)
        qwen_arr[qid] = q
        qwen_norm[qid] = minmax(q)
        bm25_norm[qid] = minmax(b)
        docids[qid] = [s["docid"] for s in items]
        CONF[qid] = confidence_margin(q)

    qids = list(qwen_arr.keys())
    type_qids = defaultdict(set)
    for qid, t in qtypes.items():
        type_qids[t].add(qid)
    scopes = [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]

    # Confidence distribution stats (per scope)
    print("\nConfidence distribution (p1 − p2) / |p1 + p2|:")
    print(f"  {'scope':<8}  {'mean':>7}  {'median':>7}  "
          f"{'p10':>7}  {'p25':>7}  {'p75':>7}  {'p90':>7}")
    for sc, qs in [("global", set(qids))] + [(t, type_qids[t]) for t in TYPES]:
        vals = sorted(CONF[q] for q in qs)
        n = len(vals)
        pct = lambda p: vals[min(int(p / 100 * n), n - 1)]
        print(f"  {sc:<8}  {np.mean(vals):>7.4f}  {np.median(vals):>7.4f}  "
              f"{pct(10):>7.4f}  {pct(25):>7.4f}  {pct(75):>7.4f}  {pct(90):>7.4f}")

    # Baselines
    base_run = [ScoredDoc(qid, d, float(s)) for qid in qids
                for d, s in zip(docids[qid], qwen_arr[qid])]
    base = {sc: evaluate(base_run, qrels, qs) for sc, qs in scopes}

    fuse_always_run = []
    for qid in qids:
        a = ALPHA_BY_TYPE[qtypes[qid]]
        s = a * qwen_norm[qid] + (1 - a) * bm25_norm[qid]
        for d, sv in zip(docids[qid], s):
            fuse_always_run.append(ScoredDoc(qid, d, float(sv)))
    fuse_always = {sc: evaluate(fuse_always_run, qrels, qs) for sc, qs in scopes}

    # ── Sweep τ ────────────────────────────────────────────────────────────
    rows_out = []
    print(f"\nSweeping τ ({len(TAUS)} values) — gate: fuse when conf < τ …")
    for tau in tqdm(TAUS, desc="τ"):
        run = []
        n_fused_by_type = defaultdict(int)
        n_fused_global = 0
        for qid in qids:
            a = ALPHA_BY_TYPE[qtypes[qid]]
            if CONF[qid] < tau:
                s = a * qwen_norm[qid] + (1 - a) * bm25_norm[qid]
                n_fused_by_type[qtypes[qid]] += 1
                n_fused_global += 1
            else:
                s = qwen_arr[qid]
            for d, sv in zip(docids[qid], s):
                run.append(ScoredDoc(qid, d, float(sv)))
        for sc, qs in scopes:
            m = evaluate(run, qrels, qs)
            n_unc = n_fused_global if sc == "global" else n_fused_by_type[sc]
            denom = len(qids) if sc == "global" else len(type_qids[sc])
            rows_out.append({
                "tau": float(tau), "scope": sc,
                "n_fused": n_unc, "pct_fused": 100.0 * n_unc / denom,
                **m,
            })
    df = pd.DataFrame(rows_out)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"wrote {OUT_TSV}")

    # ── Stage 1: global τ ──────────────────────────────────────────────────
    sub_g = df[df["scope"] == "global"]
    s1_best = sub_g.loc[sub_g[GLOBAL_TARGET].idxmax()].to_dict()
    print("\n" + "=" * 90)
    print(f"STAGE 1 — Global τ (target = global {GLOBAL_TARGET})")
    print("=" * 90)
    print(f"  τ*={s1_best['tau']:.5f}  "
          f"{GLOBAL_TARGET}={s1_best[GLOBAL_TARGET]:.4f}  "
          f"(pure Qwen={base['global'][GLOBAL_TARGET]:.4f}, "
          f"fuse_always={fuse_always['global'][GLOBAL_TARGET]:.4f})  "
          f"%fused={s1_best['pct_fused']:.1f}%")

    # ── Stage 2: per-type τ ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("STAGE 2 — Per-type τ (each at its target metric)")
    print("=" * 90)
    s2: dict[str, dict] = {}
    for t in TYPES:
        target = TYPE_TARGETS[t]
        sub = df[df["scope"] == t]
        best = sub.loc[sub[target].idxmax()].to_dict()
        s2[t] = best
        print(f"  {t:<8}  target={target:<7}  τ*={best['tau']:.5f}  "
              f"value={best[target]:.4f}  "
              f"(pure Qwen={base[t][target]:.4f}, "
              f"fuse_always={fuse_always[t][target]:.4f})  "
              f"%fused={best['pct_fused']:.1f}%")

    # Save params
    params = {
        "signal":         "conf = (p1 - p2) / |p1 + p2|, gate fuse when conf < τ",
        "alpha_by_type":  ALPHA_BY_TYPE,
        "type_targets":   TYPE_TARGETS,
        "stage1_global":  {"tau": float(s1_best["tau"]),
                           "ndcg@10": float(s1_best[GLOBAL_TARGET]),
                           "pct_fused": float(s1_best["pct_fused"])},
        "stage2_per_type": {
            t: {
                "tau":           float(s2[t]["tau"]),
                "alpha":         ALPHA_BY_TYPE[t],
                "target_metric": TYPE_TARGETS[t],
                "metric_value":  float(s2[t][TYPE_TARGETS[t]]),
                "pct_fused":     float(s2[t]["pct_fused"]),
            }
            for t in TYPES
        },
    }
    OUT_JSON.write_text(json.dumps(params, indent=2))
    print(f"\nwrote {OUT_JSON}")

    # ── Plot 1: global τ — 4 metrics ───────────────────────────────────────
    fig, axes = plt.subplots(1, len(METRIC_NAMES), figsize=(6 * len(METRIC_NAMES), 5))
    for ax, m in zip(axes, METRIC_NAMES):
        sub = df[df["scope"] == "global"].sort_values("tau")
        ax.plot(sub["tau"], sub[m], marker="o", ms=3, lw=1.8,
                color="#1f77b4", label="gated")
        ax.axvline(s1_best["tau"], color="red", ls="--", lw=1.4,
                   label=f"τ*={s1_best['tau']:.4f}")
        ax.axhline(base["global"][m], color="gray", ls=":", lw=1.0,
                   label=f"pure Qwen ({base['global'][m]:.4f})")
        ax.axhline(fuse_always["global"][m], color="black", ls=":", lw=1.0,
                   label=f"fuse_always ({fuse_always['global'][m]:.4f})")
        title_marker = "  ★ Stage 1 target" if m == GLOBAL_TARGET else ""
        ax.set_title(f"Global — {m}{title_marker}", fontweight="bold")
        ax.set_xlabel("τ  (fuse when conf < τ)")
        ax.set_ylabel(m)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Stage 1 — Global τ on confidence margin",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = PLOTS / "gated_conf_global.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")

    # ── Plot 2: per-type τ ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(TYPES), figsize=(6 * len(TYPES), 5))
    for ax, t in zip(axes, TYPES):
        target = TYPE_TARGETS[t]
        sub = df[df["scope"] == t].sort_values("tau")
        for m in METRIC_NAMES:
            lw = 2.4 if m == target else 1.0
            alpha = 1.0 if m == target else 0.45
            ax.plot(sub["tau"], sub[m], marker="o", ms=2.5, lw=lw,
                    alpha=alpha, label=f"{m}{' (target)' if m == target else ''}")
        ax.axvline(s2[t]["tau"], color="red", ls="-", lw=1.5,
                   label=f"τ*={s2[t]['tau']:.4f}")
        ax.axhline(base[t][target], color="gray", ls=":", lw=1.0,
                   label=f"pure Qwen ({base[t][target]:.4f})")
        ax.axhline(fuse_always[t][target], color="black", ls=":", lw=1.0,
                   label=f"fuse_always ({fuse_always[t][target]:.4f})")
        ax.set_xlabel("τ")
        ax.set_ylabel(target)
        ax.set_title(f"{t}  (target {target}, α={ALPHA_BY_TYPE[t]:.3f})",
                     fontweight="bold")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Stage 2 — Per-type τ on confidence margin (each type at its target)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    p = PLOTS / "gated_conf_per_type.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"  → {p}")


if __name__ == "__main__":
    main()
