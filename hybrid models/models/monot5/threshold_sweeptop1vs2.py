"""Threshold sweep — monoT5 gating for duoT5.

Gating signal: Δ(q) = P(true|d1) − P(true|d2)  (top-1 vs top-2 monoT5 score gap).

For each threshold τ:
  Δ(q) < τ  →  apply duoT5  (use mono_duo ranking)
  Δ(q) ≥ τ  →  trust monoT5 (keep monoT5 ranking)

Metrics for each τ come from already-computed per-query run files — no re-running.

Sweeps:
  • τ grid  → nDCG@10, MRR@10, MAP@10, % queries sent to duoT5
  • λ grid  → J(τ) = nDCG(τ) − λ·cost(τ)  with optimal τ* per λ
  • Pareto  → (cost%, nDCG@10) frontier

Usage (from project root):
  python models/monot5/threshold_sweep.py \\
      --mono  application/cache/runs/monot5/20260427T095643Z.json \\
      --duo   application/cache/runs/mono_duo/20260427T144541Z.json \\
      --cache models/monot5/margin_scores_cache.json
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models/monot5"

# τ grid — calibrated to the gap distribution (p50≈0.001, p90≈0.05)
THRESHOLDS: list[float] = [
    0.0,
    0.0001, 0.0002, 0.0005,
    0.001,  0.002,  0.005,
    0.01,   0.02,   0.05,
    0.1,    0.2,    0.5,    1.0,
    math.inf,   # always apply duoT5
]

LAMBDAS: list[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

# Uncertainty threshold grid — p*(1-p) ∈ [0, 0.25]; send to duoT5 if mau(q) >= τ_unc
UNCERTAINTY_THRESHOLDS: list[float] = [
    0.0,                                   # always duo
    0.01, 0.02, 0.03, 0.05,
    0.07, 0.10, 0.12, 0.15,
    0.18, 0.20, 0.22, 0.25,
    math.inf,                              # always mono
]

METRIC   = "ndcg_at"   # primary metric key in per-query dict
METRIC_K = "10"        # @K


# ── loaders ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def get_metric(pq_entry: dict[str, Any], key: str, k: str) -> float:
    return float(pq_entry.get("metrics", {}).get(key, {}).get(k, 0.0))


# ── uncertainty ───────────────────────────────────────────────────────────────

def compute_mau(
    qids:        list[str],
    mono_pq:     dict[str, Any],
    score_cache: dict[str, dict[str, float]],
    k: int = 10,
) -> dict[str, float]:
    """Mean Average Uncertainty@k per query.

    Uncertainty of a monoT5 score p = P(true|d) is the Bernoulli variance
    p*(1-p), maximised at p=0.5 (completely uncertain) and 0 at extremes.
    MAU@k = mean over the top-k retrieved docs for a query.
    """
    uncertainties: dict[str, float] = {}
    for qid in qids:
        if qid not in mono_pq or qid not in score_cache:
            continue
        top_k   = mono_pq[qid].get("top_docids", [])[:k]
        scores  = score_cache[qid]
        vals    = [scores[d] * (1.0 - scores[d]) for d in top_k if d in scores]
        if vals:
            uncertainties[qid] = sum(vals) / len(vals)
    return uncertainties


# ── simulation ────────────────────────────────────────────────────────────────

def simulate(
    qids:      list[str],
    gaps:      dict[str, float],
    mono_pq:   dict[str, Any],
    duo_pq:    dict[str, Any],
    tau:       float,
) -> dict[str, float]:
    """Return aggregate metrics for the mixed system at threshold τ."""
    ndcg_vals, mrr_vals, map_vals = [], [], []
    n_duo = 0

    for qid in qids:
        use_duo = qid in gaps and gaps[qid] < tau
        src = duo_pq if use_duo else mono_pq
        entry = src.get(qid) or mono_pq.get(qid)
        if entry is None:
            continue
        ndcg_vals.append(get_metric(entry, "ndcg_at", "10"))
        mrr_vals.append(get_metric(entry, "mrr_at",  "10"))
        map_vals.append(get_metric(entry, "map_at",  "10"))
        if use_duo:
            n_duo += 1

    n = len(ndcg_vals)
    return {
        "ndcg10":    sum(ndcg_vals) / n if n else 0.0,
        "mrr10":     sum(mrr_vals)  / n if n else 0.0,
        "map10":     sum(map_vals)  / n if n else 0.0,
        "pct_duo":   n_duo / len(qids) * 100 if qids else 0.0,
        "n_duo":     n_duo,
        "n_queries": len(qids),
    }


def simulate_uncertainty(
    qids:          list[str],
    uncertainties: dict[str, float],
    mono_pq:       dict[str, Any],
    duo_pq:        dict[str, Any],
    tau_unc:       float,
) -> dict[str, float]:
    """Return aggregate metrics gating on MAU@10 >= tau_unc → duoT5."""
    ndcg_vals, mrr_vals, map_vals = [], [], []
    n_duo = 0

    for qid in qids:
        use_duo = qid in uncertainties and uncertainties[qid] >= tau_unc
        src     = duo_pq if use_duo else mono_pq
        entry   = src.get(qid) or mono_pq.get(qid)
        if entry is None:
            continue
        ndcg_vals.append(get_metric(entry, "ndcg_at", "10"))
        mrr_vals.append(get_metric(entry, "mrr_at",  "10"))
        map_vals.append(get_metric(entry, "map_at",  "10"))
        if use_duo:
            n_duo += 1

    n = len(ndcg_vals)
    return {
        "ndcg10":    sum(ndcg_vals) / n if n else 0.0,
        "mrr10":     sum(mrr_vals)  / n if n else 0.0,
        "map10":     sum(map_vals)  / n if n else 0.0,
        "pct_duo":   n_duo / len(qids) * 100 if qids else 0.0,
        "n_duo":     n_duo,
        "n_queries": len(qids),
    }


# ── knee detection ────────────────────────────────────────────────────────────

def knee_index(costs: list[float], ndcgs: list[float]) -> int:
    """Index of the point farthest from the line connecting endpoints."""
    if len(costs) < 3:
        return 0
    x1, y1 = costs[0],  ndcgs[0]
    x2, y2 = costs[-1], ndcgs[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.sqrt(dx*dx + dy*dy)
    if norm == 0:
        return 0
    dists = [abs(dy*(costs[i]-x1) - dx*(ndcgs[i]-y1)) / norm
             for i in range(len(costs))]
    return int(max(range(len(dists)), key=lambda i: dists[i]))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono",  required=True)
    parser.add_argument("--duo",   required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()

    mono_run  = load_json(Path(args.mono))
    duo_run   = load_json(Path(args.duo))
    score_cache: dict[str, dict[str, float]] = load_json(Path(args.cache))

    mono_pq: dict[str, Any] = mono_run["per_query"]
    duo_pq:  dict[str, Any] = duo_run["per_query"]

    # ── compute Δ(q) for every cached query ───────────────────────────────────
    gaps: dict[str, float] = {}
    for qid, doc_scores in score_cache.items():
        if qid not in mono_pq:
            continue
        top2 = mono_pq[qid]["top_docids"][:2]
        if len(top2) == 2 and top2[0] in doc_scores and top2[1] in doc_scores:
            gaps[qid] = doc_scores[top2[0]] - doc_scores[top2[1]]

    # all queries present in both runs
    qids = sorted(set(mono_pq) & set(duo_pq))
    n_with_gap = sum(1 for q in qids if q in gaps)
    print(f"Queries: {len(qids)} shared  |  {n_with_gap} with computable gap  "
          f"|  {len(qids)-n_with_gap} defaulting to monoT5\n")

    # ── gap distribution ──────────────────────────────────────────────────────
    gap_vals = sorted(gaps[q] for q in qids if q in gaps)
    print(f"Gap Δ(q) = top-1 − top-2 monoT5 P(true):")
    for p in [0, 10, 25, 50, 75, 90, 95, 100]:
        idx = min(int(p / 100 * len(gap_vals)), len(gap_vals) - 1)
        print(f"  p{p:3d}: {gap_vals[idx]:.5f}")
    print()

    # ── MAU@10 — mean average uncertainty in top-10 ───────────────────────────
    uncertainties = compute_mau(qids, mono_pq, score_cache)
    n_with_mau = len(uncertainties)
    mau_vals = sorted(uncertainties[q] for q in qids if q in uncertainties)
    mean_mau = sum(mau_vals) / len(mau_vals) if mau_vals else 0.0

    print(f"MAU@10  =  mean p·(1−p) over top-10 docs per query  "
          f"[{n_with_mau}/{len(qids)} queries]")
    print(f"  Mean MAU@10 across all queries: {mean_mau:.5f}")
    for p in [0, 10, 25, 50, 75, 90, 95, 100]:
        idx = min(int(p / 100 * len(mau_vals)), len(mau_vals) - 1)
        print(f"  p{p:3d}: {mau_vals[idx]:.5f}")
    print()

    # ── uncertainty threshold sweep ───────────────────────────────────────────
    tau_unc_label = lambda t: "0 (always duo)" if t == 0.0 else \
                              "∞ (always mono)" if math.isinf(t) else f"{t:.4f}"

    unc_sweep: list[dict[str, Any]] = []
    for tau_unc in UNCERTAINTY_THRESHOLDS:
        res = simulate_uncertainty(qids, uncertainties, mono_pq, duo_pq, tau_unc)
        res["tau_unc"] = tau_unc
        unc_sweep.append(res)

    print(f"Uncertainty sweep  (send to duoT5 if MAU@10 ≥ τ_unc):")
    print(f"{'τ_unc':>16}  {'nDCG@10':>8}  {'MRR@10':>8}  {'MAP@10':>8}  "
          f"{'% duo':>7}  {'n duo':>6}")
    print("─" * 64)
    for r in unc_sweep:
        print(f"{tau_unc_label(r['tau_unc']):>16}  {r['ndcg10']:>8.4f}  "
              f"{r['mrr10']:>8.4f}  {r['map10']:>8.4f}  "
              f"{r['pct_duo']:>6.1f}%  {r['n_duo']:>6}")
    print()

    # knee on uncertainty Pareto
    unc_costs = [r["pct_duo"] for r in unc_sweep]
    unc_ndcgs = [r["ndcg10"]  for r in unc_sweep]
    unc_ki    = knee_index(unc_costs, unc_ndcgs)
    unc_knee  = unc_sweep[unc_ki]
    print(f"Uncertainty Pareto knee: τ_unc = {tau_unc_label(unc_knee['tau_unc'])}  "
          f"→ nDCG@10 = {unc_knee['ndcg10']:.4f}  |  % duo = {unc_knee['pct_duo']:.1f}%\n")

    # ── τ sweep ───────────────────────────────────────────────────────────────
    sweep: list[dict[str, Any]] = []
    for tau in THRESHOLDS:
        res = simulate(qids, gaps, mono_pq, duo_pq, tau)
        res["tau"] = tau
        sweep.append(res)

    # ── print sweep table ─────────────────────────────────────────────────────
    tau_label = lambda t: "∞ (always duo)" if math.isinf(t) else f"{t:.4f}"
    print(f"{'τ':>16}  {'nDCG@10':>8}  {'MRR@10':>8}  {'MAP@10':>8}  "
          f"{'% duo':>7}  {'n duo':>6}")
    print("─" * 64)
    for r in sweep:
        print(f"{tau_label(r['tau']):>16}  {r['ndcg10']:>8.4f}  {r['mrr10']:>8.4f}  "
              f"{r['map10']:>8.4f}  {r['pct_duo']:>6.1f}%  {r['n_duo']:>6}")
    print()

    # ── λ sweep: optimal τ* per λ ─────────────────────────────────────────────
    print(f"Weighted objective  J(τ) = nDCG@10(τ) − λ·cost(τ)   [cost = % duo / 100]\n")
    print(f"{'λ':>6}  {'τ*':>16}  {'J(τ*)':>8}  {'nDCG@10':>8}  {'% duo':>7}")
    print("─" * 56)
    lambda_results: list[dict[str, Any]] = []
    for lam in LAMBDAS:
        best = max(sweep, key=lambda r: r["ndcg10"] - lam * r["pct_duo"] / 100)
        j    = best["ndcg10"] - lam * best["pct_duo"] / 100
        print(f"{lam:>6.1f}  {tau_label(best['tau']):>16}  {j:>8.4f}  "
              f"{best['ndcg10']:>8.4f}  {best['pct_duo']:>6.1f}%")
        lambda_results.append({"lambda": lam, **best})
    print()

    # ── knee point ────────────────────────────────────────────────────────────
    costs = [r["pct_duo"] for r in sweep]
    ndcgs = [r["ndcg10"]  for r in sweep]
    ki    = knee_index(costs, ndcgs)
    knee  = sweep[ki]
    print(f"Pareto knee point: τ = {tau_label(knee['tau'])}  "
          f"→ nDCG@10 = {knee['ndcg10']:.4f}  |  % duo = {knee['pct_duo']:.1f}%\n")

    # ── per-qtype breakdown at knee τ ─────────────────────────────────────────
    qtypes = sorted({mono_pq[q].get("qtype", "?") for q in qids})
    print(f"Per-qtype nDCG@10 at knee τ = {tau_label(knee['tau'])}:")
    print(f"  {'qtype':<10}  {'monoT5':>8}  {'mixed':>8}  {'mono_duo':>10}  {'n':>5}")
    print("  " + "─" * 44)
    for qt in qtypes:
        qt_qids = [q for q in qids if mono_pq[q].get("qtype") == qt]
        def mean_ndcg(pq: dict[str, Any]) -> float:
            vals = [get_metric(pq[q], "ndcg_at", "10") for q in qt_qids if q in pq]
            return sum(vals) / len(vals) if vals else 0.0
        mixed_vals = []
        for q in qt_qids:
            use_duo = q in gaps and gaps[q] < knee["tau"]
            src = duo_pq if use_duo else mono_pq
            entry = src.get(q) or mono_pq.get(q)
            if entry:
                mixed_vals.append(get_metric(entry, "ndcg_at", "10"))
        mixed_mean = sum(mixed_vals) / len(mixed_vals) if mixed_vals else 0.0
        print(f"  {qt:<10}  {mean_ndcg(mono_pq):>8.4f}  {mixed_mean:>8.4f}  "
              f"{mean_ndcg(duo_pq):>10.4f}  {len(qt_qids):>5}")
    print()

    # ── save CSV ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "threshold_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tau", "ndcg10", "mrr10", "map10",
                                           "pct_duo", "n_duo", "n_queries"])
        w.writeheader()
        for r in sweep:
            row = dict(r)
            row["tau"] = "inf" if math.isinf(r["tau"]) else r["tau"]
            w.writerow(row)
    print(f"CSV → {csv_path}")

    # ── charts ────────────────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid")
    plt.style.use("ggplot")

    # finite τ for plotting x-axis
    finite = [r for r in sweep if not math.isinf(r["tau"])]
    tau_x  = [r["tau"] for r in finite]
    ndcg_y = [r["ndcg10"] for r in finite]
    cost_y = [r["pct_duo"] for r in finite]

    # reference lines: pure monoT5 (τ=0) and pure mono_duo (τ=∞)
    mono_ndcg = sweep[0]["ndcg10"]
    duo_ndcg  = sweep[-1]["ndcg10"]

    # Chart 1 — nDCG@10 + % duo vs τ (dual axis, log x)
    fig1: Any
    ax1: Any
    ax1b: Any
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    ax1b = ax1.twinx()

    ax1.plot(tau_x, ndcg_y, color="#4c72b0", marker="o", linewidth=2, label="nDCG@10")
    ax1b.plot(tau_x, cost_y, color="#dd8452", marker="s", linewidth=2,
              linestyle="--", label="% queries → duoT5")

    # mark knee
    if not math.isinf(knee["tau"]):
        ax1.axvline(knee["tau"], color="green", linestyle=":", linewidth=1.5,
                    label=f"knee τ={knee['tau']:.4f}")

    ax1.axhline(mono_ndcg, color="#4c72b0", linestyle=":", linewidth=1, alpha=0.5,
                label=f"pure monoT5 ({mono_ndcg:.4f})")
    ax1.axhline(duo_ndcg,  color="red",     linestyle=":", linewidth=1, alpha=0.5,
                label=f"pure mono_duo ({duo_ndcg:.4f})")

    ax1.set_xscale("symlog", linthresh=0.0001)
    ax1.set_xlabel("Threshold τ  (log scale)")
    ax1.set_ylabel("nDCG@10")
    ax1b.set_ylabel("% queries sent to duoT5")
    ax1b.set_ylim(0, 110)
    ax1.set_title(f"Threshold sweep — monoT5 gating for duoT5  ({len(qids)} queries)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
    fig1.tight_layout()
    p1 = OUT_DIR / "threshold_sweep_plot.png"
    fig1.savefig(p1, dpi=150)
    print(f"Plot 1 → {p1}")

    # Chart 2 — Pareto curve: % duo (cost) vs nDCG@10
    fig2: Any
    ax2: Any
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.plot(costs, ndcgs, color="#4c72b0", marker="o", linewidth=2)

    # annotate a few key τ labels
    label_taus = {0.0, 0.001, 0.005, 0.01, 0.05, 0.1, math.inf}
    for r in sweep:
        if r["tau"] in label_taus:
            ax2.annotate(
                tau_label(r["tau"]),
                (r["pct_duo"], r["ndcg10"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )

    # mark knee
    ax2.scatter([knee["pct_duo"]], [knee["ndcg10"]], color="green", zorder=5,
                s=80, label=f"knee  τ={tau_label(knee['tau'])}")

    ax2.set_xlabel("% queries sent to duoT5  (cost)")
    ax2.set_ylabel("nDCG@10")
    ax2.set_title("Pareto frontier — effectiveness vs reranking cost")
    ax2.legend()
    fig2.tight_layout()
    p2 = OUT_DIR / "threshold_pareto_plot.png"
    fig2.savefig(p2, dpi=150)
    print(f"Plot 2 → {p2}")

    # Chart 3 — J(τ) = nDCG(τ) − λ·cost(τ) for each λ
    fig3: Any
    ax3: Any
    fig3, ax3 = plt.subplots(figsize=(11, 5))
    colors = sns.color_palette("tab10", len(LAMBDAS))
    for lam, col in zip(LAMBDAS, colors):
        j_vals = [r["ndcg10"] - lam * r["pct_duo"] / 100 for r in finite]
        ax3.plot(tau_x, j_vals, marker="o", linewidth=1.5,
                 color=col, label=f"λ={lam}")
        # mark optimal τ*
        best_i = max(range(len(j_vals)), key=lambda i: j_vals[i])
        ax3.scatter([tau_x[best_i]], [j_vals[best_i]], color=col, zorder=5, s=60)

    ax3.set_xscale("symlog", linthresh=0.0001)
    ax3.set_xlabel("Threshold τ  (log scale)")
    ax3.set_ylabel("J(τ) = nDCG@10 − λ·cost")
    ax3.set_title("Weighted objective J(τ) for each λ  (dot = optimal τ*)")
    ax3.legend(fontsize=8)
    fig3.tight_layout()
    p3 = OUT_DIR / "threshold_objective_plot.png"
    fig3.savefig(p3, dpi=150)
    print(f"Plot 3 → {p3}")

    # Chart 4 — uncertainty sweep: nDCG@10 + % duo vs τ_unc
    fig4: Any
    ax4: Any
    ax4b: Any
    finite_unc = [r for r in unc_sweep
                  if not math.isinf(r["tau_unc"]) and r["tau_unc"] != 0.0]
    tau_unc_x  = [r["tau_unc"] for r in finite_unc]
    ndcg_unc_y = [r["ndcg10"]  for r in finite_unc]
    cost_unc_y = [r["pct_duo"] for r in finite_unc]

    fig4, ax4 = plt.subplots(figsize=(11, 5))
    ax4b = ax4.twinx()

    ax4.plot(tau_unc_x, ndcg_unc_y, color="#4c72b0", marker="o",
             linewidth=2, label="nDCG@10")
    ax4b.plot(tau_unc_x, cost_unc_y, color="#dd8452", marker="s",
              linewidth=2, linestyle="--", label="% queries → duoT5")

    if not math.isinf(unc_knee["tau_unc"]) and unc_knee["tau_unc"] != 0.0:
        ax4.axvline(unc_knee["tau_unc"], color="green", linestyle=":",
                    linewidth=1.5,
                    label=f"knee τ_unc={unc_knee['tau_unc']:.4f}")

    ax4.axhline(mono_ndcg, color="#4c72b0", linestyle=":", linewidth=1, alpha=0.5,
                label=f"pure monoT5 ({mono_ndcg:.4f})")
    ax4.axhline(duo_ndcg,  color="red",     linestyle=":", linewidth=1, alpha=0.5,
                label=f"pure mono_duo ({duo_ndcg:.4f})")

    # annotate mean MAU line
    ax4.axvline(mean_mau, color="purple", linestyle="-.", linewidth=1.2, alpha=0.7,
                label=f"mean MAU@10 ({mean_mau:.4f})")

    ax4.set_xlabel("Uncertainty threshold τ_unc  (MAU@10 ≥ τ_unc → duoT5)")
    ax4.set_ylabel("nDCG@10")
    ax4b.set_ylabel("% queries sent to duoT5")
    ax4b.set_ylim(0, 110)
    ax4.set_title(f"Uncertainty sweep — MAU@10 gating for duoT5  "
                  f"(mean MAU@10 = {mean_mau:.4f},  {len(qids)} queries)")

    lines4,  labels4  = ax4.get_legend_handles_labels()
    lines4b, labels4b = ax4b.get_legend_handles_labels()
    ax4.legend(lines4 + lines4b, labels4 + labels4b, loc="lower right", fontsize=8)
    fig4.tight_layout()
    p4 = OUT_DIR / "threshold_uncertainty_plot.png"
    fig4.savefig(p4, dpi=150)
    print(f"Plot 4 → {p4}")

    plt.show()


if __name__ == "__main__":
    main()
