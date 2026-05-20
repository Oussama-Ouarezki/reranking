"""MonoGatedDuo grid search — compare gating signals for duoT5.

Signals compared
  GAP    — Δ(q) = P(true|rank1) − P(true|rank2)        send to duo when Δ < τ
  MAU@5  — mean p*(1−p) over top-5 monoT5 scores        send to duo when MAU ≥ τ
  MAU@10 — mean p*(1−p) over top-10 monoT5 scores       send to duo when MAU ≥ τ
  H@20   — norm. rank-distribution entropy over top-20   send to duo when H ≥ τ
  H@50   — norm. rank-distribution entropy over top-50   send to duo when H ≥ τ
  All entropy / GAP / MAU signals derived from the full 50-doc score pool.

Caches
  mono  cache  : application/cache/runs/monot5/20260427T171106Z.json  (nDCG@10=0.8728)
  duo   cache  : application/cache/runs/mono_duo/20260429T134302Z.json (nDCG@10=0.8982)
  score cache  : models/monot5/margin_scores_cache.json  (50 docs × P(true) per query)

Outputs
  results/gap_sweep.csv  results/mau5_sweep.csv  results/mau10_sweep.csv
  results/entropy20_sweep.csv  results/entropy50_sweep.csv  results/summary.txt
  plots/gap_sweep.png  plots/mau5_sweep.png  plots/mau10_sweep.png
  plots/entropy_sweep.png  plots/pareto_comparison.png  plots/signal_comparison.png
"""

import csv
import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

MONO_CACHE  = ROOT / "application/cache/runs/monot5/20260427T171106Z.json"
DUO_CACHE   = ROOT / "application/cache/runs/mono_duo/20260429T134302Z.json"
SCORE_CACHE = ROOT / "models/monot5/margin_scores_cache.json"

OUT_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = OUT_DIR / "results"
PLOTS_DIR   = OUT_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── theme ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── tau grids ─────────────────────────────────────────────────────────────────

# GAP: send to duo when gap < τ
GAP_GRID: list[float] = sorted({
    0.0,
    *[round(v, 6) for v in np.logspace(-5, -0.3, 60)],
    0.001, 0.002, 0.003, 0.005, 0.007,
    0.01,  0.02,  0.03,  0.05,  0.07,
    0.1,   0.15,  0.2,   0.3,   0.5,   0.8, 1.0,
    math.inf,
})

# MAU: send to duo when MAU ≥ τ
MAU_GRID: list[float] = sorted({
    0.0,
    *[round(v, 6) for v in np.linspace(0.001, 0.249, 80)],
    0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25,
    math.inf,
})

# Entropy: send to duo when H_norm ≥ τ
ENT_GRID: list[float] = sorted({
    0.0,
    *[round(v, 5) for v in np.linspace(0.05, 0.95, 100)],
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    math.inf,
})

ENTROPY_KS = [20, 50]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def get_metric(entry: dict, key: str, k: str) -> float:
    return float(entry.get("metrics", {}).get(key, {}).get(k, 0.0))


def knee_index(costs: list[float], ndcgs: list[float]) -> int:
    if len(costs) < 3:
        return 0
    x1, y1 = costs[0],  ndcgs[0]
    x2, y2 = costs[-1], ndcgs[-1]
    dx, dy = x2 - x1, y2 - y1
    norm = math.sqrt(dx * dx + dy * dy)
    if norm == 0:
        return 0
    dists = [abs(dy * (costs[i] - x1) - dx * (ndcgs[i] - y1)) / norm
             for i in range(len(costs))]
    return int(max(range(len(dists)), key=lambda i: dists[i]))


def _norm_entropy(vals: list[float]) -> float | None:
    if not vals or sum(vals) == 0:
        return None
    s = sum(vals)
    probs = [v / s for v in vals]
    h = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
    return h / math.log(len(probs))


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading caches …")
mono_run   = load_json(MONO_CACHE)
duo_run    = load_json(DUO_CACHE)
scores_raw = load_json(SCORE_CACHE)

mono_pq: dict = mono_run["per_query"]
duo_pq:  dict = duo_run["per_query"]

qids = sorted(set(mono_pq) & set(duo_pq))
print(f"  mono  cache : {len(mono_pq)} queries  nDCG@10={mono_run['aggregate']['ndcg_at']['10']:.4f}")
print(f"  duo   cache : {len(duo_pq)} queries  nDCG@10={duo_run['aggregate']['ndcg_at']['10']:.4f}")
print(f"  score cache : {len(scores_raw)} queries  (50 docs × P(true) each)")
print(f"  shared qids : {len(qids)}")

# ── compute per-query signals ─────────────────────────────────────────────────
# All signals derived from the full 50-doc score pool (sorted by P(true) desc).

gaps:           dict[str, float] = {}
mau5_scores:    dict[str, float] = {}
mau10_scores:   dict[str, float] = {}
entropy_scores: dict[int, dict[str, float]] = {k: {} for k in ENTROPY_KS}

for qid in qids:
    if qid not in scores_raw:
        continue
    doc_scores = scores_raw[qid]  # {docid: P(true)}, 50 docs

    # Sort all 50 docs by P(true) descending → full monoT5 ranking
    ranked = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)
    ranked_vals = [doc_scores[d] for d in ranked]

    # GAP: top-1 vs top-2
    if len(ranked_vals) >= 2:
        gaps[qid] = ranked_vals[0] - ranked_vals[1]

    # MAU@5 / MAU@10
    for k, store in [(5, mau5_scores), (10, mau10_scores)]:
        vals = [ranked_vals[i] * (1 - ranked_vals[i]) for i in range(min(k, len(ranked_vals)))]
        if vals:
            store[qid] = sum(vals) / len(vals)

    # Entropy @20 / @50  (from full 50-doc pool)
    for k in ENTROPY_KS:
        vals = ranked_vals[:k]
        h = _norm_entropy(vals)
        if h is not None:
            entropy_scores[k][qid] = h

n = len(qids)
print(f"\nSignal coverage (of {n} shared queries):")
print(f"  GAP   : {sum(1 for q in qids if q in gaps)}")
print(f"  MAU@5 : {sum(1 for q in qids if q in mau5_scores)}")
print(f"  MAU@10: {sum(1 for q in qids if q in mau10_scores)}")
for k in ENTROPY_KS:
    print(f"  H@{k:<3d} : {sum(1 for q in qids if q in entropy_scores[k])}")


def _pct_summary(vals: list[float], label: str) -> None:
    if not vals:
        return
    mean = sum(vals) / len(vals)
    print(f"\n{label}  (mean={mean:.4f}):")
    for pct in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        idx = min(int(pct / 100 * len(vals)), len(vals) - 1)
        print(f"  p{pct:3d}: {vals[idx]:.6f}")


_pct_summary(sorted(gaps.values()),         "GAP  (top-1 − top-2 P(true))")
_pct_summary(sorted(mau5_scores.values()),  "MAU@5")
_pct_summary(sorted(mau10_scores.values()), "MAU@10")
for k in ENTROPY_KS:
    _pct_summary(sorted(entropy_scores[k].values()), f"H@{k}  (normalised entropy)")

# ── simulation ────────────────────────────────────────────────────────────────

def _aggregate(entries: list[dict], flags: list[bool], tau_key: str, tau_val: float) -> dict:
    n1, n5, n10, mrr_vals, map_vals = [], [], [], [], []
    n_duo = sum(flags)
    for entry in entries:
        n1.append(get_metric(entry,  "ndcg_at", "1"))
        n5.append(get_metric(entry,  "ndcg_at", "5"))
        n10.append(get_metric(entry, "ndcg_at", "10"))
        mrr_vals.append(get_metric(entry, "mrr_at", "10"))
        map_vals.append(get_metric(entry, "map_at", "10"))
    nt = len(qids)
    return {
        tau_key:       tau_val,
        "ndcg1":       sum(n1)       / len(n1)       if n1  else 0.0,
        "ndcg5":       sum(n5)       / len(n5)       if n5  else 0.0,
        "ndcg10":      sum(n10)      / len(n10)      if n10 else 0.0,
        "mrr10":       sum(mrr_vals) / len(mrr_vals) if mrr_vals else 0.0,
        "map10":       sum(map_vals) / len(map_vals) if map_vals else 0.0,
        "pct_duo":     n_duo / nt * 100 if nt else 0.0,
        "n_duo":       n_duo,
        "n_saved":     nt - n_duo,
        "n_queries":   nt,
    }


def simulate_gap(tau: float) -> dict:
    entries, flags = [], []
    for qid in qids:
        use_duo = qid in gaps and gaps[qid] < tau
        entry = (duo_pq if use_duo else mono_pq).get(qid) or mono_pq.get(qid)
        if entry is None:
            continue
        entries.append(entry); flags.append(use_duo)
    return _aggregate(entries, flags, "tau", tau)


def simulate_mau(tau: float, score_dict: dict[str, float]) -> dict:
    entries, flags = [], []
    for qid in qids:
        use_duo = qid in score_dict and score_dict[qid] >= tau
        entry = (duo_pq if use_duo else mono_pq).get(qid) or mono_pq.get(qid)
        if entry is None:
            continue
        entries.append(entry); flags.append(use_duo)
    return _aggregate(entries, flags, "tau_unc", tau)


# ── run sweeps ────────────────────────────────────────────────────────────────

print("\nRunning GAP sweep …")
gap_sweep  = [simulate_gap(t)               for t in GAP_GRID]

print("Running MAU@5 sweep …")
mau5_sweep = [simulate_mau(t, mau5_scores)  for t in MAU_GRID]

print("Running MAU@10 sweep …")
mau10_sweep = [simulate_mau(t, mau10_scores) for t in MAU_GRID]

print("Running Entropy sweeps …")
ent_sweeps: dict[int, list[dict]] = {}
for k in ENTROPY_KS:
    print(f"  H@{k} …")
    ent_sweeps[k] = [simulate_mau(t, entropy_scores[k]) for t in ENT_GRID]

# ── baselines ─────────────────────────────────────────────────────────────────
mono_ndcg = gap_sweep[0]["ndcg10"];   mono_n1 = gap_sweep[0]["ndcg1"];  mono_n5 = gap_sweep[0]["ndcg5"]
duo_ndcg  = gap_sweep[-1]["ndcg10"];  duo_n1  = gap_sweep[-1]["ndcg1"]; duo_n5  = gap_sweep[-1]["ndcg5"]

print(f"\n  Baseline monoT5   nDCG@10={mono_ndcg:.4f}")
print(f"  Baseline mono_duo nDCG@10={duo_ndcg:.4f}")

# ── best and Pareto knee per signal ──────────────────────────────────────────

best_gap   = max(gap_sweep,   key=lambda r: r["ndcg10"])
best_mau5  = max(mau5_sweep,  key=lambda r: r["ndcg10"])
best_mau10 = max(mau10_sweep, key=lambda r: r["ndcg10"])
best_ent: dict[int, dict] = {k: max(ent_sweeps[k], key=lambda r: r["ndcg10"]) for k in ENTROPY_KS}


def _knee(sweep: list[dict], cost_key: str = "pct_duo") -> dict:
    costs = [r[cost_key] for r in sweep]
    ndcgs = [r["ndcg10"] for r in sweep]
    return sweep[knee_index(costs, ndcgs)]


gap_knee   = _knee(gap_sweep)
mau5_knee  = _knee(mau5_sweep)
mau10_knee = _knee(mau10_sweep)
ent_knees: dict[int, dict] = {k: _knee(ent_sweeps[k]) for k in ENTROPY_KS}


def fmt_tau(t: float, fmt: str = ".5f") -> str:
    return "∞ (always duo)" if math.isinf(t) else f"{t:{fmt}}"


# ── print tables ──────────────────────────────────────────────────────────────

HDR = (f"{'τ':>16}  {'nDCG@1':>7}  {'nDCG@5':>7}  {'nDCG@10':>8}  "
       f"{'MRR@10':>8}  {'%duo':>6}  {'n_duo':>6}  {'saved':>6}")
SEP = "─" * 80


def _row(r: dict, tau_k: str, best_r: dict, knee_r: dict) -> str:
    marker = " ◄ BEST" if r is best_r else (" ◄ KNEE" if r is knee_r else "")
    tv = r[tau_k]
    ts = "∞ (always duo)" if math.isinf(tv) else f"{tv:.5f}"
    return (f"{ts:>16}  {r['ndcg1']:>7.4f}  {r['ndcg5']:>7.4f}  "
            f"{r['ndcg10']:>8.4f}  {r['mrr10']:>8.4f}  {r['pct_duo']:>5.1f}%  "
            f"{r['n_duo']:>6}  {r['n_saved']:>6}{marker}")


def _print_sweep(title: str, sweep: list[dict], tau_k: str, best_r: dict, knee_r: dict):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    print(HDR); print(SEP)
    for r in sweep:
        print(_row(r, tau_k, best_r, knee_r))


_print_sweep("GAP SWEEP  (send to duo when Δ = top1−top2 < τ)",
             gap_sweep,   "tau",     best_gap,   gap_knee)
_print_sweep("MAU@5 SWEEP  (send to duo when MAU@5 ≥ τ)",
             mau5_sweep,  "tau_unc", best_mau5,  mau5_knee)
_print_sweep("MAU@10 SWEEP  (send to duo when MAU@10 ≥ τ)",
             mau10_sweep, "tau_unc", best_mau10, mau10_knee)
for k in ENTROPY_KS:
    _print_sweep(f"H@{k} SWEEP  (send to duo when H@{k} ≥ τ, normalised to [0,1])",
                 ent_sweeps[k], "tau_unc", best_ent[k], ent_knees[k])


# ── summary ───────────────────────────────────────────────────────────────────

def _sig_block(label: str, r_best: dict, r_knee: dict, tau_k: str) -> str:
    t_best = fmt_tau(r_best[tau_k])
    t_knee = fmt_tau(r_knee[tau_k])
    return (f"{'─'*80}\n{label}\n"
            f"  Best τ  : {t_best}\n"
            f"  Best    : @1={r_best['ndcg1']:.4f}  @5={r_best['ndcg5']:.4f}  @10={r_best['ndcg10']:.4f}\n"
            f"  → duo   : {r_best['n_duo']}/{n} ({r_best['pct_duo']:.1f}%)  "
            f"saved {r_best['n_saved']}q\n"
            f"  Knee τ  : {t_knee}\n"
            f"  Knee    : @1={r_knee['ndcg1']:.4f}  @5={r_knee['ndcg5']:.4f}  @10={r_knee['ndcg10']:.4f}\n"
            f"  → duo   : {r_knee['n_duo']}/{n} ({r_knee['pct_duo']:.1f}%)  "
            f"saved {r_knee['n_saved']}q")


all_best = {
    "GAP":   (best_gap,   "tau"),
    "MAU@5": (best_mau5,  "tau_unc"),
    "MAU@10":(best_mau10, "tau_unc"),
    **{f"H@{k}": (best_ent[k], "tau_unc") for k in ENTROPY_KS},
}
winner_k, (winner_r, winner_tau_k) = max(all_best.items(), key=lambda kv: kv[1][0]["ndcg10"])

all_knees = {
    "GAP":   (gap_knee,   "tau"),
    "MAU@5": (mau5_knee,  "tau_unc"),
    "MAU@10":(mau10_knee, "tau_unc"),
    **{f"H@{k}": (ent_knees[k], "tau_unc") for k in ENTROPY_KS},
}
kw_k, (kw_r, kw_tau) = max(all_knees.items(), key=lambda kv: kv[1][0]["ndcg10"])

summary = textwrap.dedent(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MonoGatedDuo Grid Search — Summary  (nDCG @1 / @5 / @10)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

Baselines                   nDCG@1    nDCG@5   nDCG@10    total queries
  monoT5  (never duo)      {mono_n1:.4f}    {mono_n5:.4f}    {mono_ndcg:.4f}    {n}
  mono_duo (always duo)    {duo_n1:.4f}    {duo_n5:.4f}    {duo_ndcg:.4f}
  Δ (duo − mono)          {duo_n1-mono_n1:+.4f}   {duo_n5-mono_n5:+.4f}   {duo_ndcg-mono_ndcg:+.4f}
""").strip() + "\n\n"

summary += _sig_block("GAP  (top-1 vs top-2 P(true) gap, send to duo when gap < τ)",
                      best_gap, gap_knee, "tau") + "\n\n"
summary += _sig_block("MAU@5  (mean Bernoulli variance over top-5)",
                      best_mau5, mau5_knee, "tau_unc") + "\n\n"
summary += _sig_block("MAU@10  (mean Bernoulli variance over top-10)",
                      best_mau10, mau10_knee, "tau_unc") + "\n\n"
for k in ENTROPY_KS:
    summary += _sig_block(f"H@{k}  (rank-distribution entropy, top-{k} of full 50-doc pool)",
                          best_ent[k], ent_knees[k], "tau_unc") + "\n\n"

summary += (f"{'─'*80}\n"
            f"WINNER (best nDCG@10): {winner_k}  τ={fmt_tau(winner_r[winner_tau_k])}\n"
            f"  nDCG  @1={winner_r['ndcg1']:.4f}  @5={winner_r['ndcg5']:.4f}  "
            f"@10={winner_r['ndcg10']:.4f}\n"
            f"  Δ vs monoT5: @1={winner_r['ndcg1']-mono_n1:+.4f}  "
            f"@5={winner_r['ndcg5']-mono_n5:+.4f}  @10={winner_r['ndcg10']-mono_ndcg:+.4f}\n\n"
            f"WINNER (Pareto knee): {kw_k}  τ={fmt_tau(kw_r[kw_tau])}\n"
            f"  nDCG  @1={kw_r['ndcg1']:.4f}  @5={kw_r['ndcg5']:.4f}  "
            f"@10={kw_r['ndcg10']:.4f}\n"
            f"  Saved {kw_r['n_saved']}q  ({100 - kw_r['pct_duo']:.1f}% cost reduction)\n")

print("\n" + summary)
(RESULTS_DIR / "summary.txt").write_text(summary + "\n")
print(f"Summary → {RESULTS_DIR / 'summary.txt'}")


# ── save CSVs ─────────────────────────────────────────────────────────────────

BASE_FIELDS = ["ndcg1", "ndcg5", "ndcg10", "mrr10", "map10",
               "pct_duo", "n_duo", "n_saved", "n_queries"]


def _write_csv(path: Path, sweep: list[dict], tau_key: str):
    fields = [tau_key] + BASE_FIELDS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sweep:
            row = dict(r)
            row[tau_key] = "inf" if math.isinf(r[tau_key]) else r[tau_key]
            w.writerow(row)
    print(f"CSV → {path}")


_write_csv(RESULTS_DIR / "gap_sweep.csv",     gap_sweep,   "tau")
_write_csv(RESULTS_DIR / "mau5_sweep.csv",    mau5_sweep,  "tau_unc")
_write_csv(RESULTS_DIR / "mau10_sweep.csv",   mau10_sweep, "tau_unc")
for k in ENTROPY_KS:
    _write_csv(RESULTS_DIR / f"entropy{k}_sweep.csv", ent_sweeps[k], "tau_unc")


# ── colours ───────────────────────────────────────────────────────────────────

BLUE   = "#4c72b0"
ORANGE = "#dd8452"
GREEN  = "#2ca02c"
RED    = "#d62728"
CYAN   = "#17becf"
PINK   = "#e377c2"

ENT_COLORS = {20: "#bcbd22", 50: "#8c564b"}


# ── helper: dual-axis sweep plot ──────────────────────────────────────────────

def _dual_axis_sweep(sweep: list[dict], tau_key: str, title: str,
                     best_r: dict, knee_r: dict, out_path: Path,
                     xlabel: str, is_log: bool = False):
    finite = [r for r in sweep if not math.isinf(r[tau_key]) and r[tau_key] > 0]
    tx   = [r[tau_key]   for r in finite]
    n1_  = [r["ndcg1"]   for r in finite]
    n5_  = [r["ndcg5"]   for r in finite]
    n10_ = [r["ndcg10"]  for r in finite]
    cost = [r["pct_duo"] for r in finite]

    fig, ax = plt.subplots(figsize=(12, 5))
    axr = ax.twinx()

    ax.plot(tx, n1_,  color=PINK,      ms=3, lw=1.8, marker="o", label="nDCG@1")
    ax.plot(tx, n5_,  color=BLUE,      ms=3, lw=1.8, marker="o", label="nDCG@5")
    ax.plot(tx, n10_, color="#1f77b4", ms=3, lw=2.2, marker="o", label="nDCG@10", alpha=0.9)
    axr.plot(tx, cost, color=ORANGE, ms=3, lw=2, ls="--", marker="s",
             label="% queries → duoT5")

    for val, bval, col in [(mono_n1, duo_n1, PINK), (mono_n5, duo_n5, BLUE),
                            (mono_ndcg, duo_ndcg, "#1f77b4")]:
        ax.axhline(val,  color=col, ls=":", lw=0.9, alpha=0.45)
        ax.axhline(bval, color=col, ls="-", lw=0.9, alpha=0.22)

    if best_r is not None and not math.isinf(best_r[tau_key]) and best_r[tau_key] > 0:
        ax.axvline(best_r[tau_key], color=GREEN, ls="-", lw=1.5,
                   label=(f"best τ={best_r[tau_key]:.5f}  "
                          f"@10={best_r['ndcg10']:.4f}  saved {best_r['n_saved']}q"))
    if knee_r is not None and not math.isinf(knee_r[tau_key]) and knee_r[tau_key] > 0:
        ax.axvline(knee_r[tau_key], color=CYAN, ls="-.", lw=1.5,
                   label=(f"knee τ={knee_r[tau_key]:.5f}  "
                          f"@10={knee_r['ndcg10']:.4f}  saved {knee_r['n_saved']}q"))

    if is_log:
        ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("nDCG", fontsize=11)
    axr.set_ylabel("% queries → duoT5", color=ORANGE, fontsize=11)
    axr.set_ylim(0, 110)
    ax.set_title(title, fontsize=13, fontweight="bold")

    ll, ls_ = ax.get_legend_handles_labels()
    rl, rs_ = axr.get_legend_handles_labels()
    ax.legend(ll + rl, ls_ + rs_, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Plot → {out_path}")


# ── individual sweep plots ────────────────────────────────────────────────────

_dual_axis_sweep(gap_sweep, "tau",
                 "GAP  (top-1 vs top-2 monoT5 P(true) gap)",
                 best_gap, gap_knee,
                 PLOTS_DIR / "gap_sweep.png",
                 "Threshold τ  (gap < τ → send to duoT5)  [log scale]", is_log=True)

_dual_axis_sweep(mau5_sweep, "tau_unc",
                 "MAU@5  (mean Bernoulli uncertainty, top-5)",
                 best_mau5, mau5_knee,
                 PLOTS_DIR / "mau5_sweep.png",
                 "Threshold τ  (MAU@5 ≥ τ → send to duoT5)")

_dual_axis_sweep(mau10_sweep, "tau_unc",
                 "MAU@10  (mean Bernoulli uncertainty, top-10)",
                 best_mau10, mau10_knee,
                 PLOTS_DIR / "mau10_sweep.png",
                 "Threshold τ  (MAU@10 ≥ τ → send to duoT5)")

# Entropy H@20 / H@50 side by side
fig5, axes5 = plt.subplots(1, 2, figsize=(18, 6))
for ax5, k in zip(axes5, ENTROPY_KS):
    col_k   = ENT_COLORS[k]
    sweep_k = ent_sweeps[k]
    finite  = [r for r in sweep_k if not math.isinf(r["tau_unc"]) and r["tau_unc"] > 0]
    tx   = [r["tau_unc"]  for r in finite]
    n1_  = [r["ndcg1"]    for r in finite]
    n5_  = [r["ndcg5"]    for r in finite]
    n10_ = [r["ndcg10"]   for r in finite]
    cost = [r["pct_duo"]  for r in finite]

    axr = ax5.twinx()
    ax5.plot(tx, n1_,  color=PINK,      lw=1.6, ms=2.5, marker="o", label="nDCG@1")
    ax5.plot(tx, n5_,  color=BLUE,      lw=1.6, ms=2.5, marker="o", label="nDCG@5")
    ax5.plot(tx, n10_, color="#1f77b4", lw=2.0, ms=2.5, marker="o", label="nDCG@10")
    axr.plot(tx, cost, color=ORANGE,    lw=1.8, ms=2.5, ls="--", marker="s", label="% duo")

    for val, col in [(mono_n1, PINK), (mono_n5, BLUE), (mono_ndcg, "#1f77b4")]:
        ax5.axhline(val, color=col, ls=":", lw=0.8, alpha=0.45)
    for val, col in [(duo_n1, PINK), (duo_n5, BLUE), (duo_ndcg, "#1f77b4")]:
        ax5.axhline(val, color=col, ls="-", lw=0.8, alpha=0.2)

    kn = ent_knees[k]
    if not math.isinf(kn["tau_unc"]) and kn["tau_unc"] > 0:
        ax5.axvline(kn["tau_unc"], color=CYAN, ls="-.", lw=1.5,
                    label=f"knee τ={kn['tau_unc']:.3f}  @10={kn['ndcg10']:.4f}  saved {kn['n_saved']}q")
    bn = best_ent[k]
    if not math.isinf(bn["tau_unc"]) and bn["tau_unc"] > 0:
        ax5.axvline(bn["tau_unc"], color=GREEN, ls="-", lw=1.5,
                    label=f"best τ={bn['tau_unc']:.3f}  @10={bn['ndcg10']:.4f}  saved {bn['n_saved']}q")

    ax5.set_xlabel(f"τ  (H@{k} ≥ τ → duoT5)", fontsize=10)
    ax5.set_ylabel("nDCG", fontsize=10)
    axr.set_ylabel("% queries → duoT5", color=ORANGE, fontsize=9)
    axr.set_ylim(0, 110)
    ax5.set_title(f"H@{k}  (top-{k} of full 50-doc pool)", fontsize=11, fontweight="bold")
    ll, ls_ = ax5.get_legend_handles_labels()
    rl, rs_ = axr.get_legend_handles_labels()
    ax5.legend(ll + rl, ls_ + rs_, fontsize=7.5, loc="center right")

fig5.suptitle(f"Entropy-gated mono→duo  —  H@20 vs H@50  ({n} queries)",
              fontsize=13, fontweight="bold")
fig5.tight_layout()
p5 = PLOTS_DIR / "entropy_sweep.png"
fig5.savefig(p5, dpi=160); plt.close(fig5); print(f"Plot → {p5}")


# ── Pareto comparison: 3 rows × 5 signals ────────────────────────────────────
METRICS   = [("ndcg1", "nDCG@1"), ("ndcg5", "nDCG@5"), ("ndcg10", "nDCG@10")]
BASELINES = [(mono_n1, duo_n1), (mono_n5, duo_n5), (mono_ndcg, duo_ndcg)]

SIGNALS = [
    ("GAP",    gap_sweep,      gap_knee,   best_gap,   "tau",     BLUE,              {0.001, 0.01, 0.1}),
    ("MAU@5",  mau5_sweep,     mau5_knee,  best_mau5,  "tau_unc", GREEN,             {0.01, 0.05, 0.10}),
    ("MAU@10", mau10_sweep,    mau10_knee, best_mau10, "tau_unc", ORANGE,            {0.01, 0.05, 0.10}),
    ("H@20",   ent_sweeps[20], ent_knees[20], best_ent[20], "tau_unc", ENT_COLORS[20], {0.5, 0.7, 0.9}),
    ("H@50",   ent_sweeps[50], ent_knees[50], best_ent[50], "tau_unc", ENT_COLORS[50], {0.5, 0.7, 0.9}),
]

ncols = len(SIGNALS)
fig3, axes3 = plt.subplots(3, ncols, figsize=(5 * ncols, 15), sharey="row")

for row_i, ((mkey, mlabel), (bm, bd)) in enumerate(zip(METRICS, BASELINES)):
    for col_i, (sname, sweep, kn, bn, tk, col, ataus) in enumerate(SIGNALS):
        ax = axes3[row_i, col_i]
        costs = [r["pct_duo"] for r in sweep]
        vals  = [r[mkey]      for r in sweep]
        ax.plot(costs, vals, color=col, marker="o", ms=2.5, lw=2, alpha=0.9)
        ax.scatter([kn["pct_duo"]], [kn[mkey]], color=CYAN, s=90, zorder=5,
                   label=(f"knee τ={fmt_tau(kn[tk], '.3f')}\n"
                          f"  {mlabel}={kn[mkey]:.4f}\n"
                          f"  saved {kn['n_saved']}q"))
        ax.scatter([bn["pct_duo"]], [bn[mkey]], color=GREEN, s=90, zorder=5, marker="*",
                   label=(f"best τ={fmt_tau(bn[tk], '.3f')}\n"
                          f"  {mlabel}={bn[mkey]:.4f}\n"
                          f"  saved {bn['n_saved']}q"))
        ax.axhline(bm, color="gray", ls=":", lw=1, alpha=0.7, label=f"mono ({bm:.4f})")
        ax.axhline(bd, color=RED,    ls=":", lw=1, alpha=0.6, label=f"duo ({bd:.4f})")
        ax.axhspan(bm, bd, alpha=0.05, color="green")
        for r in sweep:
            tv = r[tk]
            if not math.isinf(tv) and round(tv, 3) in {round(a, 3) for a in ataus}:
                ax.annotate(f"{tv:.3f}", (r["pct_duo"], r[mkey]),
                            textcoords="offset points", xytext=(3, 3), fontsize=6)
        if col_i == 0:
            ax.set_ylabel(mlabel, fontsize=11)
        if row_i == 0:
            ax.set_title(sname, fontsize=12, fontweight="bold")
        if row_i == 2:
            ax.set_xlabel("% queries → duoT5  (cost)", fontsize=10)
        ax.legend(fontsize=6.5, loc="lower right")

fig3.suptitle(f"Pareto frontiers — all signals × 3 nDCG cutoffs  ({n} queries)",
              fontsize=14, fontweight="bold")
fig3.tight_layout()
p3 = PLOTS_DIR / "pareto_comparison.png"
fig3.savefig(p3, dpi=140); plt.close(fig3); print(f"Plot → {p3}")


# ── signal_comparison: all 5 signals overlaid, 3 panels ──────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

for ax4, (mkey, mlabel), (bm, bd) in zip(axes4, METRICS, BASELINES):
    for sname, sweep, kn, bn, tk, col, _ in SIGNALS:
        costs = [r["pct_duo"] for r in sweep]
        vals  = [r[mkey]      for r in sweep]
        ax4.plot(costs, vals, color=col, marker="o", ms=2, lw=2, alpha=0.85, label=sname)
        ax4.scatter([kn["pct_duo"]], [kn[mkey]], color=col, s=90, zorder=6, marker="D")

    ax4.axhline(bm, color="black", ls=":", lw=1.1, alpha=0.5, label=f"mono ({bm:.4f})")
    ax4.axhline(bd, color=RED,    ls=":", lw=1.1, alpha=0.5, label=f"duo  ({bd:.4f})")
    ax4.axhspan(bm, bd, alpha=0.06, color="green")
    ax4.set_xlabel("% queries → duoT5  (cost)", fontsize=11)
    ax4.set_ylabel(mlabel, fontsize=11)
    ax4.set_title(f"Pareto  —  {mlabel}", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8.5, loc="lower right")

# winner badge
kw_tau_val = kw_r[kw_tau]
axes4[2].text(0.98, 0.02,
              f"Knee winner: {kw_k}\n"
              f"τ = {fmt_tau(kw_tau_val, '.4f')}\n"
              f"nDCG@1={kw_r['ndcg1']:.4f}  "
              f"@5={kw_r['ndcg5']:.4f}  "
              f"@10={kw_r['ndcg10']:.4f}\n"
              f"saved {kw_r['n_saved']}q  ({100 - kw_r['pct_duo']:.1f}%)",
              transform=axes4[2].transAxes, ha="right", va="bottom",
              fontsize=9, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", alpha=0.9))

fig4.suptitle(f"All signals — Pareto by nDCG cutoff  ({n} queries)", fontsize=13, fontweight="bold")
fig4.tight_layout()
p4 = PLOTS_DIR / "signal_comparison.png"
fig4.savefig(p4, dpi=160); plt.close(fig4); print(f"Plot → {p4}")

print("\nDone. All outputs in:", OUT_DIR)
