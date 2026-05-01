"""MonoGatedDuo grid search — compare two gating signals for duoT5.

Uses pre-computed per-query results from cached runs to simulate the
MonoGatedDuoCascade at every threshold without re-running any model.

Two signals compared:
  GAP    — Δ(q) = P(true|d1) − P(true|d2)  (top-1 vs top-2 monoT5 gap)
            send to duoT5 when Δ(q) < τ   (small gap = mono is uncertain at top)

  MAU10  — mean p*(1−p) over top-10 monoT5 docs per query
            send to duoT5 when MAU@10 ≥ τ  (high mean uncertainty = need pairwise)

Fixed data sources:
  mono cache  : application/cache/runs/monot5/20260429T113438Z.json
  duo  cache  : application/cache/runs/mono_duo/20260429T134302Z.json
  score cache : models/monot5/margin_scores_cache.json

Outputs:
  results/gap_sweep.csv
  results/mau_sweep.csv
  results/summary.txt
  plots/gap_sweep.png
  plots/mau_sweep.png
  plots/pareto_comparison.png
  plots/signal_comparison.png
"""

import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

MONO_CACHE  = ROOT / "application/cache/runs/monot5/20260429T113438Z.json"
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
# GAP signal: Δ = P(true|d1) − P(true|d2) ∈ [0, 1]
# send to duoT5 when Δ < τ  →  τ=0 → never duo, τ=∞ → always duo
GAP_GRID: list[float] = sorted({
    0.0,
    *[round(v, 6) for v in np.logspace(-5, -0.3, 60)],  # 1e-5 … 0.5
    0.001, 0.002, 0.003, 0.005, 0.007,
    0.01,  0.02,  0.03,  0.05,  0.07,
    0.1,   0.15,  0.2,   0.3,   0.5,   0.8, 1.0,
    math.inf,
})

# MAU10 signal: p*(1−p) ∈ [0, 0.25]
# send to duoT5 when MAU ≥ τ  →  τ=0 → always duo, τ=∞ → never duo
MAU_GRID: list[float] = sorted({
    0.0,
    *[round(v, 6) for v in np.linspace(0.001, 0.249, 80)],
    0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25,
    math.inf,
})

# ── helpers ───────────────────────────────────────────────────────────────────

def load_json(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def get_metric(entry: dict, key: str, k: str) -> float:
    return float(entry.get("metrics", {}).get(key, {}).get(k, 0.0))


def knee_index(costs: list[float], ndcgs: list[float]) -> int:
    """Index of the Pareto-frontier point farthest from the endpoints line."""
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


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading caches …")
mono_run   = load_json(MONO_CACHE)
duo_run    = load_json(DUO_CACHE)
scores_raw = load_json(SCORE_CACHE)   # {qid: {docid: P(true)}}

mono_pq: dict = mono_run["per_query"]
duo_pq:  dict = duo_run["per_query"]

# shared queries present in both run caches
qids = sorted(set(mono_pq) & set(duo_pq))
print(f"  mono cache  : {len(mono_pq)} queries")
print(f"  duo  cache  : {len(duo_pq)} queries")
print(f"  score cache : {len(scores_raw)} queries")
print(f"  shared qids : {len(qids)}")

# ── compute per-query signals ─────────────────────────────────────────────────

ENTROPY_KS = [5, 10, 20, 50]   # top-k windows for entropy signal

gaps:        dict[str, float] = {}   # top-1 vs top-2 P(true) gap
mau5_scores: dict[str, float] = {}   # mean average uncertainty @5
mau10_scores: dict[str, float] = {}  # mean average uncertainty @10
# entropy[k] = {qid: normalised H(top-k P(true) distribution)}
entropy_scores: dict[int, dict[str, float]] = {k: {} for k in ENTROPY_KS}


def _norm_entropy(vals: list[float]) -> float | None:
    """Normalised Shannon entropy of a score vector (in [0, 1]).

    Scores are treated as un-normalised weights; we normalise to a probability
    simplex, then compute H / log(k) so the result is always in [0, 1].
    High value → flat distribution → model is uncertain about the ranking.
    """
    if not vals or sum(vals) == 0:
        return None
    s = sum(vals)
    probs = [v / s for v in vals]
    h = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
    return h / math.log(len(probs))   # normalise: max = log(k)/log(k) = 1


for qid in qids:
    if qid not in scores_raw:
        continue
    doc_scores = scores_raw[qid]   # {docid: P(true)}
    top_docids = mono_pq[qid].get("top_docids", [])

    # GAP signal: top-1 vs top-2 P(true) gap
    top2 = top_docids[:2]
    if len(top2) == 2 and top2[0] in doc_scores and top2[1] in doc_scores:
        gaps[qid] = doc_scores[top2[0]] - doc_scores[top2[1]]

    # MAU@5 / MAU@10
    for k, store in [(5, mau5_scores), (10, mau10_scores)]:
        topk = top_docids[:k]
        vals = [doc_scores[d] * (1.0 - doc_scores[d]) for d in topk if d in doc_scores]
        if vals:
            store[qid] = sum(vals) / len(vals)

    # Entropy @5 / @10 / @20 / @50  (normalised to [0,1])
    for k in ENTROPY_KS:
        topk = top_docids[:k]
        vals = [doc_scores[d] for d in topk if d in doc_scores]
        h = _norm_entropy(vals)
        if h is not None:
            entropy_scores[k][qid] = h

n_with_gap  = sum(1 for q in qids if q in gaps)
n_with_mau5 = sum(1 for q in qids if q in mau5_scores)
n_with_mau  = sum(1 for q in qids if q in mau10_scores)
print(f"\nSignal coverage:")
print(f"  GAP   : {n_with_gap}/{len(qids)} queries")
print(f"  MAU@5 : {n_with_mau5}/{len(qids)} queries")
print(f"  MAU@10: {n_with_mau}/{len(qids)} queries")
for k in ENTROPY_KS:
    n = sum(1 for q in qids if q in entropy_scores[k])
    print(f"  H@{k:<3d} : {n}/{len(qids)} queries")

# distribution stats
gap_vals  = sorted(gaps[q]         for q in qids if q in gaps)
mau5_vals = sorted(mau5_scores[q]  for q in qids if q in mau5_scores)
mau_vals  = sorted(mau10_scores[q] for q in qids if q in mau10_scores)

def _pct_summary(vals: list[float], label: str) -> None:
    if not vals:
        return
    mean = sum(vals) / len(vals)
    print(f"\n{label}  (mean={mean:.4f}):")
    for pct in [0, 25, 50, 75, 90, 100]:
        idx = min(int(pct / 100 * len(vals)), len(vals) - 1)
        print(f"  p{pct:3d}: {vals[idx]:.6f}")

_pct_summary(gap_vals,  "GAP distribution (top-1 − top-2 P(true))")
mean_mau5 = sum(mau5_vals) / len(mau5_vals) if mau5_vals else 0.0
_pct_summary(mau5_vals, "MAU@5")
mean_mau  = sum(mau_vals)  / len(mau_vals)  if mau_vals  else 0.0
_pct_summary(mau_vals,  "MAU@10")
for k in ENTROPY_KS:
    ev = sorted(entropy_scores[k][q] for q in qids if q in entropy_scores[k])
    _pct_summary(ev, f"Entropy@{k}  (normalised H / log({k}))")

# ── simulation functions ──────────────────────────────────────────────────────

def _aggregate(entries: list[dict], use_duo_flags: list[bool], tau_key: str, tau_val: float) -> dict:
    n1, n5, n10 = [], [], []
    mrr_vals, map_vals = [], []
    n_duo = sum(use_duo_flags)
    for entry in entries:
        n1.append(get_metric(entry,  "ndcg_at", "1"))
        n5.append(get_metric(entry,  "ndcg_at", "5"))
        n10.append(get_metric(entry, "ndcg_at", "10"))
        mrr_vals.append(get_metric(entry, "mrr_at", "10"))
        map_vals.append(get_metric(entry, "map_at", "10"))
    n = len(entries)
    n_total = len(qids)
    return {
        tau_key:      tau_val,
        "ndcg1":      sum(n1)  / n if n else 0.0,
        "ndcg5":      sum(n5)  / n if n else 0.0,
        "ndcg10":     sum(n10) / n if n else 0.0,
        "mrr10":      sum(mrr_vals) / n if n else 0.0,
        "map10":      sum(map_vals) / n if n else 0.0,
        "pct_duo":    n_duo / n_total * 100 if n_total else 0.0,
        "n_duo":      n_duo,
        "n_mono_only": n_total - n_duo,
        "n_queries":  n_total,
    }


def simulate_gap(tau: float) -> dict:
    entries, flags = [], []
    for qid in qids:
        use_duo = qid in gaps and gaps[qid] < tau
        entry   = (duo_pq if use_duo else mono_pq).get(qid) or mono_pq.get(qid)
        if entry is None:
            continue
        entries.append(entry)
        flags.append(use_duo)
    return _aggregate(entries, flags, "tau", tau)


def simulate_mau(tau_unc: float, score_dict: dict) -> dict:
    entries, flags = [], []
    for qid in qids:
        use_duo = qid in score_dict and score_dict[qid] >= tau_unc
        entry   = (duo_pq if use_duo else mono_pq).get(qid) or mono_pq.get(qid)
        if entry is None:
            continue
        entries.append(entry)
        flags.append(use_duo)
    return _aggregate(entries, flags, "tau_unc", tau_unc)


# ── run sweeps ────────────────────────────────────────────────────────────────

# Entropy grid: normalised H ∈ [0,1]; finer near 0 (confident) and near 1 (maximal entropy)
ENT_GRID: list[float] = sorted({
    0.0,
    *[round(v, 5) for v in np.linspace(0.05, 0.95, 100)],
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    math.inf,
})

print("\nRunning GAP sweep …")
gap_sweep  = [simulate_gap(t) for t in GAP_GRID]

print("Running MAU@5 sweep …")
mau5_sweep = [simulate_mau(t, mau5_scores)  for t in MAU_GRID]

print("Running MAU@10 sweep …")
mau_sweep  = [simulate_mau(t, mau10_scores) for t in MAU_GRID]

print("Running Entropy sweeps …")
# entropy: send to duoT5 when H_norm(q) >= tau  (high entropy = uncertain = needs pairwise)
ent_sweeps: dict[int, list[dict]] = {}
for k in ENTROPY_KS:
    print(f"  H@{k} …")
    ent_sweeps[k] = [simulate_mau(t, entropy_scores[k]) for t in ENT_GRID]

# reference baselines (τ=0 for gap = pure mono, τ=∞ for gap = pure duo)
mono_ndcg = gap_sweep[0]["ndcg10"]   # τ=0 → all mono
duo_ndcg  = gap_sweep[-1]["ndcg10"]  # τ=∞ → all duo

print(f"\n  Baseline monoT5  nDCG@10 = {mono_ndcg:.4f}")
print(f"  Baseline mono_duo nDCG@10 = {duo_ndcg:.4f}  (always duoT5)")

# ── find best tau ─────────────────────────────────────────────────────────────

best_gap  = max(gap_sweep,  key=lambda r: r["ndcg10"])
best_mau5 = max(mau5_sweep, key=lambda r: r["ndcg10"])
best_mau  = max(mau_sweep,  key=lambda r: r["ndcg10"])
best_ent: dict[int, dict] = {k: max(ent_sweeps[k], key=lambda r: r["ndcg10"])
                              for k in ENTROPY_KS}

# Pareto knee (max nDCG gain per unit cost)
gap_costs   = [r["pct_duo"] for r in gap_sweep]
gap_ndcgs   = [r["ndcg10"]  for r in gap_sweep]
mau5_costs  = [r["pct_duo"] for r in mau5_sweep]
mau5_ndcgs  = [r["ndcg10"]  for r in mau5_sweep]
mau_costs   = [r["pct_duo"] for r in mau_sweep]
mau_ndcgs   = [r["ndcg10"]  for r in mau_sweep]

gap_ki    = knee_index(gap_costs,  gap_ndcgs)
mau5_ki   = knee_index(mau5_costs, mau5_ndcgs)
mau_ki    = knee_index(mau_costs,  mau_ndcgs)
gap_knee  = gap_sweep[gap_ki]
mau5_knee = mau5_sweep[mau5_ki]
mau_knee  = mau_sweep[mau_ki]

ent_knees: dict[int, dict] = {}
for k in ENTROPY_KS:
    costs_k = [r["pct_duo"] for r in ent_sweeps[k]]
    ndcgs_k = [r["ndcg10"]  for r in ent_sweeps[k]]
    ki = knee_index(costs_k, ndcgs_k)
    ent_knees[k] = ent_sweeps[k][ki]

def fmt_tau(t: float, fmt: str = ".5f") -> str:
    return "∞ (always duo)" if math.isinf(t) else f"{t:{fmt}}"


# ── print tables ──────────────────────────────────────────────────────────────

HDR  = f"{'τ':>16}  {'nDCG@1':>7}  {'nDCG@5':>7}  {'nDCG@10':>8}  {'MRR@10':>8}  {'%duo':>6}  {'n_duo':>6}  {'n_saved':>7}"
SEP  = "─" * 78

def _row_gap(r, best, knee):
    marker = " ◄ BEST" if r is best else (" ◄ KNEE" if r is knee else "")
    return (f"{fmt_tau(r['tau']):>16}  {r['ndcg1']:>7.4f}  {r['ndcg5']:>7.4f}  "
            f"{r['ndcg10']:>8.4f}  {r['mrr10']:>8.4f}  {r['pct_duo']:>5.1f}%  "
            f"{r['n_duo']:>6}  {r['n_mono_only']:>7}{marker}")

def _row_mau(r, best, knee):
    marker = " ◄ BEST" if r is best else (" ◄ KNEE" if r is knee else "")
    return (f"{fmt_tau(r['tau_unc']):>16}  {r['ndcg1']:>7.4f}  {r['ndcg5']:>7.4f}  "
            f"{r['ndcg10']:>8.4f}  {r['mrr10']:>8.4f}  {r['pct_duo']:>5.1f}%  "
            f"{r['n_duo']:>6}  {r['n_mono_only']:>7}{marker}")

print("\n" + "=" * 78)
print("GAP SWEEP  (send to duoT5 when Δ = top1−top2 < τ)")
print("=" * 78)
print(HDR); print(SEP)
for r in gap_sweep:
    print(_row_gap(r, best_gap, gap_knee))

print("\n" + "=" * 78)
print("MAU@5 SWEEP  (send to duoT5 when MAU@5 ≥ τ_unc)")
print("=" * 78)
print(HDR); print(SEP)
for r in mau5_sweep:
    print(_row_mau(r, best_mau5, mau5_knee))

print("\n" + "=" * 78)
print("MAU@10 SWEEP  (send to duoT5 when MAU@10 ≥ τ_unc)")
print("=" * 78)
print(HDR); print(SEP)
for r in mau_sweep:
    print(_row_mau(r, best_mau, mau_knee))

for k in ENTROPY_KS:
    print(f"\n{'=' * 78}")
    print(f"ENTROPY@{k} SWEEP  (send to duoT5 when H@{k} ≥ τ_ent,  H normalised to [0,1])")
    print("=" * 78)
    print(HDR); print(SEP)
    for r in ent_sweeps[k]:
        print(_row_mau(r, best_ent[k], ent_knees[k]))


# ── summary ───────────────────────────────────────────────────────────────────

all_best   = {"GAP": best_gap, "MAU@5": best_mau5, "MAU@10": best_mau,
              **{f"H@{k}": best_ent[k] for k in ENTROPY_KS}}
winner     = max(all_best, key=lambda k: all_best[k]["ndcg10"])
winner_r   = all_best[winner]
winner_t   = fmt_tau(winner_r.get("tau", winner_r.get("tau_unc", 0)))
winner_ndcg = winner_r["ndcg10"]

# Pareto knee winner (best nDCG at knee)
knee_by_signal = {"GAP": gap_knee, "MAU@5": mau5_knee, "MAU@10": mau_knee,
                  **{f"H@{k}": ent_knees[k] for k in ENTROPY_KS}}
knee_winner    = max(knee_by_signal, key=lambda k: knee_by_signal[k]["ndcg10"])

# baselines at all three cutoffs
mono_n1  = gap_sweep[0]["ndcg1"];  duo_n1  = gap_sweep[-1]["ndcg1"]
mono_n5  = gap_sweep[0]["ndcg5"];  duo_n5  = gap_sweep[-1]["ndcg5"]

def _sig_block(label, r_best, r_knee, tau_key):
    t_best = fmt_tau(r_best[tau_key])
    t_knee = fmt_tau(r_knee[tau_key])
    return f"""\
──────────────────────────────────────────────────────────────────────
{label}
  Best τ     : {t_best}
  Best nDCG  : @1={r_best['ndcg1']:.4f}  @5={r_best['ndcg5']:.4f}  @10={r_best['ndcg10']:.4f}
  Queries to duo    : {r_best['n_duo']} / {len(qids)}  ({r_best['pct_duo']:.1f}%)
  Queries kept mono : {r_best['n_mono_only']} / {len(qids)}  ({100 - r_best['pct_duo']:.1f}%)
  Pareto knee τ : {t_knee}
    nDCG  @1={r_knee['ndcg1']:.4f}  @5={r_knee['ndcg5']:.4f}  @10={r_knee['ndcg10']:.4f}
    saved {r_knee['n_mono_only']}q  ({100 - r_knee['pct_duo']:.1f}% cost reduction)"""

summary = textwrap.dedent(f"""
╔══════════════════════════════════════════════════════════════════════╗
║     MonoGatedDuo Grid Search — Summary  (nDCG @1 / @5 / @10)       ║
╚══════════════════════════════════════════════════════════════════════╝

Baselines                  nDCG@1    nDCG@5   nDCG@10
  monoT5  (never duo)     {mono_n1:.4f}    {mono_n5:.4f}    {mono_ndcg:.4f}
  mono_duo (always duo)   {duo_n1:.4f}    {duo_n5:.4f}    {duo_ndcg:.4f}
  Δ (duo − mono)         {duo_n1-mono_n1:+.4f}   {duo_n5-mono_n5:+.4f}   {duo_ndcg-mono_ndcg:+.4f}
  Total queries           {len(qids)}
""").strip()

summary += "\n\n"
summary += _sig_block("Signal 1 — GAP  (top-1 vs top-2 P(true) gap)",
                      best_gap, gap_knee, "tau")
summary += "\n\n"
summary += _sig_block("Signal 2 — MAU@5  (mean Bernoulli variance over top-5)",
                      best_mau5, mau5_knee, "tau_unc")
summary += "\n\n"
summary += _sig_block("Signal 3 — MAU@10  (mean Bernoulli variance over top-10)",
                      best_mau, mau_knee, "tau_unc")

for k in ENTROPY_KS:
    summary += "\n\n"
    summary += _sig_block(f"Signal H@{k}  — Rank-distribution entropy  (top-{k}, H/log({k}))",
                          best_ent[k], ent_knees[k], "tau_unc")

kw_knee   = knee_by_signal[knee_winner]
kw_tk     = "tau" if knee_winner == "GAP" else "tau_unc"
summary += f"""

──────────────────────────────────────────────────────────────────────
WINNER (best nDCG@10): {winner}  τ={fmt_tau(winner_r.get('tau', winner_r.get('tau_unc', 0)))}
  nDCG  @1={winner_r['ndcg1']:.4f}  @5={winner_r['ndcg5']:.4f}  @10={winner_ndcg:.4f}
  Δ vs monoT5: @1={winner_r['ndcg1']-mono_n1:+.4f}  @5={winner_r['ndcg5']-mono_n5:+.4f}  @10={winner_ndcg-mono_ndcg:+.4f}

WINNER (Pareto knee): {knee_winner}  τ={fmt_tau(kw_knee[kw_tk])}
  nDCG  @1={kw_knee['ndcg1']:.4f}  @5={kw_knee['ndcg5']:.4f}  @10={kw_knee['ndcg10']:.4f}
  Saved {kw_knee['n_mono_only']} queries  ({100 - kw_knee['pct_duo']:.1f}% cost reduction)
"""

print("\n" + summary)

# write to file
(RESULTS_DIR / "summary.txt").write_text(summary + "\n")
print(f"\nSummary → {RESULTS_DIR / 'summary.txt'}")


# ── save CSVs ─────────────────────────────────────────────────────────────────

GAP_FIELDS = ["tau",     "ndcg1", "ndcg5", "ndcg10", "mrr10", "map10",
              "pct_duo", "n_duo", "n_mono_only", "n_queries"]
MAU_FIELDS = ["tau_unc", "ndcg1", "ndcg5", "ndcg10", "mrr10", "map10",
              "pct_duo", "n_duo", "n_mono_only", "n_queries"]

gap_csv = RESULTS_DIR / "gap_sweep.csv"
with open(gap_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=GAP_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in gap_sweep:
        row = dict(r); row["tau"] = "inf" if math.isinf(r["tau"]) else r["tau"]
        w.writerow(row)
print(f"GAP CSV   → {gap_csv}")

mau5_csv = RESULTS_DIR / "mau5_sweep.csv"
with open(mau5_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=MAU_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in mau5_sweep:
        row = dict(r); row["tau_unc"] = "inf" if math.isinf(r["tau_unc"]) else r["tau_unc"]
        w.writerow(row)
print(f"MAU@5 CSV → {mau5_csv}")

mau_csv = RESULTS_DIR / "mau10_sweep.csv"
with open(mau_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=MAU_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in mau_sweep:
        row = dict(r); row["tau_unc"] = "inf" if math.isinf(r["tau_unc"]) else r["tau_unc"]
        w.writerow(row)
print(f"MAU@10 CSV → {mau_csv}")

for k in ENTROPY_KS:
    ent_csv = RESULTS_DIR / f"entropy{k}_sweep.csv"
    with open(ent_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MAU_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in ent_sweeps[k]:
            row = dict(r); row["tau_unc"] = "inf" if math.isinf(r["tau_unc"]) else r["tau_unc"]
            w.writerow(row)
    print(f"H@{k} CSV  → {ent_csv}")


# ── plotting ──────────────────────────────────────────────────────────────────

BLUE    = "#4c72b0"
ORANGE  = "#dd8452"
GREEN   = "#2ca02c"
RED     = "#d62728"
PURPLE  = "#9467bd"
CYAN    = "#17becf"

def _dual_axis_sweep(sweep, tau_key, title, xscale, best_r, knee_r, out_path,
                     xlabel, is_log=False):
    """Dual-axis plot: nDCG@1/@5/@10 on left, % duo on right."""
    finite = [r for r in sweep if not math.isinf(r[tau_key]) and r[tau_key] > 0]
    tx  = [r[tau_key]      for r in finite]
    n1  = [r["ndcg1"]      for r in finite]
    n5  = [r["ndcg5"]      for r in finite]
    n10 = [r["ndcg10"]     for r in finite]
    cost = [r["pct_duo"]   for r in finite]
    saved = [r["n_mono_only"] for r in finite]

    fig, ax = plt.subplots(figsize=(12, 5))
    axr = ax.twinx()

    ax.plot(tx, n1,  color="#e377c2", marker="o", ms=3, lw=1.8, label="nDCG@1")
    ax.plot(tx, n5,  color=BLUE,      marker="o", ms=3, lw=1.8, label="nDCG@5")
    ax.plot(tx, n10, color="#1f77b4",  marker="o", ms=3, lw=2.2, label="nDCG@10", alpha=0.9)
    axr.plot(tx, cost, color=ORANGE, marker="s", ms=3, lw=2, ls="--",
             label="% queries → duoT5")

    # baselines for each cutoff
    for val, base_val, col, k in [
        (mono_n1,  duo_n1,  "#e377c2", "@1"),
        (mono_n5,  duo_n5,  BLUE,      "@5"),
        (mono_ndcg, duo_ndcg, "#1f77b4","@10"),
    ]:
        ax.axhline(val,      color=col, ls=":", lw=0.9, alpha=0.45)
        ax.axhline(base_val, color=col, ls="-", lw=0.9, alpha=0.25)

    if best_r is not None and not math.isinf(best_r[tau_key]) and best_r[tau_key] > 0:
        ax.axvline(best_r[tau_key], color=GREEN, ls="-", lw=1.5,
                   label=f"best τ={best_r[tau_key]:.5f}")
    if knee_r is not None and not math.isinf(knee_r[tau_key]) and knee_r[tau_key] > 0:
        ax.axvline(knee_r[tau_key], color=CYAN, ls="-.", lw=1.5,
                   label=f"knee τ={knee_r[tau_key]:.5f}")

    if is_log:
        ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("nDCG", fontsize=11)
    axr.set_ylabel("% queries → duoT5", color=ORANGE, fontsize=11)
    axr.set_ylim(0, 110)
    ax.set_title(title, fontsize=13, fontweight="bold")

    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = axr.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="center right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Plot → {out_path}")


_dual_axis_sweep(gap_sweep,  "tau",     "Signal 1 — GAP  (top-1 vs top-2 monoT5 P(true) gap)",
                 "symlog", best_gap,  gap_knee,
                 PLOTS_DIR / "gap_sweep.png",
                 "Threshold τ  (Δ < τ → send to duoT5)  [log scale]", is_log=True)

# ── Plot 2: MAU@10 sweep — dual axis ─────────────────────────────────────────
_dual_axis_sweep(mau5_sweep, "tau_unc", "Signal 2 — MAU@5  (mean Bernoulli uncertainty over top-5 docs)",
                 None, best_mau5, mau5_knee,
                 PLOTS_DIR / "mau5_sweep.png",
                 "Threshold τ_unc  (MAU@5 ≥ τ_unc → send to duoT5)")

_dual_axis_sweep(mau_sweep, "tau_unc", "Signal 3 — MAU@10  (mean Bernoulli uncertainty over top-10 docs)",
                 None, best_mau, mau_knee,
                 PLOTS_DIR / "mau10_sweep.png",
                 "Threshold τ_unc  (MAU@10 ≥ τ_unc → send to duoT5)")

# ── Plot 3: Pareto — 3 rows (nDCG@1/@5/@10), 3 cols (GAP/MAU5/MAU10) ────────
METRICS   = [("ndcg1", "nDCG@1"), ("ndcg5", "nDCG@5"), ("ndcg10", "nDCG@10")]
BASELINES = [(mono_n1, duo_n1), (mono_n5, duo_n5), (mono_ndcg, duo_ndcg)]
SIGNALS   = [
    ("GAP",    gap_sweep,   gap_knee,   best_gap,   "tau",     BLUE,     {0.001, 0.01, 0.1}),
    ("MAU@5",  mau5_sweep,  mau5_knee,  best_mau5,  "tau_unc", "#2ca02c",{0.01, 0.05, 0.10}),
    ("MAU@10", mau_sweep,   mau_knee,   best_mau,   "tau_unc", ORANGE,   {0.01, 0.05, 0.10}),
]

fig3, axes3 = plt.subplots(3, 3, figsize=(18, 15), sharey="row")

for row_i, ((mkey, mlabel), (base_mono, base_duo)) in enumerate(zip(METRICS, BASELINES)):
    for col_i, (sig_name, sweep, knee_r, best_r, tk, col, ataus) in enumerate(SIGNALS):
        ax = axes3[row_i, col_i]
        costs = [r["pct_duo"] for r in sweep]
        vals  = [r[mkey]      for r in sweep]
        ax.plot(costs, vals, color=col, marker="o", ms=3, lw=2, alpha=0.9)
        ax.scatter([knee_r["pct_duo"]], [knee_r[mkey]],
                   color=CYAN,  s=90, zorder=5,
                   label=f"knee  τ={fmt_tau(knee_r[tk], '.4f')}\n  {mlabel}={knee_r[mkey]:.4f}")
        ax.scatter([best_r["pct_duo"]], [best_r[mkey]],
                   color=GREEN, s=90, zorder=5, marker="*",
                   label=f"best  τ={fmt_tau(best_r[tk], '.4f')}\n  {mlabel}={best_r[mkey]:.4f}")
        ax.axhline(base_mono, color="gray",  ls=":", lw=1, alpha=0.7,
                   label=f"monoT5 ({base_mono:.4f})")
        ax.axhline(base_duo,  color=RED,     ls=":", lw=1, alpha=0.6,
                   label=f"mono_duo ({base_duo:.4f})")
        ax.axhspan(base_mono, base_duo, alpha=0.05, color="green")
        for r in sweep:
            if r[tk] in ataus:
                ax.annotate(f"{r[tk]:.3f}", (r["pct_duo"], r[mkey]),
                            textcoords="offset points", xytext=(3, 3), fontsize=6.5)
        if col_i == 0:
            ax.set_ylabel(mlabel, fontsize=11)
        if row_i == 0:
            ax.set_title(f"{sig_name} signal", fontsize=12, fontweight="bold")
        if row_i == 2:
            ax.set_xlabel("% queries → duoT5  (cost)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")

fig3.suptitle(f"Pareto frontiers — 3 signals × 3 nDCG cutoffs  ({len(qids)} queries)",
              fontsize=14, fontweight="bold")
fig3.tight_layout()
p3 = PLOTS_DIR / "pareto_comparison.png"
fig3.savefig(p3, dpi=150)
plt.close(fig3)
print(f"Plot 3 → {p3}")

# ── Plot 4: all signals overlaid, one panel per nDCG cutoff ──────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

kw_knee    = knee_by_signal[knee_winner]
kw_tau_key = "tau" if knee_winner == "GAP" else "tau_unc"

for ax4, (mkey, mlabel), (base_mono, base_duo) in zip(axes4, METRICS, BASELINES):
    for sig_name, sweep, knee_r, best_r, tk, col, _ in SIGNALS:
        costs = [r["pct_duo"] for r in sweep]
        vals  = [r[mkey]      for r in sweep]
        ax4.plot(costs, vals, color=col, marker="o", ms=2, lw=2, alpha=0.85,
                 label=sig_name)
        ax4.scatter([knee_r["pct_duo"]], [knee_r[mkey]],
                    color=col, s=100, zorder=6, marker="D")

    ax4.axhline(base_mono, color="black", ls=":", lw=1.1, alpha=0.5,
                label=f"monoT5 ({base_mono:.4f})")
    ax4.axhline(base_duo,  color=RED,    ls=":", lw=1.1, alpha=0.5,
                label=f"mono_duo ({base_duo:.4f})")
    ax4.axhspan(base_mono, base_duo, alpha=0.06, color="green")
    ax4.set_xlabel("% queries → duoT5  (cost)", fontsize=11)
    ax4.set_ylabel(mlabel, fontsize=11)
    ax4.set_title(f"Pareto  —  {mlabel}", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=9, loc="lower right")

# badge on rightmost panel only
axes4[2].text(0.98, 0.02,
              f"Knee winner: {knee_winner}\n"
              f"τ = {fmt_tau(kw_knee[kw_tau_key], '.4f')}\n"
              f"nDCG@1={kw_knee['ndcg1']:.4f}  "
              f"@5={kw_knee['ndcg5']:.4f}  "
              f"@10={kw_knee['ndcg10']:.4f}\n"
              f"saved {kw_knee['n_mono_only']}q  ({100 - kw_knee['pct_duo']:.1f}%)",
              transform=axes4[2].transAxes, ha="right", va="bottom",
              fontsize=9, fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", alpha=0.9))

fig4.suptitle(f"GAP vs MAU@5 vs MAU@10 — Pareto by nDCG cutoff  ({len(qids)} queries)",
              fontsize=13, fontweight="bold")
fig4.tight_layout()
p4 = PLOTS_DIR / "signal_comparison.png"
fig4.savefig(p4, dpi=160)
plt.close(fig4)
print(f"Plot 4 → {p4}")

# ── Plot 5: Entropy sweeps — 2×2 dual-axis (nDCG@1/@5/@10 + %duo vs τ_ent) ──
ENT_COLORS = {5: "#e377c2", 10: "#17becf", 20: "#bcbd22", 50: "#8c564b"}
fig5, axes5 = plt.subplots(2, 2, figsize=(18, 10))
axes5_flat = axes5.flatten()

for ax5, k in zip(axes5_flat, ENTROPY_KS):
    sweep_k = ent_sweeps[k]
    finite_k = [r for r in sweep_k if not math.isinf(r["tau_unc"]) and r["tau_unc"] > 0]
    tx   = [r["tau_unc"] for r in finite_k]
    n1_  = [r["ndcg1"]   for r in finite_k]
    n5_  = [r["ndcg5"]   for r in finite_k]
    n10_ = [r["ndcg10"]  for r in finite_k]
    cost = [r["pct_duo"] for r in finite_k]

    axr = ax5.twinx()
    ax5.plot(tx, n1_,  color="#e377c2", lw=1.6, marker="o", ms=2.5, label="nDCG@1")
    ax5.plot(tx, n5_,  color=BLUE,     lw=1.6, marker="o", ms=2.5, label="nDCG@5")
    ax5.plot(tx, n10_, color="#1f77b4",  lw=2.0, marker="o", ms=2.5, label="nDCG@10")
    axr.plot(tx, cost, color=ORANGE,   lw=1.8, marker="s", ms=2.5, ls="--",
             label="% duo")

    for val, col in [(mono_n1, "#e377c2"), (mono_n5, BLUE), (mono_ndcg, "#1f77b4")]:
        ax5.axhline(val, color=col, ls=":", lw=0.8, alpha=0.45)
    for val, col in [(duo_n1, "#e377c2"), (duo_n5, BLUE), (duo_ndcg, "#1f77b4")]:
        ax5.axhline(val, color=col, ls="-", lw=0.8, alpha=0.2)

    knee_k = ent_knees[k]
    if not math.isinf(knee_k["tau_unc"]) and knee_k["tau_unc"] > 0:
        ax5.axvline(knee_k["tau_unc"], color=CYAN, ls="-.", lw=1.5,
                    label=f"knee τ={knee_k['tau_unc']:.3f}\n"
                          f"  @10={knee_k['ndcg10']:.4f} saved {knee_k['n_mono_only']}q")

    ax5.set_xlabel(f"τ_ent  (H@{k} ≥ τ → duoT5)", fontsize=10)
    ax5.set_ylabel("nDCG", fontsize=10)
    axr.set_ylabel("% queries → duoT5", color=ORANGE, fontsize=9)
    axr.set_ylim(0, 110)
    ax5.set_title(f"H@{k}  (normalised rank entropy, top-{k} docs)", fontsize=11, fontweight="bold")

    lines_l, labels_l = ax5.get_legend_handles_labels()
    lines_r, labels_r = axr.get_legend_handles_labels()
    ax5.legend(lines_l + lines_r, labels_l + labels_r, fontsize=7.5, loc="center right")

fig5.suptitle(f"Rank-distribution entropy gating — H@k for k∈{{5,10,20,50}}  ({len(qids)} queries)",
              fontsize=13, fontweight="bold")
fig5.tight_layout()
p5 = PLOTS_DIR / "entropy_sweep.png"
fig5.savefig(p5, dpi=160)
plt.close(fig5)
print(f"Plot 5 → {p5}")

# ── Plot 6: Entropy vs GAP Pareto — one panel per nDCG cutoff ────────────────
ENT_SIGNAL_DEFS = [(f"H@{k}", ent_sweeps[k], ent_knees[k], "tau_unc", ENT_COLORS[k])
                   for k in ENTROPY_KS]

fig6, axes6 = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

for ax6, (mkey, mlabel), (base_mono, base_duo) in zip(axes6, METRICS, BASELINES):
    # GAP baseline
    gap_c = [r["pct_duo"] for r in gap_sweep]
    gap_v = [r[mkey]      for r in gap_sweep]
    ax6.plot(gap_c, gap_v, color=BLUE, lw=2.5, marker="o", ms=2, alpha=0.9,
             label="GAP  (top-1 vs top-2)", zorder=4)
    ax6.scatter([gap_knee["pct_duo"]], [gap_knee[mkey]],
                color=BLUE, s=100, marker="D", zorder=5)

    # Entropy curves
    for sig_name, sweep_k, knee_k, tk, col in ENT_SIGNAL_DEFS:
        c = [r["pct_duo"] for r in sweep_k]
        v = [r[mkey]      for r in sweep_k]
        ax6.plot(c, v, color=col, lw=2, marker="o", ms=2, alpha=0.85,
                 label=sig_name)
        ax6.scatter([knee_k["pct_duo"]], [knee_k[mkey]],
                    color=col, s=90, marker="D", zorder=5,
                    label=f"  knee τ={knee_k[tk]:.3f}  {mlabel}={knee_k[mkey]:.4f}  "
                          f"saved {knee_k['n_mono_only']}q")

    ax6.axhline(base_mono, color="black", ls=":", lw=1, alpha=0.5,
                label=f"monoT5 ({base_mono:.4f})")
    ax6.axhline(base_duo,  color=RED,    ls=":", lw=1, alpha=0.5,
                label=f"mono_duo ({base_duo:.4f})")
    ax6.axhspan(base_mono, base_duo, alpha=0.05, color="green")
    ax6.set_xlabel("% queries → duoT5  (cost)", fontsize=11)
    ax6.set_ylabel(mlabel, fontsize=11)
    ax6.set_title(f"{mlabel}  —  Entropy vs GAP Pareto", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=6.5, loc="lower right")

fig6.suptitle(f"Rank-entropy signals vs GAP — nDCG@1 / @5 / @10  ({len(qids)} queries)",
              fontsize=13, fontweight="bold")
fig6.tight_layout()
p6 = PLOTS_DIR / "entropy_vs_gap.png"
fig6.savefig(p6, dpi=160)
plt.close(fig6)
print(f"Plot 6 → {p6}")

print("\nDone. All outputs in:", OUT_DIR)
