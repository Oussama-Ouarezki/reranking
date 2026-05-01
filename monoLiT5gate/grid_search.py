"""MonoGatedLiT5 grid search — entropy H@20 and H@50 as gating signals for LiT5.

Uses pre-computed per-query results from cached runs to simulate the
MonoEntropyGatedLiT5 cascade at every threshold without re-running any model.

Signal:
  H@k  — normalised rank-distribution entropy over top-k monoT5 P(true) scores
          H_norm = -Σ p_i log(p_i) / log(k),  p_i = s_i / Σ s_j
          send to LiT5 when H_norm(q) >= τ  (high entropy = flat/uncertain ranking)

Two windows compared:  H@20  and  H@50

Fixed data sources:
  mono  cache  : application/cache/runs/monot5/20260427T171106Z.json  (nDCG@10=0.8728)
  lit5  cache  : application/cache/runs/lit5/20260429T115008Z.json    (nDCG@10=0.8735)
  score cache  : models/monot5/margin_scores_cache.json
                 (P(true) scores — covers 100% of the mono cache top-20 doc pool)

Outputs:
  results/entropy20_sweep.csv
  results/entropy50_sweep.csv
  results/summary.txt
  plots/entropy_sweep.png       — dual-axis nDCG/@pct side-by-side H@20 vs H@50
  plots/pareto_comparison.png   — Pareto curves: 3 rows (nDCG@1/@5/@10) × 2 cols
  plots/signal_comparison.png   — H@20 vs H@50 overlaid per nDCG cutoff
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

MONO_CACHE  = ROOT / "application/cache/runs/monot5/20260427T171106Z.json"
LIT5_CACHE  = ROOT / "application/cache/runs/lit5/20260429T115008Z.json"
SCORE_CACHE = ROOT / "models/monot5/margin_scores_cache.json"

OUT_DIR     = Path(__file__).resolve().parent
RESULTS_DIR = OUT_DIR / "results"
PLOTS_DIR   = OUT_DIR / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── theme ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid")
plt.style.use("ggplot")

# ── entropy tau grid ──────────────────────────────────────────────────────────
# H_norm ∈ [0, 1]; send to LiT5 when H_norm >= τ
# τ=0 → always LiT5 (H always >= 0)
# τ=1 → never LiT5 (H_norm never exceeds 1)
ENT_GRID: list[float] = sorted({
    0.0,
    *[round(v, 5) for v in np.linspace(0.05, 0.98, 120)],
    0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.98, 0.99,
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


def _norm_entropy(vals: list[float]) -> float | None:
    """Normalised Shannon entropy of a score vector in [0, 1].

    Scores are treated as un-normalised weights; we normalise to a probability
    simplex, then compute H / log(k) so the result is always in [0, 1].
    High value → flat distribution → monoT5 is uncertain about the ranking.
    """
    if not vals or sum(vals) == 0:
        return None
    s = sum(vals)
    probs = [v / s for v in vals]
    h = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
    return h / math.log(len(probs))


# ── load data ─────────────────────────────────────────────────────────────────

print("Loading caches …")
mono_run   = load_json(MONO_CACHE)
lit5_run   = load_json(LIT5_CACHE)
scores_raw = load_json(SCORE_CACHE)

mono_pq: dict = mono_run["per_query"]
lit5_pq: dict = lit5_run["per_query"]

qids = sorted(set(mono_pq) & set(lit5_pq))
print(f"  mono  cache : {len(mono_pq)} queries  nDCG@10={mono_run['aggregate']['ndcg_at']['10']:.4f}")
print(f"  lit5  cache : {len(lit5_pq)} queries  nDCG@10={lit5_run['aggregate']['ndcg_at']['10']:.4f}")
print(f"  score cache : {len(scores_raw)} queries")
print(f"  shared qids : {len(qids)}")

# ── compute per-query entropy signals ─────────────────────────────────────────

entropy_scores: dict[int, dict[str, float]] = {k: {} for k in ENTROPY_KS}

for qid in qids:
    if qid not in scores_raw:
        continue
    doc_scores = scores_raw[qid]  # {docid: P(true)} for all 50 BM25 docs
    # Sort all 50 BM25 docs by P(true) descending → full monoT5 ranking
    all_ranked = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)

    for k in ENTROPY_KS:
        topk = all_ranked[:k]   # top-k from the full 50-doc ranking
        vals = [doc_scores[d] for d in topk]
        h = _norm_entropy(vals)
        if h is not None:
            entropy_scores[k][qid] = h

print(f"\nSignal coverage:")
for k in ENTROPY_KS:
    n = sum(1 for q in qids if q in entropy_scores[k])
    print(f"  H@{k:<3d} : {n}/{len(qids)} queries")

# distribution stats
def _pct_summary(vals: list[float], label: str) -> None:
    if not vals:
        return
    mean = sum(vals) / len(vals)
    print(f"\n{label}  (mean={mean:.4f}):")
    for pct in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        idx = min(int(pct / 100 * len(vals)), len(vals) - 1)
        print(f"  p{pct:3d}: {vals[idx]:.6f}")

for k in ENTROPY_KS:
    ev = sorted(entropy_scores[k][q] for q in qids if q in entropy_scores[k])
    _pct_summary(ev, f"H@{k}  (normalised rank entropy)")

# ── simulation ────────────────────────────────────────────────────────────────

def _aggregate(entries: list[dict], use_lit5_flags: list[bool], tau_val: float) -> dict:
    n1, n5, n10 = [], [], []
    mrr_vals, map_vals = [], []
    n_lit5 = sum(use_lit5_flags)
    for entry in entries:
        n1.append(get_metric(entry,  "ndcg_at", "1"))
        n5.append(get_metric(entry,  "ndcg_at", "5"))
        n10.append(get_metric(entry, "ndcg_at", "10"))
        mrr_vals.append(get_metric(entry, "mrr_at", "10"))
        map_vals.append(get_metric(entry, "map_at", "10"))
    n = len(entries)
    n_total = len(qids)
    return {
        "tau_ent":     tau_val,
        "ndcg1":       sum(n1)       / n if n else 0.0,
        "ndcg5":       sum(n5)       / n if n else 0.0,
        "ndcg10":      sum(n10)      / n if n else 0.0,
        "mrr10":       sum(mrr_vals) / n if n else 0.0,
        "map10":       sum(map_vals) / n if n else 0.0,
        "pct_lit5":    n_lit5 / n_total * 100 if n_total else 0.0,
        "n_lit5":      n_lit5,
        "n_mono_only": n_total - n_lit5,
        "n_queries":   n_total,
    }


def simulate(tau: float, score_dict: dict[str, float]) -> dict:
    entries, flags = [], []
    for qid in qids:
        use_lit5 = qid in score_dict and score_dict[qid] >= tau
        entry = (lit5_pq if use_lit5 else mono_pq).get(qid) or mono_pq.get(qid)
        if entry is None:
            continue
        entries.append(entry)
        flags.append(use_lit5)
    return _aggregate(entries, flags, tau)


# ── run sweeps ────────────────────────────────────────────────────────────────

print("\nRunning H@20 entropy sweep …")
ent20_sweep = [simulate(t, entropy_scores[20]) for t in ENT_GRID]

print("Running H@50 entropy sweep …")
ent50_sweep = [simulate(t, entropy_scores[50]) for t in ENT_GRID]

sweeps: dict[int, list[dict]] = {20: ent20_sweep, 50: ent50_sweep}

# reference baselines
mono_n1   = _aggregate(
    [mono_pq[q] for q in qids if q in mono_pq],
    [False] * len(qids), 0.0)["ndcg1"]
mono_n5   = _aggregate(
    [mono_pq[q] for q in qids if q in mono_pq],
    [False] * len(qids), 0.0)["ndcg5"]
mono_ndcg = _aggregate(
    [mono_pq[q] for q in qids if q in mono_pq],
    [False] * len(qids), 0.0)["ndcg10"]

lit5_n1   = _aggregate(
    [lit5_pq[q] for q in qids if q in lit5_pq],
    [True] * len(qids), math.inf)["ndcg1"]
lit5_n5   = _aggregate(
    [lit5_pq[q] for q in qids if q in lit5_pq],
    [True] * len(qids), math.inf)["ndcg5"]
lit5_ndcg = _aggregate(
    [lit5_pq[q] for q in qids if q in lit5_pq],
    [True] * len(qids), math.inf)["ndcg10"]

print(f"\n  Baseline monoT5      nDCG@1={mono_n1:.4f}  @5={mono_n5:.4f}  @10={mono_ndcg:.4f}")
print(f"  Baseline LiT5 (all)  nDCG@1={lit5_n1:.4f}  @5={lit5_n5:.4f}  @10={lit5_ndcg:.4f}")

# ── find best tau and Pareto knee ─────────────────────────────────────────────

best: dict[int, dict] = {}
knees: dict[int, dict] = {}

for k in ENTROPY_KS:
    best[k] = max(sweeps[k], key=lambda r: r["ndcg10"])
    costs_k = [r["pct_lit5"] for r in sweeps[k]]
    ndcgs_k = [r["ndcg10"]   for r in sweeps[k]]
    ki = knee_index(costs_k, ndcgs_k)
    knees[k] = sweeps[k][ki]


def fmt_tau(t: float, fmt: str = ".5f") -> str:
    return "∞ (always LiT5)" if math.isinf(t) else f"{t:{fmt}}"


# ── print tables ──────────────────────────────────────────────────────────────

HDR = (f"{'τ_ent':>16}  {'nDCG@1':>7}  {'nDCG@5':>7}  {'nDCG@10':>8}  "
       f"{'MRR@10':>8}  {'%lit5':>6}  {'n_lit5':>6}  {'n_saved':>7}")
SEP = "─" * 80


def _row(r: dict, best_r: dict, knee_r: dict) -> str:
    marker = " ◄ BEST" if r is best_r else (" ◄ KNEE" if r is knee_r else "")
    return (f"{fmt_tau(r['tau_ent']):>16}  {r['ndcg1']:>7.4f}  {r['ndcg5']:>7.4f}  "
            f"{r['ndcg10']:>8.4f}  {r['mrr10']:>8.4f}  {r['pct_lit5']:>5.1f}%  "
            f"{r['n_lit5']:>6}  {r['n_mono_only']:>7}{marker}")


for k in ENTROPY_KS:
    print(f"\n{'=' * 80}")
    print(f"H@{k} SWEEP  (send to LiT5 when H@{k} ≥ τ_ent,  H normalised to [0,1])")
    print("=" * 80)
    print(HDR); print(SEP)
    for r in sweeps[k]:
        print(_row(r, best[k], knees[k]))


# ── summary ───────────────────────────────────────────────────────────────────

def _sig_block(k: int) -> str:
    r_best = best[k]
    r_knee = knees[k]
    return f"""\
──────────────────────────────────────────────────────────────────────────────
H@{k}  — Rank-distribution entropy  (top-{k} monoT5 P(true) scores, H/log({k}))
  Best τ     : {fmt_tau(r_best['tau_ent'])}
  Best nDCG  : @1={r_best['ndcg1']:.4f}  @5={r_best['ndcg5']:.4f}  @10={r_best['ndcg10']:.4f}
  Queries → LiT5     : {r_best['n_lit5']} / {len(qids)}  ({r_best['pct_lit5']:.1f}%)
  Queries kept mono  : {r_best['n_mono_only']} / {len(qids)}  ({100 - r_best['pct_lit5']:.1f}%)
  Pareto knee τ : {fmt_tau(r_knee['tau_ent'])}
    nDCG  @1={r_knee['ndcg1']:.4f}  @5={r_knee['ndcg5']:.4f}  @10={r_knee['ndcg10']:.4f}
    saved {r_knee['n_mono_only']}q  ({100 - r_knee['pct_lit5']:.1f}% cost reduction)"""


winner_k   = max(ENTROPY_KS, key=lambda k: best[k]["ndcg10"])
winner_r   = best[winner_k]
knee_winner_k = max(ENTROPY_KS, key=lambda k: knees[k]["ndcg10"])
kw_knee    = knees[knee_winner_k]

summary = textwrap.dedent(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MonoGatedLiT5 Grid Search — H@20 vs H@50  (nDCG @1 / @5 / @10)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Baselines                   nDCG@1    nDCG@5   nDCG@10
  monoT5  (never LiT5)     {mono_n1:.4f}    {mono_n5:.4f}    {mono_ndcg:.4f}
  LiT5    (always LiT5)    {lit5_n1:.4f}    {lit5_n5:.4f}    {lit5_ndcg:.4f}
  Δ (LiT5 − mono)         {lit5_n1-mono_n1:+.4f}   {lit5_n5-mono_n5:+.4f}   {lit5_ndcg-mono_ndcg:+.4f}
  Total queries            {len(qids)}
""").strip()

for k in ENTROPY_KS:
    summary += "\n\n" + _sig_block(k)

summary += f"""

──────────────────────────────────────────────────────────────────────────────
WINNER (best nDCG@10): H@{winner_k}  τ={fmt_tau(winner_r['tau_ent'])}
  nDCG  @1={winner_r['ndcg1']:.4f}  @5={winner_r['ndcg5']:.4f}  @10={winner_r['ndcg10']:.4f}
  Δ vs monoT5: @1={winner_r['ndcg1']-mono_n1:+.4f}  @5={winner_r['ndcg5']-mono_n5:+.4f}  @10={winner_r['ndcg10']-mono_ndcg:+.4f}

WINNER (Pareto knee): H@{knee_winner_k}  τ={fmt_tau(kw_knee['tau_ent'])}
  nDCG  @1={kw_knee['ndcg1']:.4f}  @5={kw_knee['ndcg5']:.4f}  @10={kw_knee['ndcg10']:.4f}
  Saved {kw_knee['n_mono_only']} queries  ({100 - kw_knee['pct_lit5']:.1f}% cost reduction)
"""

print("\n" + summary)
(RESULTS_DIR / "summary.txt").write_text(summary + "\n")
print(f"Summary → {RESULTS_DIR / 'summary.txt'}")


# ── save CSVs ─────────────────────────────────────────────────────────────────

FIELDS = ["tau_ent", "ndcg1", "ndcg5", "ndcg10", "mrr10", "map10",
          "pct_lit5", "n_lit5", "n_mono_only", "n_queries"]

for k in ENTROPY_KS:
    out_csv = RESULTS_DIR / f"entropy{k}_sweep.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in sweeps[k]:
            row = dict(r)
            row["tau_ent"] = "inf" if math.isinf(r["tau_ent"]) else r["tau_ent"]
            w.writerow(row)
    print(f"H@{k} CSV → {out_csv}")


# ── colours ───────────────────────────────────────────────────────────────────

BLUE   = "#4c72b0"
ORANGE = "#dd8452"
GREEN  = "#2ca02c"
RED    = "#d62728"
CYAN   = "#17becf"
PINK   = "#e377c2"

ENT_COLORS = {20: "#bcbd22", 50: "#8c564b"}


# ── Plot 1: dual-axis sweep — H@20 and H@50 side by side ─────────────────────

fig1, axes1 = plt.subplots(1, 2, figsize=(18, 6))

for ax, k in zip(axes1, ENTROPY_KS):
    col_k = ENT_COLORS[k]
    sweep_k = sweeps[k]
    finite  = [r for r in sweep_k if not math.isinf(r["tau_ent"]) and r["tau_ent"] > 0]

    tx   = [r["tau_ent"]    for r in finite]
    n1_  = [r["ndcg1"]      for r in finite]
    n5_  = [r["ndcg5"]      for r in finite]
    n10_ = [r["ndcg10"]     for r in finite]
    cost = [r["pct_lit5"]   for r in finite]

    axr = ax.twinx()
    ax.plot(tx, n1_,  color=PINK,   lw=1.6, marker="o", ms=2.5, label="nDCG@1")
    ax.plot(tx, n5_,  color=BLUE,   lw=1.6, marker="o", ms=2.5, label="nDCG@5")
    ax.plot(tx, n10_, color="#1f77b4", lw=2.2, marker="o", ms=2.5, label="nDCG@10", alpha=0.9)
    saved = [r["n_mono_only"] for r in finite]
    axr.plot(tx, cost, color=ORANGE, lw=1.8, marker="s", ms=2.5, ls="--",
             label="% queries → LiT5")

    # monoT5 baselines (dotted) and LiT5 baselines (faint solid)
    for mono_val, lit5_val, col in [
        (mono_n1,   lit5_n1,   PINK),
        (mono_n5,   lit5_n5,   BLUE),
        (mono_ndcg, lit5_ndcg, "#1f77b4"),
    ]:
        ax.axhline(mono_val, color=col, ls=":", lw=0.9, alpha=0.45)
        ax.axhline(lit5_val, color=col, ls="-", lw=0.9, alpha=0.22)

    best_k = best[k]
    knee_k = knees[k]
    if not math.isinf(best_k["tau_ent"]) and best_k["tau_ent"] > 0:
        ax.axvline(best_k["tau_ent"], color=GREEN, ls="-", lw=1.5,
                   label=f"best τ={best_k['tau_ent']:.4f}  @10={best_k['ndcg10']:.4f}  saved {best_k['n_mono_only']}q")
    if not math.isinf(knee_k["tau_ent"]) and knee_k["tau_ent"] > 0:
        ax.axvline(knee_k["tau_ent"], color=CYAN, ls="-.", lw=1.5,
                   label=f"knee τ={knee_k['tau_ent']:.4f}  @10={knee_k['ndcg10']:.4f}  saved {knee_k['n_mono_only']}q")

    ax.set_xlabel(f"τ_ent  (H@{k} ≥ τ → LiT5)", fontsize=11)
    ax.set_ylabel("nDCG", fontsize=11)
    axr.set_ylabel("% queries → LiT5", color=ORANGE, fontsize=10)
    axr.set_ylim(0, 110)
    ax.set_title(f"H@{k}  — monoT5 → LiT5 entropy gate  (top-{k} docs)",
                 fontsize=12, fontweight="bold")

    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = axr.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="center right", fontsize=8)

fig1.suptitle(f"Entropy-gated monoT5 → LiT5  —  H@20 vs H@50  ({len(qids)} queries)",
              fontsize=13, fontweight="bold")
fig1.tight_layout()
p1 = PLOTS_DIR / "entropy_sweep.png"
fig1.savefig(p1, dpi=160)
plt.close(fig1)
print(f"Plot 1 → {p1}")


# ── Plot 2: Pareto — 3 rows (nDCG@1/@5/@10), 2 cols (H@20 / H@50) ────────────

METRICS   = [("ndcg1", "nDCG@1"), ("ndcg5", "nDCG@5"), ("ndcg10", "nDCG@10")]
BASELINES = [(mono_n1, lit5_n1), (mono_n5, lit5_n5), (mono_ndcg, lit5_ndcg)]

fig2, axes2 = plt.subplots(3, 2, figsize=(14, 15), sharey="row")

for row_i, ((mkey, mlabel), (base_mono, base_lit5)) in enumerate(zip(METRICS, BASELINES)):
    for col_i, k in enumerate(ENTROPY_KS):
        ax = axes2[row_i, col_i]
        col_k = ENT_COLORS[k]
        sweep_k = sweeps[k]
        costs = [r["pct_lit5"] for r in sweep_k]
        vals  = [r[mkey]       for r in sweep_k]

        ax.plot(costs, vals, color=col_k, marker="o", ms=3, lw=2, alpha=0.9)

        knee_k = knees[k]
        best_k = best[k]
        ax.scatter([knee_k["pct_lit5"]], [knee_k[mkey]],
                   color=CYAN, s=100, zorder=5,
                   label=f"knee  τ={knee_k['tau_ent']:.4f}\n  {mlabel}={knee_k[mkey]:.4f}\n  saved {knee_k['n_mono_only']}q")
        ax.scatter([best_k["pct_lit5"]], [best_k[mkey]],
                   color=GREEN, s=100, zorder=5, marker="*",
                   label=f"best  τ={best_k['tau_ent']:.4f}\n  {mlabel}={best_k[mkey]:.4f}")

        ax.axhline(base_mono, color="gray", ls=":",  lw=1,   alpha=0.7,
                   label=f"monoT5  ({base_mono:.4f})")
        ax.axhline(base_lit5, color=RED,   ls=":",  lw=1,   alpha=0.6,
                   label=f"LiT5 all ({base_lit5:.4f})")
        ax.axhspan(base_mono, base_lit5, alpha=0.05, color="green")

        # annotate key tau points
        for r in sweep_k:
            if round(r["tau_ent"], 2) in {0.5, 0.7, 0.8, 0.9, 0.95}:
                ax.annotate(f"{r['tau_ent']:.2f}", (r["pct_lit5"], r[mkey]),
                            textcoords="offset points", xytext=(3, 3), fontsize=6.5)

        if col_i == 0:
            ax.set_ylabel(mlabel, fontsize=11)
        if row_i == 0:
            ax.set_title(f"H@{k}  signal", fontsize=12, fontweight="bold")
        if row_i == 2:
            ax.set_xlabel("% queries → LiT5  (cost)", fontsize=10)
        ax.legend(fontsize=7, loc="lower right")

fig2.suptitle(f"Pareto frontiers — H@20 vs H@50  ×  nDCG@1/@5/@10  ({len(qids)} queries)",
              fontsize=13, fontweight="bold")
fig2.tight_layout()
p2 = PLOTS_DIR / "pareto_comparison.png"
fig2.savefig(p2, dpi=150)
plt.close(fig2)
print(f"Plot 2 → {p2}")


# ── Plot 3: H@20 vs H@50 overlaid — one panel per nDCG cutoff ────────────────

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6), sharey=False)

for ax3, (mkey, mlabel), (base_mono, base_lit5) in zip(axes3, METRICS, BASELINES):
    for k in ENTROPY_KS:
        col_k   = ENT_COLORS[k]
        sweep_k = sweeps[k]
        knee_k  = knees[k]
        costs = [r["pct_lit5"] for r in sweep_k]
        vals  = [r[mkey]       for r in sweep_k]
        ax3.plot(costs, vals, color=col_k, marker="o", ms=2, lw=2, alpha=0.85,
                 label=f"H@{k}")
        ax3.scatter([knee_k["pct_lit5"]], [knee_k[mkey]],
                    color=col_k, s=100, zorder=6, marker="D",
                    label=f"  H@{k} knee τ={knee_k['tau_ent']:.3f}  {mlabel}={knee_k[mkey]:.4f}  saved {knee_k['n_mono_only']}q ({100-knee_k['pct_lit5']:.1f}%)")

    ax3.axhline(base_mono, color="black", ls=":", lw=1.1, alpha=0.5,
                label=f"monoT5 ({base_mono:.4f})")
    ax3.axhline(base_lit5, color=RED,    ls=":", lw=1.1, alpha=0.5,
                label=f"LiT5 all ({base_lit5:.4f})")
    ax3.axhspan(base_mono, base_lit5, alpha=0.05, color="green")
    ax3.set_xlabel("% queries → LiT5  (cost)", fontsize=11)
    ax3.set_ylabel(mlabel, fontsize=11)
    ax3.set_title(f"Pareto  —  {mlabel}", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8.5, loc="lower right")

# summary badge on rightmost panel
kw = kw_knee
fig3.axes[-1].text(
    0.98, 0.02,
    f"Knee winner: H@{knee_winner_k}\n"
    f"τ = {fmt_tau(kw['tau_ent'], '.4f')}\n"
    f"nDCG@1={kw['ndcg1']:.4f}  "
    f"@5={kw['ndcg5']:.4f}  "
    f"@10={kw['ndcg10']:.4f}\n"
    f"saved {kw['n_mono_only']}q  ({100 - kw['pct_lit5']:.1f}%)",
    transform=fig3.axes[-1].transAxes, ha="right", va="bottom",
    fontsize=9, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="black", alpha=0.9))

fig3.suptitle(f"H@20 vs H@50 — monoT5→LiT5 entropy gate  ({len(qids)} queries)",
              fontsize=13, fontweight="bold")
fig3.tight_layout()
p3 = PLOTS_DIR / "signal_comparison.png"
fig3.savefig(p3, dpi=160)
plt.close(fig3)
print(f"Plot 3 → {p3}")

print("\nDone. All outputs in:", OUT_DIR)
