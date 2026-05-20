"""Pairwise coverage analysis — monoT5 vs mono_duo (duoT5 tournament).

For each query, compares the top-20 ranking from monoT5 with the top-20
ranking after the full duoT5 tournament (380 ordered pair comparisons).

Shows:
  - How many pairs actually changed order (inversions) between the two models.
  - For each window size (top-5, top-10, top-15, top-20): how many of those
    inversions fall within that window, and how many comparisons it costs.
  - The sweet spot: fewest comparisons that captures most inversions.

Usage (from project root):
  python models/monot5/pairwise_coverage_analysis.py \\
      --mono  application/cache/runs/monot5/20260427T095643Z.json \\
      --duo   application/cache/runs/mono_duo/20260427T144541Z.json
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "models/monot5"

WINDOWS = [5, 10, 15, 20]   # top-K windows to evaluate


# ── helpers ───────────────────────────────────────────────────────────────────

def load_run(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def inversions_in_window(
    mono_order: list[str],
    duo_order:  list[str],
    window_k:   int,
) -> int:
    """Count (i,j) pairs where i<j in mono_order but reversed in duo_order,
    with both positions within the top-window_k of mono_order."""
    # only consider docs present in both rankings
    duo_rank = {doc: idx for idx, doc in enumerate(duo_order)}
    count = 0
    top = mono_order[:window_k]
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            di, dj = top[i], top[j]
            if di in duo_rank and dj in duo_rank:
                # inversion: di ranked after dj in duo but before in mono
                if duo_rank[di] > duo_rank[dj]:
                    count += 1
    return count


def total_inversions(mono_order: list[str], duo_order: list[str]) -> int:
    return inversions_in_window(mono_order, duo_order, len(mono_order))


def n_changed_positions(mono_order: list[str], duo_order: list[str]) -> int:
    """Docs in different positions between the two top-20 lists."""
    return sum(1 for a, b in zip(mono_order, duo_order) if a != b)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono", required=True, help="monoT5 run JSON")
    parser.add_argument("--duo",  required=True, help="mono_duo run JSON")
    args = parser.parse_args()

    mono_run = load_run(Path(args.mono))
    duo_run  = load_run(Path(args.duo))

    mono_pq: dict[str, Any] = mono_run["per_query"]
    duo_pq:  dict[str, Any] = duo_run["per_query"]

    # shared queries only
    qids = sorted(set(mono_pq) & set(duo_pq))
    print(f"Queries: {len(qids)} shared between both runs\n")

    # ── per-query analysis ────────────────────────────────────────────────────
    rows = []
    for qid in qids:
        mono_docs: list[str] = mono_pq[qid]["top_docids"]
        duo_docs:  list[str] = duo_pq[qid]["top_docids"]
        n = min(20, len(mono_docs), len(duo_docs))
        mono_top = mono_docs[:n]
        duo_top  = duo_docs[:n]

        total_inv  = total_inversions(mono_top, duo_top)
        n_changed  = n_changed_positions(mono_top, duo_top)
        qtype      = mono_pq[qid].get("qtype", "?")

        # inversions captured per window
        win_inv  = {k: inversions_in_window(mono_top, duo_top, k) for k in WINDOWS}
        # comparisons done per window (all ordered pairs within top-k)
        win_comp = {k: k * (k - 1) for k in WINDOWS}

        rows.append({
            "qid":        qid,
            "qtype":      qtype,
            "n_docs":     n,
            "n_changed":  n_changed,
            "total_inv":  total_inv,
            **{f"inv_top{k}": win_inv[k]  for k in WINDOWS},
            **{f"cmp_top{k}": win_comp[k] for k in WINDOWS},
        })

    # ── aggregate summary ─────────────────────────────────────────────────────
    n_q = len(rows)
    mean = lambda key: sum(r[key] for r in rows) / n_q

    full_cmp = 20 * 19  # 380 ordered pairs (full tournament)
    mean_total_inv = mean("total_inv")

    print("=" * 72)
    print(f"{'Window':>8}  {'Comparisons':>12}  {'Savings %':>10}  "
          f"{'Mean inv captured':>18}  {'Coverage %':>11}")
    print("─" * 72)
    for k in WINDOWS:
        cmp       = k * (k - 1)
        savings   = (1 - cmp / full_cmp) * 100
        mean_inv  = mean(f"inv_top{k}")
        coverage  = (mean_inv / mean_total_inv * 100) if mean_total_inv > 0 else 0.0
        print(f"  top-{k:<4}  {cmp:>12}  {savings:>9.1f}%  {mean_inv:>18.2f}  {coverage:>10.1f}%")
    print("─" * 72)
    print(f"  full    {full_cmp:>12}  {'0.0':>9}%  {mean_total_inv:>18.2f}  {'100.0':>10}%")
    print()
    print(f"Mean changed positions per query : {mean('n_changed'):.1f} / 20")
    print(f"Mean total inversions per query  : {mean_total_inv:.1f}")
    print(f"Queries with 0 inversions        : "
          f"{sum(1 for r in rows if r['total_inv'] == 0)} / {n_q}")
    print()

    # ── per-query table ───────────────────────────────────────────────────────
    col_w = [28, 10, 9, 10] + [10] * len(WINDOWS)
    header = (f"{'qid':<{col_w[0]}}  {'qtype':<{col_w[1]}}  "
              f"{'changed':>{col_w[2]}}  {'total_inv':>{col_w[3]}}" +
              "".join(f"  {'inv_top'+str(k):>{col_w[4+i]}}" for i, k in enumerate(WINDOWS)))
    print(header)
    print("─" * len(header))
    for r in rows:
        inv_cols = "".join(f"  {r['inv_top'+str(k)]:>{col_w[4+i]}}" for i, k in enumerate(WINDOWS))
        print(f"{r['qid']:<{col_w[0]}}  {r['qtype']:<{col_w[1]}}  "
              f"{r['n_changed']:>{col_w[2]}}  {r['total_inv']:>{col_w[3]}}{inv_cols}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "pairwise_coverage.csv"
    fieldnames = ["qid", "qtype", "n_docs", "n_changed", "total_inv"] + \
                 [f"inv_top{k}" for k in WINDOWS] + \
                 [f"cmp_top{k}" for k in WINDOWS]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV → {csv_path}")

    # ── charts ────────────────────────────────────────────────────────────────
    sns.set_theme(style="darkgrid")
    plt.style.use("ggplot")

    labels   = [f"top-{k}" for k in WINDOWS] + ["full"]
    cmp_vals = [k * (k - 1) for k in WINDOWS] + [full_cmp]
    cov_vals = [mean(f"inv_top{k}") / mean_total_inv * 100
                if mean_total_inv > 0 else 0.0 for k in WINDOWS] + [100.0]

    # Chart 1 — comparisons vs coverage (dual axis)
    fig1: Any
    ax1: Any
    ax1b: Any
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1b = ax1.twinx()

    x = range(len(labels))
    bars = ax1.bar(x, cmp_vals, color="#4c72b0", width=0.4, label="Comparisons")
    ax1b.plot(x, cov_vals, color="#dd8452", marker="o", linewidth=2, label="Coverage %")
    ax1b.set_ylim(0, 115)

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Window")
    ax1.set_ylabel("Number of pairwise comparisons")
    ax1b.set_ylabel("Inversions captured (%)")
    ax1.set_title(
        f"Comparisons vs inversion coverage — monoT5 → mono_duo  "
        f"({len(qids)} queries, top-20)"
    )

    for bar, cv in zip(bars, cov_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                 f"{cv:.0f}%", ha="center", va="bottom", fontsize=9, color="#dd8452")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig1.tight_layout()
    p1 = OUT_DIR / "pairwise_coverage_plot.png"
    fig1.savefig(p1, dpi=150)
    print(f"Plot 1 → {p1}")

    # Chart 2 — distribution of total inversions per query
    fig2: Any
    ax2: Any
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    inv_dist = [r["total_inv"] for r in rows]
    max_inv  = max(inv_dist) if inv_dist else 1
    bins = list(range(0, max_inv + 2))
    ax2.hist(inv_dist, bins=bins, color="#4c72b0", edgecolor="white", align="left")
    ax2.axvline(mean_total_inv, color="red", linestyle="--", linewidth=1.5,
                label=f"mean = {mean_total_inv:.1f}")
    ax2.set_xlabel("Total inversions per query")
    ax2.set_ylabel("Number of queries")
    ax2.set_title(
        f"Distribution of inversions (monoT5 → mono_duo top-20)  "
        f"({len(qids)} queries)"
    )
    ax2.legend()
    fig2.tight_layout()
    p2 = OUT_DIR / "pairwise_inversions_dist.png"
    fig2.savefig(p2, dpi=150)
    print(f"Plot 2 → {p2}")

    plt.show()


if __name__ == "__main__":
    main()
