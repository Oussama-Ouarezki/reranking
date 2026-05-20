"""Entropy gate: H@20 → duoT5 or Qwen.

For each query compute H@20 = Shannon entropy of softmax(qwen_probs[:20]).
If H@20 > τ  → use duoT5 ranking (already cached in scores_duot5.jsonl).
Else          → keep pure-Qwen ranking.

Sweeps τ on 41 points over [H_min, H_max] and reports nDCG@1/5/10 + MRR@10.
Prints best τ and comparison table: Qwen-only vs duoT5-only vs best gate.

Reads:
  qwen4b_uncertainty/data/qwen_scores_test.jsonl
  qwenDuot5/scores_duot5.jsonl
  data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv
Writes:
  qwenDuot5/entropy_gate20_sweep.tsv
  qwenDuot5/entropy_gate20_best.json
"""

import json
import math
from pathlib import Path

import numpy as np
import ir_measures
from ir_measures import nDCG, RR, Qrel, ScoredDoc

ROOT = Path(__file__).resolve().parent.parent
QWEN_F   = ROOT / "qwen4b_uncertainty/data/qwen_scores_test.jsonl"
DUOT5_F  = ROOT / "qwenDuot5/scores_duot5.jsonl"
QRELS_F  = ROOT / "data/bioasq/raw/Task13BGoldenEnriched/qrels.tsv"
OUT_TSV  = ROOT / "qwenDuot5/entropy_gate20_sweep.tsv"
OUT_BEST = ROOT / "qwenDuot5/entropy_gate20_best.json"

METRICS      = [nDCG @ 1, nDCG @ 5, nDCG @ 10, RR @ 10]
METRIC_NAMES = ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@10"]
N_TAU        = 41
TOP_K        = 20   # entropy window


# ── helpers ────────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def shannon_entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


# ── load data ──────────────────────────────────────────────────────────────────

def load_qrels() -> list[Qrel]:
    qrels = []
    with QRELS_F.open() as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                qrels.append(Qrel(p[0], p[1], int(p[2])))
    return qrels


def load_qwen() -> dict[str, dict]:
    """Returns {qid: {"entropy": float, "run": list[ScoredDoc]}}"""
    data = {}
    with QWEN_F.open() as f:
        for line in f:
            row = json.loads(line)
            qid    = row["qid"]
            scores = row["scores"]          # already ranked by qwen_prob desc
            probs  = np.array([s["qwen_prob"] for s in scores[:TOP_K]], dtype=float)
            dist   = softmax(probs)
            H      = shannon_entropy(dist)
            run    = [ScoredDoc(query_id=qid, doc_id=s["docid"], score=s["qwen_prob"])
                      for s in scores]
            data[qid] = {"entropy": H, "run": run}
    return data


def load_duot5() -> dict[str, list[ScoredDoc]]:
    """Returns {qid: list[ScoredDoc]} from duoT5 cached results."""
    data = {}
    with DUOT5_F.open() as f:
        for line in f:
            row = json.loads(line)
            qid  = row["qid"]
            data[qid] = [
                ScoredDoc(query_id=qid, doc_id=item["docid"], score=item["duo_score"])
                for item in row["ranked"]
            ]
    return data


# ── evaluation ─────────────────────────────────────────────────────────────────

def evaluate(run: list[ScoredDoc], qrels: list[Qrel]) -> dict[str, float]:
    res = ir_measures.calc_aggregate(METRICS, qrels, run)
    return {METRIC_NAMES[i]: float(res[METRICS[i]]) for i in range(len(METRICS))}


def build_gated_run(qwen_data: dict, duot5_data: dict, tau: float) -> tuple[list[ScoredDoc], int]:
    """Returns (run, n_routed_to_duot5)."""
    run      = []
    n_routed = 0
    for qid, entry in qwen_data.items():
        if entry["entropy"] > tau and qid in duot5_data:
            run.extend(duot5_data[qid])
            n_routed += 1
        else:
            run.extend(entry["run"])
    return run, n_routed


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data ...")
    qrels      = load_qrels()
    qwen_data  = load_qwen()
    duot5_data = load_duot5()
    n_queries  = len(qwen_data)
    print(f"  {n_queries} queries | {len(qrels)} qrel rows")

    # Baselines
    qwen_run  = [doc for entry in qwen_data.values() for doc in entry["run"]]
    duo_run   = [doc for docs in duot5_data.values() for doc in docs]
    qwen_res  = evaluate(qwen_run, qrels)
    duo_res   = evaluate(duo_run, qrels)

    print("\nBaselines:")
    print(f"  {'metric':<10}  {'Qwen-only':>10}  {'duoT5-only':>10}")
    for m in METRIC_NAMES:
        print(f"  {m:<10}  {qwen_res[m]:>10.4f}  {duo_res[m]:>10.4f}")

    # Entropy range
    entropies = [entry["entropy"] for entry in qwen_data.values()]
    tau_min, tau_max = min(entropies), max(entropies)
    taus = np.linspace(tau_min, tau_max, N_TAU)

    print(f"\nSweeping {N_TAU} τ values in [{tau_min:.3f}, {tau_max:.3f}] ...")

    rows = []
    for tau in taus:
        run, n_routed = build_gated_run(qwen_data, duot5_data, tau)
        res = evaluate(run, qrels)
        pct = 100.0 * n_routed / n_queries
        rows.append({"tau": tau, "pct_routed": pct, **res})

    # Write sweep TSV
    with OUT_TSV.open("w") as f:
        f.write("\t".join(["tau", "pct_routed"] + METRIC_NAMES) + "\n")
        for r in rows:
            vals = [f"{r['tau']:.4f}", f"{r['pct_routed']:.1f}"] + \
                   [f"{r[m]:.4f}" for m in METRIC_NAMES]
            f.write("\t".join(vals) + "\n")
    print(f"Sweep → {OUT_TSV}")

    # Best τ by nDCG@10
    best = max(rows, key=lambda r: r["ndcg@10"])
    with OUT_BEST.open("w") as f:
        json.dump(best, f, indent=2)
    print(f"Best  → {OUT_BEST}")

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'metric':<10}  {'Qwen-only':>10}  {'duoT5-only':>10}  {'best gate':>10}  {'Δ vs Qwen':>10}")
    print("=" * 70)
    for m in METRIC_NAMES:
        delta = best[m] - qwen_res[m]
        sign  = "+" if delta >= 0 else ""
        print(f"  {m:<10}  {qwen_res[m]:>10.4f}  {duo_res[m]:>10.4f}  {best[m]:>10.4f}  {sign}{delta:>9.4f}")
    print("=" * 70)
    print(f"  Best τ = {best['tau']:.4f}  |  {best['pct_routed']:.1f}% of queries routed to duoT5")


if __name__ == "__main__":
    main()
