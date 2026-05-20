"""Statistical significance testing between two generation runs.

The Wilcoxon signed-rank test compares paired observations (per-query QA
scores from two different model runs) and reports whether the median of the
paired differences is significantly different from zero. We use the normal
approximation; the per-qtype sample sizes are ~80-100, far above the n>=20
rule-of-thumb threshold where the approximation is reliable. No scipy — the
rest of the project deliberately stays scipy-free (see ``correlation.py``).
"""

from __future__ import annotations

import math
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .generation import _load_gen_run

router = APIRouter()

QTYPES = ("factoid", "yesno", "list", "summary")


# ---------- Wilcoxon signed-rank ----------------------------------------------


def _avg_ranks(xs: list[float]) -> list[float]:
    """1-indexed average ranks; ties share the mean of the ranks they cover."""
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _normal_cdf(z: float) -> float:
    """Phi(z) via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_signed_rank(deltas: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test using the normal approximation.

    Zero differences are dropped (Wilcoxon's original convention). For ties in
    |delta| we apply the standard tie correction to the variance.

    Returns: w_plus, w_minus, statistic (=min(w+, w-)), z, p_value, n_effective,
    n_zeros, mean_delta, median_delta.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n_zeros = len(deltas) - len(nonzero)
    n = len(nonzero)
    mean_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    sorted_all = sorted(deltas)
    if len(deltas) == 0:
        median_delta = 0.0
    else:
        m = len(sorted_all) // 2
        median_delta = (
            sorted_all[m]
            if len(sorted_all) % 2 == 1
            else (sorted_all[m - 1] + sorted_all[m]) / 2.0
        )

    if n == 0:
        return {
            "w_plus": 0.0,
            "w_minus": 0.0,
            "statistic": 0.0,
            "z": 0.0,
            "p_value": 1.0,
            "n_effective": 0,
            "n_zeros": n_zeros,
            "mean_delta": round(mean_delta, 6),
            "median_delta": round(median_delta, 6),
        }

    abs_deltas = [abs(d) for d in nonzero]
    ranks = _avg_ranks(abs_deltas)
    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    statistic = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    # Variance with tie correction.
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    tie_groups: dict[float, int] = {}
    for a in abs_deltas:
        tie_groups[a] = tie_groups.get(a, 0) + 1
    tie_correction = sum(t * (t * t - 1) for t in tie_groups.values() if t > 1) / 48.0
    var_w -= tie_correction
    if var_w <= 0:
        return {
            "w_plus": round(w_plus, 4),
            "w_minus": round(w_minus, 4),
            "statistic": round(statistic, 4),
            "z": 0.0,
            "p_value": 1.0,
            "n_effective": n,
            "n_zeros": n_zeros,
            "mean_delta": round(mean_delta, 6),
            "median_delta": round(median_delta, 6),
        }

    # Continuity correction toward the mean.
    diff = w_plus - mean_w
    if diff > 0:
        diff -= 0.5
    elif diff < 0:
        diff += 0.5
    z = diff / math.sqrt(var_w)
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    return {
        "w_plus": round(w_plus, 4),
        "w_minus": round(w_minus, 4),
        "statistic": round(statistic, 4),
        "z": round(z, 4),
        "p_value": round(p_value, 6),
        "n_effective": n,
        "n_zeros": n_zeros,
        "mean_delta": round(mean_delta, 6),
        "median_delta": round(median_delta, 6),
    }


# ---------- request/response schemas ------------------------------------------


class StatTestRequest(BaseModel):
    run_a: str  # generation run id (model under test)
    run_b: str  # generation run id (baseline)
    qtypes: Optional[list[str]] = None  # restrict scope; None = all + global


class StatBlock(BaseModel):
    n_pairs: int
    n_a_wins: int
    n_b_wins: int
    n_ties: int
    mean_a: float
    mean_b: float
    mean_delta: float
    median_delta: float
    w_plus: float
    w_minus: float
    statistic: float
    z: float
    p_value: float
    significant: bool  # p < 0.05


class StatTestResponse(BaseModel):
    run_a: str
    run_b: str
    retrieval_model_a: Optional[str] = None
    retrieval_model_b: Optional[str] = None
    k_a: Optional[int] = None
    k_b: Optional[int] = None
    metric_labels: dict[str, str]
    by_qtype: dict[str, StatBlock]
    global_block: StatBlock


# ---------- helpers -----------------------------------------------------------


METRIC_LABELS = {
    "factoid": "MRR",
    "yesno": "Accuracy",
    "list": "F1",
    "summary": "Judge",
    "global": "QA score",
}


def _block_for(pairs: list[tuple[float, float]]) -> StatBlock:
    n = len(pairs)
    if n == 0:
        zero = wilcoxon_signed_rank([])
        return StatBlock(
            n_pairs=0,
            n_a_wins=0,
            n_b_wins=0,
            n_ties=0,
            mean_a=0.0,
            mean_b=0.0,
            mean_delta=0.0,
            median_delta=0.0,
            w_plus=zero["w_plus"],
            w_minus=zero["w_minus"],
            statistic=zero["statistic"],
            z=zero["z"],
            p_value=zero["p_value"],
            significant=False,
        )
    deltas = [a - b for a, b in pairs]
    a_wins = sum(1 for d in deltas if d > 0)
    b_wins = sum(1 for d in deltas if d < 0)
    ties = n - a_wins - b_wins
    wsr = wilcoxon_signed_rank(deltas)
    return StatBlock(
        n_pairs=n,
        n_a_wins=a_wins,
        n_b_wins=b_wins,
        n_ties=ties,
        mean_a=round(sum(a for a, _ in pairs) / n, 4),
        mean_b=round(sum(b for _, b in pairs) / n, 4),
        mean_delta=round(sum(deltas) / n, 6),
        median_delta=wsr["median_delta"],
        w_plus=wsr["w_plus"],
        w_minus=wsr["w_minus"],
        statistic=wsr["statistic"],
        z=wsr["z"],
        p_value=wsr["p_value"],
        significant=wsr["p_value"] < 0.05 and n > 0,
    )


def _paired_qa(
    run_a: dict, run_b: dict
) -> tuple[dict[str, list[tuple[float, float]]], list[tuple[float, float]]]:
    """Pair per-query QA scores between two gen runs.

    Returns (by_qtype, global). A query contributes to a qtype only if BOTH
    runs have a numeric qa_score for it AND agree on qtype.
    """
    pq_a = run_a.get("per_query") or {}
    pq_b = run_b.get("per_query") or {}
    by_qtype: dict[str, list[tuple[float, float]]] = {}
    global_pairs: list[tuple[float, float]] = []
    for qid, ea in pq_a.items():
        eb = pq_b.get(qid)
        if not eb:
            continue
        sa = ea.get("qa_score")
        sb = eb.get("qa_score")
        if sa is None or sb is None:
            continue
        ta = ea.get("qtype")
        tb = eb.get("qtype")
        if ta and tb and ta == tb:
            by_qtype.setdefault(ta, []).append((float(sa), float(sb)))
        global_pairs.append((float(sa), float(sb)))
    return by_qtype, global_pairs


# ---------- endpoint ----------------------------------------------------------


@router.post("/statistical/compare", response_model=StatTestResponse)
def compare(req: StatTestRequest) -> StatTestResponse:
    run_a = _load_gen_run(req.run_a)
    if run_a is None:
        raise HTTPException(404, f"gen run not found: {req.run_a}")
    run_b = _load_gen_run(req.run_b)
    if run_b is None:
        raise HTTPException(404, f"gen run not found: {req.run_b}")

    by_qtype_pairs, global_pairs = _paired_qa(run_a, run_b)

    wanted = set(req.qtypes) if req.qtypes else set(QTYPES)
    out: dict[str, StatBlock] = {}
    for qt in QTYPES:
        if qt not in wanted:
            continue
        out[qt] = _block_for(by_qtype_pairs.get(qt, []))

    return StatTestResponse(
        run_a=req.run_a,
        run_b=req.run_b,
        retrieval_model_a=run_a.get("retrieval_model"),
        retrieval_model_b=run_b.get("retrieval_model"),
        k_a=run_a.get("k"),
        k_b=run_b.get("k"),
        metric_labels=METRIC_LABELS,
        by_qtype=out,
        global_block=_block_for(global_pairs),
    )
