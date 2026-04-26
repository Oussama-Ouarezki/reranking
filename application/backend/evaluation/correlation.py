"""Spearman rank correlation matrices.

Used by the generation page: per question type, build a matrix correlating
retrieval metrics @ chosen k with the type's QA score, across that type's
queries.
"""

from __future__ import annotations

import math


def _ranks(xs: list[float]) -> list[float]:
    """Average-rank ranking. Ties get the average of the ranks they cover.

    Stable enough for our small samples (~70-110 per qtype). No scipy dep.
    """
    n = len(xs)
    indexed = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[indexed[j + 1]] == xs[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-indexed average
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = 0.0
    sx = 0.0
    sy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        dy = y - my
        num += dx * dy
        sx += dx * dx
        sy += dy * dy
    denom = math.sqrt(sx * sy)
    if denom == 0:
        return 0.0
    return num / denom


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman's rho between two equal-length lists. Returns 0 on degenerate."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return _pearson(_ranks(xs), _ranks(ys))


def spearman_matrix(rows: list[list[float]], var_names: list[str]) -> dict:
    """Pairwise Spearman over columns of ``rows``.

    rows: N samples × D columns. Returns:
        { "variables": [...], "matrix": D×D, "n": N }
    """
    n = len(rows)
    d = len(var_names)
    if d == 0 or n < 2:
        return {
            "variables": var_names,
            "matrix": [[0.0] * d for _ in range(d)],
            "n": n,
        }
    # Transpose to columns
    cols: list[list[float]] = [[float(r[i]) for r in rows] for i in range(d)]
    matrix: list[list[float]] = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(d):
            if i == j:
                matrix[i][j] = 1.0
            elif j < i:
                matrix[i][j] = matrix[j][i]
            else:
                matrix[i][j] = round(spearman(cols[i], cols[j]), 4)
    return {"variables": var_names, "matrix": matrix, "n": n}
