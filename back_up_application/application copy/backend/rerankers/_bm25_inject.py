"""BM25-score injection helpers.

Three formats can be prepended to each candidate's text before reranking, so
the reranker model sees BM25's prior. Toggle from the eval config UI; backend
wiring is one commentable line per reranker.

Modes:
  "raw"    → "[BM25=12.34] {text}"     two-decimal score (precision 1e-2)
  "norm"   → "[BM25=0.87] {text}"      min-max scaled into [0, 1] per query
  "bucket" → "[BM25=high] {text}"      tertile rank (low / medium / high) per query
  None / "off" / unknown → no injection
"""

from collections.abc import Iterable

Mode = str  # "raw" | "norm" | "bucket" | "off" | None


def _format_raw(score: float) -> str:
    return f"{score:.2f}"


def _format_norm(score: float, lo: float, hi: float) -> str:
    if hi <= lo:
        return "0.00"
    return f"{(score - lo) / (hi - lo):.2f}"


def _format_bucket(score: float, t1: float, t2: float) -> str:
    if score >= t2:
        return "high"
    if score >= t1:
        return "medium"
    return "low"


def inject(
    candidates: list[tuple[str, str]],
    bm25_scores: dict[str, float] | None,
    mode: Mode | None,
) -> list[tuple[str, str]]:
    """Return a new candidates list with BM25 prior prepended to each text.

    No-op when `bm25_scores` is None/empty or `mode` is falsy/"off"/unknown.
    """
    if not bm25_scores or not mode or mode == "off":
        return candidates

    scores = [bm25_scores.get(d, 0.0) for d, _ in candidates]
    if not scores:
        return candidates

    if mode == "raw":
        tagged = [_format_raw(s) for s in scores]
    elif mode == "norm":
        lo, hi = min(scores), max(scores)
        tagged = [_format_norm(s, lo, hi) for s in scores]
    elif mode == "bucket":
        srt = sorted(scores)
        n = len(srt)
        t1 = srt[n // 3] if n >= 3 else srt[0]
        t2 = srt[(2 * n) // 3] if n >= 3 else srt[-1]
        tagged = [_format_bucket(s, t1, t2) for s in scores]
    else:
        return candidates

    return [(d, f"[BM25={tag}] {t}") for (d, t), tag in zip(candidates, tagged)]


VALID_MODES: Iterable[str] = ("off", "raw", "norm", "bucket")
