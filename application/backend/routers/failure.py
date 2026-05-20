"""Failure analysis between two cached generation runs.

For each query present in both runs we compute the per-query QA delta and
flag the query as a 'failure case' when the drop exceeds the qtype-specific
threshold. Direction is bidirectional: a failure can be assigned to A or to
B, depending on which side scored lower.

No re-generation happens here — we only read the cached run files.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .generation import _load_gen_run

router = APIRouter()


# Thresholds match the spec: a "failure" is recorded when |delta| exceeds
# the qtype's threshold. yesno uses >0 (any flip is a failure).
QTYPE_THRESHOLDS = {
    "yesno": 0.0,      # binary flip
    "factoid": 0.1,    # MRR
    "list": 0.1,       # F1
    "summary": 0.5,    # judge / 0–1 score
}
QTYPE_METRIC_LABEL = {
    "factoid": "MRR",
    "yesno": "Accuracy",
    "list": "F1",
    "summary": "Judge",
}


class FailureRequest(BaseModel):
    run_a: str  # baseline
    run_b: str  # comparison


class FailureRecord(BaseModel):
    qid: str
    qtype: str
    question: str
    score_a: float
    score_b: float
    delta: float  # b - a (positive = B is better, negative = B is worse)
    failed_model: str  # "a" or "b" — whichever scored lower
    threshold: float
    answer_a: Optional[str] = None
    answer_b: Optional[str] = None
    # Retrieval metrics @ k for each side, plus delta = b - a. Each is a dict
    # with keys ndcg / p / r / mrr / map (or None if qrels weren't available).
    # This lets the UI tell whether the QA failure tracks a retrieval gap.
    retrieval_a: Optional[dict[str, float]] = None
    retrieval_b: Optional[dict[str, float]] = None
    retrieval_delta: Optional[dict[str, float]] = None


class FailureBucket(BaseModel):
    qtype: str
    metric_label: str
    threshold: float
    n_pairs: int  # queries of this qtype present in both runs
    n_failures: int  # total failures (a-failed + b-failed)
    n_a_failed: int  # cases where A scored worse by > threshold
    n_b_failed: int  # cases where B scored worse by > threshold
    failures: list[FailureRecord]


class FailureResponse(BaseModel):
    run_a: str
    run_b: str
    retrieval_model_a: Optional[str] = None
    retrieval_model_b: Optional[str] = None
    k_a: Optional[int] = None
    k_b: Optional[int] = None
    n_overlapping: int
    total_failures: int
    total_a_failed: int
    total_b_failed: int
    by_qtype: list[FailureBucket]


@router.post("/failure/compare", response_model=FailureResponse)
def compare(req: FailureRequest) -> FailureResponse:
    run_a = _load_gen_run(req.run_a)
    if run_a is None:
        raise HTTPException(404, f"gen run not found: {req.run_a}")
    run_b = _load_gen_run(req.run_b)
    if run_b is None:
        raise HTTPException(404, f"gen run not found: {req.run_b}")

    pq_a = run_a.get("per_query") or {}
    pq_b = run_b.get("per_query") or {}

    buckets: dict[str, FailureBucket] = {
        qt: FailureBucket(
            qtype=qt,
            metric_label=QTYPE_METRIC_LABEL[qt],
            threshold=QTYPE_THRESHOLDS[qt],
            n_pairs=0,
            n_failures=0,
            n_a_failed=0,
            n_b_failed=0,
            failures=[],
        )
        for qt in QTYPE_THRESHOLDS
    }

    n_overlap = 0
    for qid, ea in pq_a.items():
        eb = pq_b.get(qid)
        if not eb:
            continue
        n_overlap += 1
        sa = ea.get("qa_score")
        sb = eb.get("qa_score")
        if sa is None or sb is None:
            continue
        qt = ea.get("qtype") or eb.get("qtype")
        if qt not in buckets:
            continue
        bucket = buckets[qt]
        bucket.n_pairs += 1
        delta = float(sb) - float(sa)
        threshold = QTYPE_THRESHOLDS[qt]
        # yesno uses strict >0 (any flip); other types use > threshold magnitude.
        is_failure = abs(delta) > threshold if qt != "yesno" else delta != 0.0
        if not is_failure:
            continue
        failed = "b" if delta < 0 else "a"
        if failed == "a":
            bucket.n_a_failed += 1
        else:
            bucket.n_b_failed += 1
        bucket.n_failures += 1

        rm_a = ea.get("retrieval_metrics")
        rm_b = eb.get("retrieval_metrics")
        retr_a = (
            {k: round(float(v), 4) for k, v in rm_a.items() if v is not None}
            if isinstance(rm_a, dict)
            else None
        )
        retr_b = (
            {k: round(float(v), 4) for k, v in rm_b.items() if v is not None}
            if isinstance(rm_b, dict)
            else None
        )
        retr_delta: Optional[dict[str, float]] = None
        if retr_a is not None and retr_b is not None:
            retr_delta = {
                key: round(retr_b[key] - retr_a[key], 4)
                for key in retr_a
                if key in retr_b
            }

        bucket.failures.append(
            FailureRecord(
                qid=qid,
                qtype=qt,
                question=ea.get("question") or eb.get("question") or "",
                score_a=round(float(sa), 4),
                score_b=round(float(sb), 4),
                delta=round(delta, 4),
                failed_model=failed,
                threshold=threshold,
                answer_a=ea.get("answer"),
                answer_b=eb.get("answer"),
                retrieval_a=retr_a,
                retrieval_b=retr_b,
                retrieval_delta=retr_delta,
            )
        )

    # Sort failures within each bucket by |delta| desc so the worst cases
    # surface first.
    for b in buckets.values():
        b.failures.sort(key=lambda r: abs(r.delta), reverse=True)

    total_failures = sum(b.n_failures for b in buckets.values())
    total_a_failed = sum(b.n_a_failed for b in buckets.values())
    total_b_failed = sum(b.n_b_failed for b in buckets.values())

    return FailureResponse(
        run_a=req.run_a,
        run_b=req.run_b,
        retrieval_model_a=run_a.get("retrieval_model"),
        retrieval_model_b=run_b.get("retrieval_model"),
        k_a=run_a.get("k"),
        k_b=run_b.get("k"),
        n_overlapping=n_overlap,
        total_failures=total_failures,
        total_a_failed=total_a_failed,
        total_b_failed=total_b_failed,
        by_qtype=[buckets[qt] for qt in ("yesno", "factoid", "list", "summary")],
    )
