"""GET /api/queries — list of all queries with metadata."""

from fastapi import APIRouter

from .. import deps
from ..schemas import QueryItem

router = APIRouter()


@router.get("/queries", response_model=list[QueryItem])
def list_queries() -> list[QueryItem]:
    queries = deps.get_queries()
    qrels = deps.get_qrels()
    return [
        QueryItem(
            id=q["_id"],
            text=q["text"],
            type=q.get("type"),
            has_qrels=q["_id"] in qrels,
        )
        for q in queries
    ]


@router.get("/queries/stats")
def queries_stats():
    """Per-set counts so both retrieval and generation pages can show the
    effective denominator for averaged metrics.

    ``evaluated`` = queries with BOTH a question ``type`` and qrels — the only
    set on which retrieval *and* QA metrics are both well-defined. Both pages
    average over this set so their numbers are directly comparable.
    """
    queries = deps.get_queries()
    qrels = deps.get_qrels()
    total = len(queries)
    with_type = sum(1 for q in queries if q.get("type"))
    with_qrels = sum(1 for q in queries if q["_id"] in qrels)
    evaluated_ids = [q["_id"] for q in queries if q.get("type") and q["_id"] in qrels]
    return {
        "total": total,
        "with_type": with_type,
        "with_qrels": with_qrels,
        "evaluated": len(evaluated_ids),
        "excluded_ids": [q["_id"] for q in queries if q["_id"] not in qrels],
    }


@router.get("/queries/{qid}")
def get_query(qid: str):
    for q in deps.get_queries():
        if q["_id"] == qid:
            return q
    return {"error": "not found"}
