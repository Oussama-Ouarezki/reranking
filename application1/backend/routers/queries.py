"""GET /api/queries — list of all 340 queries with metadata."""

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


@router.get("/queries/{qid}")
def get_query(qid: str):
    for q in deps.get_queries():
        if q["_id"] == qid:
            return q
    return {"error": "not found"}
