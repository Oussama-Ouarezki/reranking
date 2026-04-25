"""POST /api/chat — retrieve, (optionally) rerank, (optionally) generate."""

import logging
import time

from fastapi import APIRouter

from .. import deps
from .. import config
from ..rerankers import registry as rerank_registry
from ..generation import rag
from ..evaluation.ranking import per_query_metrics
from ..schemas import ChatRequest, ChatResponse, RankingMetrics, RetrievedDoc, Timings

log = logging.getLogger(__name__)
router = APIRouter()

# Score thresholds below which we consider top doc not relevant.
# These differ across rerankers — BM25 raw scores are unbounded, monot5/duot5
# return probabilities, lit5 returns rank-based scores.
NO_RELEVANT_THRESHOLDS = {
    "bm25": 4.0,        # empirical: relevant docs typically score >5 in this corpus
    "monot5": 0.10,     # P(true) — relevant docs usually >0.5
    "duot5": 0.5,       # tournament agg, relevant doc near top accumulates >> 0.5
    "lit5": -1.0,       # LiT5 doesn't give meaningful scores; never trigger
    "mono_duo": -1.0,   # synthetic descending scores; never trigger
}


def _make_snippet(text: str, max_chars: int = 280) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    bm25 = deps.get_bm25()
    corpus = deps.get_corpus()
    qrels_all = deps.get_qrels()

    # Gold doc set for the active query (positive relevance only)
    gold_docs: set[str] = set()
    if req.query_id and req.query_id in qrels_all:
        gold_docs = {d for d, s in qrels_all[req.query_id].items() if s > 0}

    t_start = time.perf_counter()

    # 1. BM25 retrieve
    t0 = time.perf_counter()
    hits = bm25.search(req.message, k=config.BM25_RETRIEVE_K)
    t_retrieve = time.perf_counter() - t0

    # 2. Rerank
    t_rerank = 0.0
    if req.model != "bm25" and hits:
        t0 = time.perf_counter()
        reranker = rerank_registry.get(req.model)
        candidates: list[tuple[str, str]] = []
        for h in hits:
            text = corpus.get_text(h["docid"])
            if text:
                candidates.append((h["docid"], text))
        ranked = reranker.rerank(req.message, candidates)
        score_map = dict(ranked)
        order = [docid for docid, _ in ranked]
        hits = [
            {"docid": docid, "score": score_map[docid], "rank": i + 1}
            for i, docid in enumerate(order)
        ]
        t_rerank = time.perf_counter() - t0

    # 3. Truncate to top_k
    top_hits = hits[: req.top_k]

    # 4. Hydrate with corpus text
    retrieved: list[RetrievedDoc] = []
    docs_for_llm: list[dict] = []
    n_relevant_retrieved = 0
    for h in top_hits:
        doc = corpus.get(h["docid"]) or {"title": "", "text": "", "corpus_type": None}
        is_rel = h["docid"] in gold_docs
        if is_rel:
            n_relevant_retrieved += 1
        retrieved.append(RetrievedDoc(
            rank=h["rank"],
            docid=h["docid"],
            title=doc.get("title") or "",
            snippet=_make_snippet(doc.get("text") or ""),
            score=float(h["score"]),
            corpus_type=doc.get("corpus_type"),
            is_relevant=is_rel,
        ))
        docs_for_llm.append({
            "rank": h["rank"],
            "docid": h["docid"],
            "title": doc.get("title") or "",
            "text": doc.get("text") or "",
        })

    # 5. Determine question type
    qtype: str | None = None
    if req.query_id:
        for q in deps.get_queries():
            if q["_id"] == req.query_id:
                qtype = q.get("type")
                break
    if qtype is None and req.generate:
        qtype = rag.classify_question(req.message)

    # 6. No-relevant detection
    no_relevant = False
    if hits:
        thr = NO_RELEVANT_THRESHOLDS.get(req.model, -1.0)
        top_score = float(hits[0]["score"])
        if top_score < thr:
            no_relevant = True

    # 7. Generate
    answer = None
    t_generate = 0.0
    if req.generate:
        t0 = time.perf_counter()
        if no_relevant or not docs_for_llm:
            answer = "NO RELEVANT DOCUMENTS"
        else:
            try:
                history_dicts = [t.dict() for t in (req.history or [])]
                answer = rag.generate_answer(
                    req.message,
                    docs_for_llm,
                    qtype or "summary",
                    history=history_dicts,
                )
                # Trust the model when it says no relevant docs
                if answer.strip().upper().startswith("NO RELEVANT"):
                    no_relevant = True
            except Exception as exc:
                log.exception("LLM generation failed")
                answer = f"[generation error: {exc}]"
        t_generate = time.perf_counter() - t0

    # 8. Per-query ranking metrics if query has qrels
    metrics: RankingMetrics | None = None
    if req.query_id:
        qrels = deps.get_qrels()
        if req.query_id in qrels:
            full_ranked = [(h["docid"], float(h["score"])) for h in hits]
            m = per_query_metrics(req.query_id, full_ranked, qrels)
            metrics = RankingMetrics(**m)

    timings = Timings(
        retrieve_s=round(t_retrieve, 4),
        rerank_s=round(t_rerank, 4),
        generate_s=round(t_generate, 4),
        total_s=round(time.perf_counter() - t_start, 4),
    )

    return ChatResponse(
        answer=answer,
        retrieved=retrieved,
        question_type=qtype,
        top_k=req.top_k,
        model=req.model,
        no_relevant=no_relevant,
        metrics=metrics,
        timings=timings,
        n_relevant_retrieved=n_relevant_retrieved,
    )
