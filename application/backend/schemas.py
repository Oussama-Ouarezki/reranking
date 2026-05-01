"""Pydantic request/response schemas."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


ModelName = Literal["bm25", "monot5", "duot5", "lit5", "mono_duo", "monot5_lit5", "mono_uncertain_duo_lit5", "mono_dynamic_duo_lit5", "mono_gated_duo", "mono_proximity_duo", "mono_proximity_duo_lit5", "lit5_duo", "mono_proximity_duo_0005", "mono_proximity_duo_005_top30", "mono_mau_duo_low_cost", "mono_mau_duo_pareto", "mono_gated_lit5_top20", "mono_gated_lit5_top40", "mono_gated_lit5_top50", "bge_v2_m3", "qwen3_reranker_4b", "rank_zephyr"]
QuestionType = Literal["factoid", "list", "yesno", "summary"]


class QueryItem(BaseModel):
    id: str
    text: str
    type: Optional[QuestionType] = None
    has_qrels: bool = False


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    model: ModelName = "bm25"
    top_k: int = Field(default=10, ge=1, le=50)
    history: list[ChatTurn] = []
    query_id: Optional[str] = None  # if from sidebar, lets us bypass type detection
    generate: bool = True  # set False to skip LLM call


class RetrievedDoc(BaseModel):
    rank: int
    docid: str
    title: str
    snippet: str
    score: float
    corpus_type: Optional[str] = None
    is_relevant: bool = False  # docid appears in qrels with positive relevance


class Timings(BaseModel):
    retrieve_s: float
    rerank_s: float
    generate_s: float
    total_s: float


class RankingMetrics(BaseModel):
    """Per-query ranking metrics computed against qrels."""
    ndcg_at: dict[int, float]
    mrr_at: dict[int, float]
    p_at: dict[int, float]
    r_at: dict[int, float]
    map_at: dict[int, float] = {}


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    retrieved: list[RetrievedDoc]
    question_type: Optional[QuestionType] = None
    top_k: int
    model: ModelName
    no_relevant: bool = False
    metrics: Optional[RankingMetrics] = None  # only when query_id has qrels
    timings: Optional[Timings] = None
    n_relevant_retrieved: int = 0  # gold docs found within top_k


class EvalRequest(BaseModel):
    models: list[ModelName]
    top_k: int = 10
    generate: bool = False


class EvalProgress(BaseModel):
    type: Literal["progress", "query_done", "model_done", "done", "error"]
    current: Optional[int] = None
    total: Optional[int] = None
    model: Optional[ModelName] = None
    qid: Optional[str] = None
    metrics: Optional[dict] = None
    aggregate: Optional[dict] = None
    results: Optional[dict] = None
    message: Optional[str] = None
