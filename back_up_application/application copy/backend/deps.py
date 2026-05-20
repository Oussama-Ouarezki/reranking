"""Singleton dependency providers loaded once and reused."""

from functools import lru_cache
import json

from . import config
from .retrieval.bm25 import BM25Retriever
from .retrieval.corpus import Corpus


@lru_cache(maxsize=1)
def get_bm25() -> BM25Retriever:
    return BM25Retriever()


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    c = Corpus()
    c.load()
    return c


@lru_cache(maxsize=1)
def get_queries() -> list[dict]:
    """Load 340 queries from queries_full.jsonl."""
    out = []
    with open(config.QUERIES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


@lru_cache(maxsize=1)
def get_qrels() -> dict[str, dict[str, int]]:
    """Load qrels.tsv -> {qid: {docid: relevance}}."""
    qrels: dict[str, dict[str, int]] = {}
    with open(config.QRELS_PATH, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header line: "query-id\tcorpus-id\tscore"
        if not header.startswith("query-id"):
            f.seek(0)  # no header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, docid, score = parts[0], parts[1], int(parts[2])
            qrels.setdefault(qid, {})[docid] = score
    return qrels
