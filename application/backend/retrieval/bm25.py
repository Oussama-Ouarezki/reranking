"""Pyserini Lucene BM25 wrapper.

JAVA_HOME MUST be set before importing pyserini, so we set it from config first.
"""

import os
from .. import config

os.environ["JAVA_HOME"] = config.JAVA_HOME
os.environ["PATH"] = f"{config.JAVA_HOME}/bin:" + os.environ.get("PATH", "")

from pyserini.search.lucene import LuceneSearcher  # noqa: E402


class BM25Retriever:
    def __init__(self, index_path=None, k1=config.BM25_K1, b=config.BM25_B):
        index_path = str(index_path or config.LUCENE_INDEX)
        self.searcher = LuceneSearcher(index_path)
        self.searcher.set_bm25(k1=k1, b=b)
        self.k1 = k1
        self.b = b

    def search(self, query: str, k: int = 100):
        """Return list of {docid, score, rank} dicts."""
        hits = self.searcher.search(query, k=k)
        return [
            {"docid": h.docid, "score": float(h.score), "rank": i + 1}
            for i, h in enumerate(hits)
        ]
