"""Reranker protocol.

A reranker takes a query and a list of candidate docs (with text) and
returns them reordered by score (descending).
"""

from typing import Protocol


class Reranker(Protocol):
    name: str

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        """
        candidates: list of (docid, doc_text)
        returns:    list of (docid, score) sorted by score desc
        """
        ...
