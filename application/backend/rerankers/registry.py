"""Reranker registry — lazy-load and cache instances."""

from threading import RLock

from .base import Reranker


_INSTANCES: dict[str, Reranker] = {}
# RLock so cascades (e.g. mono_duo) can call get() from inside their __init__
# while the outer get() still holds the lock.
_LOCK = RLock()


def get(name: str) -> Reranker:
    """Return a cached reranker instance, loading it on first call."""
    if name == "bm25":
        raise ValueError("BM25 is not a reranker; skip rerank when model == 'bm25'")

    with _LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]

        if name == "monot5":
            from .monot5 import MonoT5Reranker
            inst = MonoT5Reranker()
        elif name == "duot5":
            from .duot5 import DuoT5Reranker
            inst = DuoT5Reranker()
        elif name == "lit5":
            from .lit5 import LiT5Reranker
            inst = LiT5Reranker()
        elif name == "mono_duo":
            from .cascade import MonoDuoCascade
            inst = MonoDuoCascade()
        else:
            raise ValueError(f"Unknown reranker: {name}")

        _INSTANCES[name] = inst
        return inst


def available() -> list[str]:
    return ["bm25", "monot5", "duot5", "lit5", "mono_duo"]
