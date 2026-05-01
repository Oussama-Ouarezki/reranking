"""Reranker registry — lazy-load and cache instances."""

from threading import RLock

from .. import config
from .base import Reranker


_INSTANCES: dict[str, Reranker] = {}
# RLock so cascades (e.g. mono_duo) can call get() from inside their __init__
# while the outer get() still holds the lock.
_LOCK = RLock()


# Models that the in-app eval WS can run live (BM25 + native rerankers).
# DeepSeek is produced by an offline script and shows up via the runs cache,
# so it is NOT in this list.
_EVAL_MODELS = (
    "bm25",
    "monot5",
    "monot5_bioasq",
    "duot5",
    "duot5_bioasq",
    "lit5",
    "mono_duo",
    "monot5_lit5",
    "mono_uncertain_duo_lit5",
    "mono_dynamic_duo_lit5",
    "mono_gated_duo",
    "mono_proximity_duo",
    "mono_proximity_duo_lit5",
    "lit5_duo",
    "mono_proximity_duo_0005",
    "mono_proximity_duo_005_top30",
    "mono_mau_duo_low_cost",
    "mono_mau_duo_pareto",
    "mono_gated_lit5_top20",
    "mono_gated_lit5_top40",
    "mono_gated_lit5_top50",
    "bge_v2_m3",
    "qwen3_reranker_4b",
    "rank_zephyr",
)


def get(name: str) -> Reranker:
    """Return a cached reranker instance, loading it on first call."""
    if name == "bm25":
        raise ValueError("BM25 is not a reranker; skip rerank when model == 'bm25'")

    with _LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]

        if name == "monot5":
            from .monot5 import MonoT5Reranker
            inst = MonoT5Reranker(batch_size=32)
        elif name == "monot5_bioasq":
            from .monot5 import MonoT5Reranker
            inst = MonoT5Reranker(checkpoint=config.CHECKPOINTS["monot5_bioasq"], batch_size=32)
        elif name == "duot5":
            from .duot5 import DuoT5Reranker
            inst = DuoT5Reranker(batch_size=16)
        elif name == "duot5_bioasq":
            from .duot5 import DuoT5Reranker
            inst = DuoT5Reranker(checkpoint=config.CHECKPOINTS["duot5_bioasq"], batch_size=16)
        elif name == "lit5":
            from .lit5 import LiT5Reranker
            inst = LiT5Reranker()
        elif name == "mono_duo":
            from .cascade import MonoDuoCascade
            inst = MonoDuoCascade()
        elif name == "monot5_lit5":
            from .cascade import MonoThreshLiT5Cascade
            inst = MonoThreshLiT5Cascade()
        elif name == "mono_uncertain_duo_lit5":
            from .cascade import MonoUncertainDuoLiT5Cascade
            inst = MonoUncertainDuoLiT5Cascade()
        elif name == "mono_dynamic_duo_lit5":
            from .cascade import MonoDynamicDuoLiT5Cascade
            inst = MonoDynamicDuoLiT5Cascade()
        elif name == "mono_gated_duo":
            from .cascade import MonoGatedDuoCascade
            inst = MonoGatedDuoCascade()
        elif name == "mono_proximity_duo":
            from .cascade import MonoProximityDuoCascade
            inst = MonoProximityDuoCascade()
        elif name == "mono_proximity_duo_lit5":
            from .cascade import MonoProximityDuoLiT5Cascade
            inst = MonoProximityDuoLiT5Cascade()
        elif name == "lit5_duo":
            from .cascade import LiT5DuoCascade
            inst = LiT5DuoCascade()
        elif name == "mono_proximity_duo_0005":
            from .cascade import MonoProximityDuoCascade0005
            inst = MonoProximityDuoCascade0005()
        elif name == "mono_proximity_duo_005_top30":
            from .cascade import MonoProximityDuoCascade005Top30
            inst = MonoProximityDuoCascade005Top30()
        elif name == "mono_mau_duo_low_cost":
            from .cascade import MonoMauDuoLowCost
            inst = MonoMauDuoLowCost()
        elif name == "mono_mau_duo_pareto":
            from .cascade import MonoMauDuoPareto
            inst = MonoMauDuoPareto()
        elif name == "mono_gated_lit5_top20":
            from .cascade import MonoGatedLiT5Top20
            inst = MonoGatedLiT5Top20()
        elif name == "mono_gated_lit5_top40":
            from .cascade import MonoGatedLiT5Top40
            inst = MonoGatedLiT5Top40()
        elif name == "mono_gated_lit5_top50":
            from .cascade import MonoGatedLiT5Top50
            inst = MonoGatedLiT5Top50()
        elif name == "bge_v2_m3":
            from .bge import BGERerankerV2M3
            inst = BGERerankerV2M3(batch_size=32)
        elif name == "qwen3_reranker_4b":
            from .qwen3 import Qwen3Reranker4B
            inst = Qwen3Reranker4B()
        elif name == "rank_zephyr":
            from .rank_zephyr import RankZephyrReranker
            inst = RankZephyrReranker()
        else:
            raise ValueError(f"Unknown reranker: {name}")

        _INSTANCES[name] = inst
        return inst


def eval_models() -> tuple[str, ...]:
    """Models the eval WebSocket will accept (BM25 + every reranker)."""
    return _EVAL_MODELS


def available() -> list[str]:
    return list(_EVAL_MODELS)
