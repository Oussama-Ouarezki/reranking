"""Cascade reranker: BM25 → monoT5 → duoT5.

Standard IR cascade pattern (cascadeT5 / monoduoT5):

1. BM25 retrieves the top 100 candidates (done upstream).
2. monoT5 scores all 100 pointwise → take its top 20 by P(true).
3. duoT5 runs the pairwise tournament on those 20.
4. The remaining 80 (mono-rank 21..100) are pinned below the duoT5-ranked head
   so we still return a full ranking for evaluation purposes.

Final scores are synthetic descending integers so the global ordering is
preserved for downstream metrics (nDCG, MAP, MRR, etc.).
"""

from .duot5 import TOURNAMENT_TOP_N
from . import registry


class MonoDuoCascade:
    name = "mono_duo"

    def __init__(self):
        # Reuse cached monot5 + duot5 instances if already loaded; otherwise
        # this triggers their loads via the registry (which uses an RLock so
        # nested get() from inside __init__ does not deadlock).
        self.mono = registry.get("monot5")
        self.duo = registry.get("duot5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        mono_ranked = self.mono.rerank(query, candidates)  # all 100, sorted by P(true)
        cand_map = dict(candidates)

        head = mono_ranked[:TOURNAMENT_TOP_N]   # top 20 from monoT5
        tail = mono_ranked[TOURNAMENT_TOP_N:]   # rest, in monoT5 order

        head_pairs = [(d, cand_map[d]) for d, _ in head if d in cand_map]
        duo_ranked = self.duo.rerank(query, head_pairs)  # tournament on 20

        merged = list(duo_ranked) + tail
        n = len(merged)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(merged)]
