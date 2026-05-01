"""Cascade rerankers built on top of the individual rerankers.

MonoDuoCascade           — BM25 → monoT5 (top-N) → duoT5 tournament
MonoThreshLiT5           — BM25 → monoT5 (threshold filter) → LiT5 listwise
MonoUncertainDuoLiT5     — BM25 → monoT5 (rank all) → duoT5 on fixed
                            uncertain zone (positions 15-25) → LiT5 top-20
MonoDynamicDuoLiT5       — BM25 → monoT5 (rank all) → duoT5 on docs whose
                            adjacent score gap < margin → LiT5 top-20
MonoGatedDuoCascade      — BM25 → monoT5 (rank all) → duoT5 tournament on
                            top-20 only when top-1/top-2 score gap < τ=0.001

Final scores are synthetic descending integers so the global ordering is
preserved for downstream metrics (nDCG, MAP, MRR, etc.).
"""

import logging

from .duot5 import TOURNAMENT_TOP_N
from . import registry

_log = logging.getLogger(__name__)

# P(true) threshold: docs scoring below this are skipped by LiT5 and
# kept in their monoT5 order below the LiT5-ranked head.
MONO_THRESHOLD = 0.5


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

        mono_ranked = self.mono.rerank(query, candidates)
        cand_map = dict(candidates)

        head = mono_ranked[:TOURNAMENT_TOP_N]
        tail = mono_ranked[TOURNAMENT_TOP_N:]

        head_pairs = [(d, cand_map[d]) for d, _ in head if d in cand_map]
        duo_ranked = self.duo.rerank(query, head_pairs)  # tournament on 20

        merged = list(duo_ranked) + tail
        n = len(merged)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(merged)]


class MonoThreshLiT5Cascade:
    """BM25 → monoT5 (threshold=0.7) → LiT5 listwise on survivors."""

    name = "monot5_lit5"

    def __init__(self):
        self.mono = registry.get("monot5")
        self.lit5 = registry.get("lit5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        cand_map = dict(candidates)

        # Step 1: monoT5 scores all candidates → sorted (docid, P(true))
        mono_ranked = self.mono.rerank(query, candidates)

        # Step 2: split by threshold
        head = [(d, s) for d, s in mono_ranked if s >= MONO_THRESHOLD]
        tail = [(d, s) for d, s in mono_ranked if s <  MONO_THRESHOLD]

        # Edge case: nothing above threshold → return mono order as-is
        if not head:
            n = len(mono_ranked)
            return [(docid, float(n - i)) for i, (docid, _) in enumerate(mono_ranked)]

        # Step 3: LiT5 listwise reranks the survivors (needs (docid, text))
        head_pairs = [(d, cand_map[d]) for d, _ in head if d in cand_map]
        lit5_ranked = self.lit5.rerank(query, head_pairs)

        # Step 4: merge — LiT5 head then mono-ordered tail
        merged = list(lit5_ranked) + tail
        n = len(merged)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(merged)]


# Positions (0-indexed) of the "uncertain" zone that duoT5 will reorder.
# Positions 0-13 = monoT5 is confident (top); 14-24 = uncertain middle;
# 25+ = likely irrelevant. duoT5 only runs on 11 docs, then LiT5 on top-20.
UNCERTAIN_START = 14   # inclusive
UNCERTAIN_END   = 25   # exclusive  → indices 14…24 (positions 15–25)
LIT5_INPUT_K    = 20   # how many docs (after duoT5 reorder) go to LiT5


class MonoUncertainDuoLiT5Cascade:
    """BM25 → monoT5 (all 50) → duoT5 on uncertain zone (pos 15-25) → LiT5 top-20."""

    name = "mono_uncertain_duo_lit5"

    def __init__(self):
        self.mono = registry.get("monot5")
        self.duo  = registry.get("duot5")
        self.lit5 = registry.get("lit5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        cand_map = dict(candidates)

        # Step 1: monoT5 scores and ranks all candidates
        mono_ranked = self.mono.rerank(query, candidates)

        # Step 2: split into confident head / uncertain zone / tail
        head      = mono_ranked[:UNCERTAIN_START]
        uncertain = mono_ranked[UNCERTAIN_START:UNCERTAIN_END]
        tail      = mono_ranked[UNCERTAIN_END:]

        # Step 3: duoT5 reorders only the uncertain zone
        if uncertain:
            uncertain_pairs = [(d, cand_map[d]) for d, _ in uncertain if d in cand_map]
            duo_reranked = self.duo.rerank(query, uncertain_pairs)
        else:
            duo_reranked = uncertain

        # Step 4: merge back into a single ranked list
        merged = list(head) + list(duo_reranked) + list(tail)

        # Step 5: take top-20 and pass to LiT5 for final listwise rerank
        top_k      = merged[:LIT5_INPUT_K]
        top_k_rest = merged[LIT5_INPUT_K:]

        top_k_pairs = [(d, cand_map[d]) for d, _ in top_k if d in cand_map]
        lit5_ranked = self.lit5.rerank(query, top_k_pairs)

        # Step 6: append remaining docs (positions 21+) in merged order below
        final = list(lit5_ranked) + top_k_rest
        n = len(final)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(final)]


# Score gap below which two adjacent monoT5 docs are considered "uncertain".
# monoT5 P(true) is in [0, 1]. A gap of 0.05 means the model assigns nearly
# identical relevance probability to two docs — a coin-flip duoT5 should break.
DYNAMIC_MARGIN   = 0.05
DYNAMIC_SCAN_TOP = 30    # only scan for uncertain pairs within the top-N docs
DYNAMIC_LIT5_K   = 20


class MonoDynamicDuoLiT5Cascade:
    """BM25 → monoT5 → duoT5 on dynamically identified uncertain pairs → LiT5 top-20.

    Uncertainty is defined per-query: any two adjacent docs in the monoT5
    ranking whose P(true) scores differ by less than DYNAMIC_MARGIN are both
    flagged as uncertain.  duoT5 then directly re-compares only those docs,
    and the result is merged back into their original position slots before
    LiT5 does a final listwise pass on the top-20.
    """

    name = "mono_dynamic_duo_lit5"

    def __init__(self, margin: float = DYNAMIC_MARGIN):
        self.margin = margin
        self.mono = registry.get("monot5")
        self.duo  = registry.get("duot5")
        self.lit5 = registry.get("lit5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        cand_map = dict(candidates)

        # Step 1: monoT5 scores and ranks all candidates
        mono_ranked = self.mono.rerank(query, candidates)  # [(docid, P(true))], desc

        # Step 2: flag uncertain indices — scan only within top DYNAMIC_SCAN_TOP docs
        scan_limit = min(DYNAMIC_SCAN_TOP, len(mono_ranked))
        gaps = []
        uncertain_idx: set[int] = set()
        for i in range(scan_limit - 1):
            gap = mono_ranked[i][1] - mono_ranked[i + 1][1]  # always >= 0 (sorted desc)
            gaps.append(gap)
            if gap < self.margin:
                uncertain_idx.add(i)
                uncertain_idx.add(i + 1)

        if gaps:
            min_gap = min(gaps)
            avg_gap = sum(gaps) / len(gaps)
            _log.debug(
                "margin=%.3f  flagged=%d/%d docs  "
                "gap: min=%.4f avg=%.4f  query=%r",
                self.margin, len(uncertain_idx), len(mono_ranked),
                min_gap, avg_gap, query[:60],
            )

        # Step 3: if no uncertain pairs, skip duoT5 entirely
        if not uncertain_idx:
            top_k      = mono_ranked[:DYNAMIC_LIT5_K]
            top_k_rest = mono_ranked[DYNAMIC_LIT5_K:]
            top_k_pairs = [(d, cand_map[d]) for d, _ in top_k if d in cand_map]
            lit5_ranked = self.lit5.rerank(query, top_k_pairs)
            final = list(lit5_ranked) + top_k_rest
            n = len(final)
            return [(docid, float(n - i)) for i, (docid, _) in enumerate(final)]

        # Step 4: duoT5 reorders only the uncertain docs
        uncertain_docs = [mono_ranked[i] for i in sorted(uncertain_idx)]
        uncertain_pairs = [(d, cand_map[d]) for d, _ in uncertain_docs if d in cand_map]
        duo_reranked = self.duo.rerank(query, uncertain_pairs)  # [(docid, score)], desc

        # Step 5: rebuild full ranking — certain docs stay in place,
        # uncertain slots are filled in duoT5 order (left to right)
        duo_iter = iter(duo_reranked)
        merged = []
        for i, item in enumerate(mono_ranked):
            if i in uncertain_idx:
                merged.append(next(duo_iter, item))  # fallback to mono if iter exhausted
            else:
                merged.append(item)

        # Step 6: LiT5 on top-20 of the merged list
        top_k      = merged[:DYNAMIC_LIT5_K]
        top_k_rest = merged[DYNAMIC_LIT5_K:]
        top_k_pairs = [(d, cand_map[d]) for d, _ in top_k if d in cand_map]
        lit5_ranked = self.lit5.rerank(query, top_k_pairs)

        final = list(lit5_ranked) + top_k_rest
        n = len(final)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(final)]


# Score gap below which duoT5 is triggered (top-1 vs top-2 P(true)).
# Chosen via Pareto knee-point sweep (see models/monot5/threshold_sweep.py):
# τ=0.001 sends ~50% of queries to duoT5 while recovering most of the nDCG gain.
GATED_MARGIN = 0.001
GATED_TOP_N  = 20   # duoT5 tournament on top-N docs when triggered


class MonoGatedDuoCascade:
    """BM25 → monoT5 (rank all) → duoT5 on top-20 only if gap(top1, top2) < 0.001.

    Per-query gating: if the monoT5 P(true) score gap between rank-1 and rank-2
    is below GATED_MARGIN the model is uncertain about the top pair, so duoT5
    runs a full tournament on the top-20 docs.  Otherwise monoT5 ranking is kept
    as-is, saving all duoT5 computation for that query.

    Threshold was selected by sweeping τ and picking the Pareto knee
    (max nDCG gain per unit cost on the BioASQ dev set).
    """

    name = "mono_gated_duo"

    def __init__(self, margin: float = GATED_MARGIN):
        self.margin = margin
        self.mono = registry.get("monot5")
        self.duo  = registry.get("duot5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        cand_map = dict(candidates)

        # Step 1: monoT5 scores and ranks all candidates
        mono_ranked = self.mono.rerank(query, candidates)  # [(docid, P(true))], desc

        # Step 2: compute top-1 vs top-2 gap
        gap = (mono_ranked[0][1] - mono_ranked[1][1]) if len(mono_ranked) >= 2 else 1.0

        _log.debug(
            "gap=%.4f  margin=%.4f  trigger=%s  query=%r",
            gap, self.margin, gap < self.margin, query[:60],
        )

        # Step 3: gate — only apply duoT5 when monoT5 is uncertain at the top
        if gap < self.margin:
            head       = mono_ranked[:GATED_TOP_N]
            tail       = mono_ranked[GATED_TOP_N:]
            head_pairs = [(d, cand_map[d]) for d, _ in head if d in cand_map]
            duo_ranked = self.duo.rerank(query, head_pairs)
            merged     = list(duo_ranked) + tail
        else:
            merged = mono_ranked

        n = len(merged)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(merged)]


# P(true) gap below which two docs in the top-N are considered "too close to
# trust monoT5 to separate".  All-pairs check: every pair (i, j) in top-20
# with |s_i − s_j| < margin causes both to be sent to duoT5.
PROXIMITY_MARGIN = 0.001
PROXIMITY_TOP_N  = 20


class MonoProximityDuoCascade:
    """BM25 → monoT5 → duoT5 on all docs within 0.001 P(true) of each other.

    Within the top-20 monoT5 docs every pair (i, j) is compared.  If
    |s_i − s_j| < PROXIMITY_MARGIN both docs are flagged as uncertain.
    All flagged docs are sent to duoT5 together; the result is merged back
    slot-preserving (the k-th uncertain slot receives the k-th duoT5-ranked
    doc).  Non-flagged docs stay exactly where monoT5 placed them.

    Difference from MonoDynamicDuoLiT5Cascade:
      • All-pairs check (not adjacent-only) — catches any two close docs
        regardless of whether a third doc separates them in the ranking.
      • No LiT5 final pass.
      • Tighter margin (0.001 vs 0.05), scans only top-20.
    """

    name = "mono_proximity_duo"

    def __init__(self, margin: float = PROXIMITY_MARGIN, top_n: int = PROXIMITY_TOP_N):
        self.margin = margin
        self.top_n  = top_n
        self.mono = registry.get("monot5")
        self.duo  = registry.get("duot5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        cand_map = dict(candidates)

        # Step 1: monoT5 scores and ranks all candidates
        mono_ranked = self.mono.rerank(query, candidates)  # [(docid, P(true))], desc

        # Step 2: split into top-N and tail
        top_n = mono_ranked[:self.top_n]
        tail  = mono_ranked[self.top_n:]

        # Step 3: all-pairs proximity check within top-N — O(n²) = 190 checks
        uncertain_idx: set[int] = set()
        for i in range(len(top_n)):
            for j in range(i + 1, len(top_n)):
                if abs(top_n[i][1] - top_n[j][1]) < self.margin:
                    uncertain_idx.add(i)
                    uncertain_idx.add(j)

        _log.debug(
            "margin=%.4f  flagged=%d/%d docs  query=%r",
            self.margin, len(uncertain_idx), len(top_n), query[:60],
        )

        # Step 4: if nothing is uncertain, skip duoT5 entirely
        if not uncertain_idx:
            final = list(top_n) + list(tail)
            n = len(final)
            return [(docid, float(n - i)) for i, (docid, _) in enumerate(final)]

        # Step 5: duoT5 reorders only the uncertain docs
        uncertain_docs  = [top_n[i] for i in sorted(uncertain_idx)]
        uncertain_pairs = [(d, cand_map[d]) for d, _ in uncertain_docs if d in cand_map]
        duo_reranked    = self.duo.rerank(query, uncertain_pairs)

        # Step 6: slot-preserving merge — uncertain slots filled in duoT5 order
        duo_iter   = iter(duo_reranked)
        merged_top = [
            next(duo_iter, item) if i in uncertain_idx else item
            for i, item in enumerate(top_n)
        ]

        final = merged_top + list(tail)
        n = len(final)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(final)]


class MonoProximityDuoLiT5Cascade:
    """BM25 → monoT5 → duoT5 (proximity 0.001) → LiT5 top-20.

    Identical to MonoProximityDuoCascade but adds a LiT5 listwise pass on
    the top-20 docs after the slot-preserving duoT5 merge.
    """

    name = "mono_proximity_duo_lit5"

    def __init__(self, margin: float = PROXIMITY_MARGIN):
        self.margin = margin
        self.mono = registry.get("monot5")
        self.duo  = registry.get("duot5")
        self.lit5 = registry.get("lit5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        cand_map = dict(candidates)

        mono_ranked = self.mono.rerank(query, candidates)
        top_n = mono_ranked[:PROXIMITY_TOP_N]
        tail  = mono_ranked[PROXIMITY_TOP_N:]

        uncertain_idx: set[int] = set()
        for i in range(len(top_n)):
            for j in range(i + 1, len(top_n)):
                if abs(top_n[i][1] - top_n[j][1]) < self.margin:
                    uncertain_idx.add(i)
                    uncertain_idx.add(j)

        if uncertain_idx:
            uncertain_pairs = [
                (d, cand_map[d]) for d in
                [top_n[i][0] for i in sorted(uncertain_idx)]
                if d in cand_map
            ]
            duo_reranked = self.duo.rerank(query, uncertain_pairs)
            duo_iter     = iter(duo_reranked)
            top_n = [
                next(duo_iter, item) if i in uncertain_idx else item
                for i, item in enumerate(top_n)
            ]

        # LiT5 final listwise pass on top-20
        top_k_pairs = [(d, cand_map[d]) for d, _ in top_n if d in cand_map]
        lit5_ranked = self.lit5.rerank(query, top_k_pairs)

        final = list(lit5_ranked) + list(tail)
        n = len(final)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(final)]


LIT5_DUO_TOP_N = 10   # duoT5 tournament on top-N after LiT5


class LiT5DuoCascade:
    """BM25 → LiT5 (all 50, sliding window) → duoT5 tournament on top-10.

    LiT5 provides a global listwise ordering of all 50 BM25 candidates;
    duoT5 then does precise pairwise comparison on the cream (top-10) where
    the listwise model may still leave near-ties unresolved.
    """

    name = "lit5_duo"

    def __init__(self):
        self.lit5 = registry.get("lit5")
        self.duo  = registry.get("duot5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        cand_map = dict(candidates)

        # Step 1: LiT5 reranks all BM25 candidates
        lit5_ranked = self.lit5.rerank(query, candidates)

        # Step 2: duoT5 tournament on top-10 of LiT5 output
        head = lit5_ranked[:LIT5_DUO_TOP_N]
        tail = lit5_ranked[LIT5_DUO_TOP_N:]

        head_pairs = [(d, cand_map[d]) for d, _ in head if d in cand_map]
        duo_ranked = self.duo.rerank(query, head_pairs)

        merged = list(duo_ranked) + list(tail)
        n = len(merged)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(merged)]


class MonoProximityDuoCascade0005(MonoProximityDuoCascade):
    """Same as MonoProximityDuoCascade with margin=0.0005 (tighter threshold).

    Flags fewer uncertain docs per query; duoT5 is called less often but
    targets only the most tightly-clustered score groups.
    """

    name = "mono_proximity_duo_0005"

    def __init__(self) -> None:
        super().__init__(margin=0.0005)


class MonoProximityDuoCascade005Top30(MonoProximityDuoCascade):
    """Same as MonoProximityDuoCascade with margin=0.005, scans top-30.

    Wider margin and larger scan window: flags more uncertain docs (score
    groups within 0.005 of each other) across a deeper portion of the ranking.
    Useful for queries where relevant docs are spread across positions 20-30.
    """

    name = "mono_proximity_duo_005_top30"

    def __init__(self) -> None:
        super().__init__(margin=0.005, top_n=30)


# ── MAU-sweep gated models ────────────────────────────────────────────────────
# Two operating points selected by the threshold sweep in
# models/monot5/threshold_sweep.py using the top-1/top-2 monoT5 score gap as
# the gating signal (a per-query uncertainty proxy).
#
# τ = 0.0001  → sends ~21.5% of queries to duoT5  nDCG@10 ≈ 0.8731  (cheap)
# τ = 0.0010  → sends ~50.3% of queries to duoT5  nDCG@10 ≈ 0.8797  (Pareto knee)


class MonoMauDuoLowCost(MonoGatedDuoCascade):
    """Gap-gated duoT5 at τ=0.0001 — big nDCG gain for only 21.5% duo cost.

    Selected from the MAU threshold sweep as the elbow of the gain/cost curve:
    the first point where routing the most uncertain queries to duoT5 yields a
    large improvement (+0.011 nDCG vs pure monoT5) before returns diminish.
    """

    name = "mono_mau_duo_low_cost"

    def __init__(self) -> None:
        super().__init__(margin=0.0001)


class MonoMauDuoPareto(MonoGatedDuoCascade):
    """Gap-gated duoT5 at τ=0.001 — Pareto knee of the MAU threshold sweep.

    Sends ~50% of queries to duoT5, recovering ~95% of the achievable nDCG
    gain over pure monoT5 at half the cost of always running duoT5.
    """

    name = "mono_mau_duo_pareto"

    def __init__(self) -> None:
        super().__init__(margin=0.001)


# ── Gated monoT5 → LiT5 cascade ──────────────────────────────────────────────
# Per-query gate: if top-1/top-2 monoT5 gap < τ=0.001, pass top-K docs to LiT5
# for listwise reranking; otherwise keep monoT5 order unchanged.
# Three variants differ only in how many docs LiT5 sees when triggered.


class MonoGatedLiT5Cascade:
    """BM25 → monoT5 (rank all) → LiT5 on top-K only when gap < τ.

    Gate signal: top-1 minus top-2 monoT5 P(true) gap.
    When gap < margin → monoT5 is uncertain at the top → LiT5 reranks top_k docs.
    When gap ≥ margin → monoT5 is confident → LiT5 is skipped entirely.
    """

    name = "mono_gated_lit5"

    def __init__(self, margin: float = 0.001, top_k: int = 20):
        self.margin = margin
        self.top_k  = top_k
        self.mono = registry.get("monot5")
        self.lit5 = registry.get("lit5")

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
    ) -> list[tuple[str, float]]:
        if not candidates:
            return []

        cand_map = dict(candidates)

        mono_ranked = self.mono.rerank(query, candidates)

        gap = (mono_ranked[0][1] - mono_ranked[1][1]) if len(mono_ranked) >= 2 else 1.0

        _log.debug(
            "gap=%.4f  margin=%.4f  trigger=%s  top_k=%d  query=%r",
            gap, self.margin, gap < self.margin, self.top_k, query[:60],
        )

        if gap < self.margin:
            head       = mono_ranked[:self.top_k]
            tail       = mono_ranked[self.top_k:]
            head_pairs = [(d, cand_map[d]) for d, _ in head if d in cand_map]
            lit5_ranked = self.lit5.rerank(query, head_pairs)
            merged      = list(lit5_ranked) + tail
        else:
            merged = mono_ranked

        n = len(merged)
        return [(docid, float(n - i)) for i, (docid, _) in enumerate(merged)]


class MonoGatedLiT5Top20(MonoGatedLiT5Cascade):
    """Gap-gated monoT5 → LiT5 — passes top-20 docs to LiT5 when triggered."""

    name = "mono_gated_lit5_top20"

    def __init__(self) -> None:
        super().__init__(margin=0.001, top_k=20)


class MonoGatedLiT5Top40(MonoGatedLiT5Cascade):
    """Gap-gated monoT5 → LiT5 — passes top-40 docs to LiT5 when triggered."""

    name = "mono_gated_lit5_top40"

    def __init__(self) -> None:
        super().__init__(margin=0.001, top_k=40)


class MonoGatedLiT5Top50(MonoGatedLiT5Cascade):
    """Gap-gated monoT5 → LiT5 — passes all top-50 BM25 docs to LiT5 when triggered."""

    name = "mono_gated_lit5_top50"

    def __init__(self) -> None:
        super().__init__(margin=0.001, top_k=50)
