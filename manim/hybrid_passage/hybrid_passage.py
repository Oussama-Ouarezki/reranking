"""
Hybrid Passage Reranking Cascade — Manim-Slides presentation.

A single working column on the right transforms across four stages:
    50 docs (Qwen3-0.6B)
        →  50 docs (+ BM25, Linear Fusion)
            →  top-25 (duoT5 on band 15–25)
                →  top-20 (LiT5, single pass)

No side panels, no captions — just stage titles and motion.

Render then present (or export to HTML/PPTX):
    cd /home/oussama/Desktop/reranking_project
    manim-slides render manim/hybrid_passage/hybrid_passage.py HybridPassage -ql
    manim-slides present HybridPassage             # interactive slideshow
    manim-slides convert HybridPassage out.html    # standalone HTML deck
    manim-slides convert HybridPassage out.pptx    # PowerPoint deck

Render as a plain video instead:
    manim -pql manim/hybrid_passage/hybrid_passage.py HybridPassage
"""

import random
from manim import *
from manim_slides import Slide

random.seed(11)

# ── Geometry — column lives on the right; cards grow as the list shrinks ──
COL_X       = 2.4
TOP_Y       = 3.20
COL_HEIGHT  = 6.40    # visible vertical extent of the column

DIMS_50 = dict(row_h=0.115, row_gap=0.020, row_w=1.55, bar_w=0.70, font=8)
DIMS_25 = dict(row_h=0.225, row_gap=0.040, row_w=2.50, bar_w=1.20, font=14)
DIMS_20 = dict(row_h=0.280, row_gap=0.050, row_w=3.00, bar_w=1.50, font=16)

# Uncertainty band  (LF ranks 15..25)
BAND_FROM = 15
BAND_TO   = 25
TOP_K     = 20

# ── Colours ──────────────────────────────────────────────────────────────
C_BG    = "#0D1117"
C_TEXT  = "#E6EDF3"
C_GREY  = "#8B949E"
C_HEAD  = "#58A6FF"
C_QWEN  = "#7C5CFF"
C_BM25  = "#F5C242"
C_LF    = "#22A6B3"
C_DUO   = "#E1734F"
C_LIT5  = "#F2C94C"
C_GOLD  = "#FFD700"
C_GREEN = "#3FB950"


# ── Synthetic ranking values ─────────────────────────────────────────────
def make_qwen_scores(n: int) -> list[float]:
    out = []
    for i in range(n):
        base = 0.97 - 0.85 * (i / max(n - 1, 1))
        out.append(round(min(max(base + random.uniform(-0.05, 0.05), 0.05), 0.99), 3))
    return sorted(out, reverse=True)


def make_bm25_scores(n: int) -> list[float]:
    out = []
    for i in range(n):
        base = 0.65 - 0.30 * (i / max(n - 1, 1))
        out.append(round(min(max(base + random.uniform(-0.18, 0.18), 0.05), 0.95), 3))
    return out


GOLD_RANKS_AFTER_QWEN = {2, 7, 18, 24}   # 1-indexed seeds for "good" docs


# ── Scene ────────────────────────────────────────────────────────────────
class HybridPassage(Slide):
    def construct(self):
        self.camera.background_color = ManimColor(C_BG)

        self._intro()
        self.next_slide()
        self.cards: list[VGroup] = []
        self.scores: list[float] = make_qwen_scores(50)
        self.bm25:   list[float] = make_bm25_scores(50)
        self.is_gold: list[bool] = [(i + 1) in GOLD_RANKS_AFTER_QWEN
                                    for i in range(50)]

        self.dims = DIMS_50
        self.title = self._make_title("Qwen3-Reranker-0.6B  ·  50 docs", C_HEAD)
        self.play(FadeIn(self.title))
        self._stage1_qwen()
        self.next_slide()
        self._stage2_lf()
        self.next_slide()
        self._stage3_duot5_top25()
        self.next_slide()
        self._stage4_lit5_top20()
        self.next_slide()
        self._outro()

    # ── 0. Title card ────────────────────────────────────────────────────
    def _intro(self):
        title = Text("Hybrid Passage Reranking",
                     font_size=44, color=C_TEXT, weight=BOLD)
        sub = Text("50  →  50  →  25  →  20",
                   font_size=22, color=C_GREY)
        sub.next_to(title, DOWN, buff=0.30)
        self.play(Write(title, run_time=1.0))
        self.play(FadeIn(sub))
        self.wait(1.2)
        self.play(FadeOut(VGroup(title, sub)))

    # ── 1.  50 cards · Qwen3-0.6B ────────────────────────────────────────
    def _stage1_qwen(self):
        self.cards = self._build_column(
            n=50, scores=self.scores, is_gold=self.is_gold,
            fill=C_QWEN, dims=DIMS_50,
        )
        self.play(LaggedStart(*[FadeIn(c, shift=LEFT * 0.30)
                                for c in self.cards],
                              lag_ratio=0.012),
                  run_time=1.5)
        self.play(LaggedStart(*[Indicate(c.score_bar,
                                         color=C_QWEN, scale_factor=1.18)
                                for c in self.cards[:8]],
                              lag_ratio=0.04),
                  run_time=1.0)
        self.wait(0.4)

    # ── 2. + BM25 → Linear Fusion (50 cards, in place) ───────────────────
    def _stage2_lf(self):
        new_title = self._make_title("+ BM25  →  Linear Fusion  ·  50 docs",
                                     C_HEAD)
        self.play(Transform(self.title, new_title))

        # BM25 chip slides in on the right of every card.
        chips: list[Rectangle] = []
        for i, c in enumerate(self.cards):
            chip = Rectangle(
                width=0.06,
                height=max(self.dims["row_h"] * 0.40,
                           self.dims["row_h"] * (0.40 + 0.55 * self.bm25[i])),
                fill_color=C_BM25, fill_opacity=0.95, stroke_width=0,
            )
            chip.move_to([c.get_right()[0] + 0.06,
                          c.get_center()[1], 0])
            chips.append(chip)
            c.bm25_chip = chip
        self.play(LaggedStart(*[FadeIn(ch, shift=LEFT * 0.10)
                                for ch in chips],
                              lag_ratio=0.012),
                  run_time=1.2)
        self.wait(0.3)

        # Fuse: chips dissolve into the card; bars + cards recolour LF teal.
        merge = []
        for c in self.cards:
            merge.append(c.body.animate.set_fill(C_LF, 0.30))
            merge.append(c.score_bar.animate.set_fill(C_LF, 1.0))
            merge.append(FadeOut(c.bm25_chip, scale=0.20))
        self.play(*merge, run_time=1.1)

        # Small permutations near close-ties.
        perm = list(range(50))
        for a, b in [(2, 3), (5, 6), (10, 11), (28, 29), (41, 42)]:
            perm[a], perm[b] = perm[b], perm[a]
        self.play(*[Indicate(self.cards[a].body, color=C_GREEN,
                             scale_factor=1.04)
                    for pair in [(2, 3), (5, 6), (10, 11), (28, 29), (41, 42)]
                    for a in pair],
                  run_time=0.9)
        self._reorder(perm, run_time=1.0)
        self.wait(0.4)

    # ── 3. duoT5 on band 15–25 → top-25 (cards grow) ─────────────────────
    def _stage3_duot5_top25(self):
        new_title = self._make_title("duoT5  ·  band 15–25  ·  top-25",
                                     C_HEAD)
        self.play(Transform(self.title, new_title))

        # 1.  Drop ranks 26..50.
        tail = self.cards[25:]
        self.play(*[FadeOut(c, shift=DOWN * 0.4) for c in tail],
                  run_time=0.9)
        self.cards = self.cards[:25]
        self.scores = self.scores[:25]
        self.bm25 = self.bm25[:25]
        self.is_gold = self.is_gold[:25]

        # 2.  Grow remaining 25 to the larger DIMS_25.
        self.dims = DIMS_25
        new_cards: list[VGroup] = []
        for i, old in enumerate(self.cards):
            new_c = self._make_card(i + 1, self.scores[i], self.is_gold[i],
                                    fill=C_LF, dims=DIMS_25)
            new_cards.append(new_c)
        self.play(*[Transform(old, new, replace_mobject_with_target_in_scene=False)
                    for old, new in zip(self.cards, new_cards)],
                  run_time=1.2)
        # After Transform, internal pointers (body, score_bar, rank_t, is_gold)
        # come from the *original* mobjects but their visuals match the new ones.
        # We swap the underlying state to reference new sub-objects.
        for old, new in zip(self.cards, new_cards):
            old.become(new)
            old.body = new.body
            old.score_bar = new.score_bar
            old.rank_t = new.rank_t
            old.is_gold = new.is_gold

        # 3.  Highlight band 15–25 with a soft orange box.
        band_cards = self.cards[BAND_FROM - 1:BAND_TO]
        top    = band_cards[0].get_top()[1] + 0.06
        bottom = band_cards[-1].get_bottom()[1] - 0.06
        left   = COL_X - DIMS_25["row_w"] / 2 - 0.10
        right  = COL_X + DIMS_25["row_w"] / 2 + 0.10
        band_box = Rectangle(
            width=right - left, height=top - bottom,
            stroke_color=C_DUO, stroke_width=2.2,
            fill_color=C_DUO, fill_opacity=0.06,
        ).move_to([(left + right) / 2, (top + bottom) / 2, 0])
        self.play(Create(band_box))

        # 4.  Recolour band orange.
        self.play(*[c.body.animate.set_fill(C_DUO, 0.32)
                    for c in band_cards],
                  *[c.score_bar.animate.set_fill(C_DUO, 1.0)
                    for c in band_cards],
                  run_time=0.6)

        # 5.  A handful of curved pairwise arrows (no labels).
        BAND_LO_I = BAND_FROM - 1
        sample = [(0, 4), (1, 6), (2, 8), (3, 5), (5, 9), (4, 10)]
        arrows = VGroup()
        for a, b in sample:
            ca = self.cards[BAND_LO_I + a]
            cb = self.cards[BAND_LO_I + b]
            arrows.add(CurvedDoubleArrow(
                ca.get_left() + LEFT * 0.05,
                cb.get_left() + LEFT * 0.05,
                color=C_DUO, stroke_width=1.5,
                tip_length=0.10, angle=-PI / 2.3,
            ))
        self.play(LaggedStart(*[Create(a) for a in arrows],
                              lag_ratio=0.16),
                  run_time=1.3)
        self.wait(0.3)
        self.play(FadeOut(arrows))

        # 6.  Reorder the band; pulse the top-6 promoted into top-20.
        new_band = [7, 3, 0, 5, 1, 9, 2, 4, 6, 8, 10]   # within band only
        full_perm = list(range(25))
        full_perm[BAND_LO_I:BAND_LO_I + 11] = [BAND_LO_I + i for i in new_band]
        self._reorder(full_perm, run_time=1.3)
        promoted = self.cards[BAND_LO_I:BAND_LO_I + 6]
        self.play(*[c.body.animate.set_stroke(C_GREEN, width=2.5,
                                              opacity=1.0)
                    for c in promoted], run_time=0.5)
        self.wait(0.4)
        # Cleanup: remove orange band rectangle AND reset green halos
        # so the next slide starts on a clean column.
        reset_strokes = []
        for c in promoted:
            target_color = C_GOLD if c.is_gold else WHITE
            target_w = 2.0 if c.is_gold else 0.6
            target_op = 1.0 if c.is_gold else 0.20
            reset_strokes.append(
                c.body.animate.set_stroke(target_color,
                                           width=target_w,
                                           opacity=target_op))
        self.play(FadeOut(band_box), *reset_strokes, run_time=0.6)

    # ── 4. LiT5 listwise on top-20 (one pass; cards grow) ────────────────
    def _stage4_lit5_top20(self):
        new_title = self._make_title("LiT5  ·  top-20  ·  one pass",
                                     C_HEAD)
        self.play(Transform(self.title, new_title))

        # 1.  Drop ranks 21..25.
        tail = self.cards[20:]
        self.play(*[FadeOut(c, shift=DOWN * 0.4) for c in tail],
                  run_time=0.7)
        self.cards = self.cards[:20]
        self.scores = self.scores[:20]
        self.is_gold = self.is_gold[:20]

        # 2.  Grow remaining 20 to DIMS_20.
        self.dims = DIMS_20
        new_cards: list[VGroup] = []
        for i, old in enumerate(self.cards):
            new_c = self._make_card(i + 1, self.scores[i], self.is_gold[i],
                                    fill=C_LIT5, dims=DIMS_20)
            new_cards.append(new_c)
        self.play(*[Transform(old, new, replace_mobject_with_target_in_scene=False)
                    for old, new in zip(self.cards, new_cards)],
                  run_time=1.2)
        for old, new in zip(self.cards, new_cards):
            old.become(new)
            old.body = new.body
            old.score_bar = new.score_bar
            old.rank_t = new.rank_t
            old.is_gold = new.is_gold

        # 3.  Window box around top-20 (fills the column).
        top    = self.cards[0].get_top()[1] + 0.08
        bottom = self.cards[-1].get_bottom()[1] - 0.08
        left   = COL_X - DIMS_20["row_w"] / 2 - 0.10
        right  = COL_X + DIMS_20["row_w"] / 2 + 0.10
        win = Rectangle(
            width=right - left, height=top - bottom,
            stroke_color=C_LIT5, stroke_width=3,
            fill_color=C_LIT5, fill_opacity=0.04,
        ).move_to([(left + right) / 2, (top + bottom) / 2, 0])
        self.play(Create(win))

        # 4.  One-pass listwise reorder — pull every gold doc to the head.
        head = list(range(20))
        gold_in_head = [i for i in head if self.cards[i].is_gold]
        non_gold     = [i for i in head if not self.cards[i].is_gold]
        new_head = gold_in_head + non_gold
        # Mild swap for visual rhythm
        if len(new_head) > 5:
            new_head[3], new_head[4] = new_head[4], new_head[3]
        self._reorder(new_head, run_time=1.6)

        self.wait(0.5)
        self.play(FadeOut(win))

    # ── 5. Outro ─────────────────────────────────────────────────────────
    def _outro(self):
        new_title = self._make_title("Hybrid passage cascade  ·  final ranking",
                                     C_GREEN)
        self.play(Transform(self.title, new_title))

        # Halo gold docs in the final list.
        halos = [c.body.animate.set_stroke(C_GOLD, width=3.5, opacity=1.0)
                 for c in self.cards if c.is_gold]
        if halos:
            self.play(*halos, run_time=0.6)

        metrics = VGroup(
            Text("nDCG@1   0.9583", font_size=20, color=C_TEXT),
            Text("nDCG@5   0.9170", font_size=20, color=C_TEXT),
            Text("nDCG@10  0.8913", font_size=20, color=C_TEXT),
            Text("MRR@10   0.9715", font_size=20, color=C_TEXT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.20)
        metrics.move_to([-3.6, 0.0, 0])
        self.play(LaggedStart(*[FadeIn(m) for m in metrics],
                              lag_ratio=0.15))
        self.wait(2.4)

    # ── Mobject builders / helpers ───────────────────────────────────────
    def _make_title(self, text: str, color: str) -> Text:
        t = Text(text, font_size=30, color=color, weight=BOLD)
        t.to_edge(UP, buff=0.30)
        return t

    def _make_card(self, rank: int, score: float, is_gold: bool,
                   fill: str, dims: dict) -> VGroup:
        body = RoundedRectangle(
            width=dims["row_w"], height=dims["row_h"],
            corner_radius=min(0.06, dims["row_h"] * 0.30),
            fill_color=fill, fill_opacity=0.32,
            stroke_color=C_GOLD if is_gold else WHITE,
            stroke_width=2.0 if is_gold else 0.6,
            stroke_opacity=1.0 if is_gold else 0.20,
        )
        rank_t = Text(f"{rank:02d}", font_size=dims["font"],
                      color=C_TEXT, weight=BOLD)
        rank_t.move_to(body.get_left()
                       + RIGHT * (0.10 + dims["row_w"] * 0.03))

        bar = Rectangle(
            width=max(0.05, dims["bar_w"] * score),
            height=dims["row_h"] * 0.40,
            fill_color=fill, fill_opacity=1.0, stroke_width=0,
        )
        bar.move_to([body.get_right()[0] - 0.08 - bar.width / 2,
                     body.get_center()[1], 0])

        g = VGroup(body, rank_t, bar)
        # Position: vertical centre of column = 0; rows fill TOP_Y down.
        # We always use the current self.dims for vertical placement so
        # later stages re-stack cleanly.
        y = self._y_of(rank, dims)
        g.move_to([COL_X, y, 0])
        g.body = body
        g.rank_t = rank_t
        g.score_bar = bar
        g.is_gold = is_gold
        return g

    def _y_of(self, rank: int, dims: dict | None = None) -> float:
        d = dims or self.dims
        # Vertically centre the column on y=0 ish.
        n = len(self.cards) if self.cards else 50
        total_h = n * d["row_h"] + (n - 1) * d["row_gap"]
        top = total_h / 2
        return top - (rank - 1) * (d["row_h"] + d["row_gap"]) - d["row_h"] / 2

    def _build_column(self, n: int, scores: list[float], is_gold: list[bool],
                      fill: str, dims: dict) -> list[VGroup]:
        cards = []
        # Compute layout assuming this column's size is the n we just passed.
        total_h = n * dims["row_h"] + (n - 1) * dims["row_gap"]
        top = total_h / 2
        for i in range(n):
            c = self._make_card(i + 1, scores[i], is_gold[i],
                                fill=fill, dims=dims)
            y = top - i * (dims["row_h"] + dims["row_gap"]) - dims["row_h"] / 2
            c.move_to([COL_X, y, 0])
            cards.append(c)
        return cards

    def _reorder(self, perm: list[int], run_time: float = 1.4) -> None:
        """`perm[i]` = old index that should become rank i+1."""
        d = self.dims
        n = len(self.cards)
        total_h = n * d["row_h"] + (n - 1) * d["row_gap"]
        top = total_h / 2
        anims = []
        new_cards = [self.cards[i] for i in perm]
        for new_idx, c in enumerate(new_cards):
            y = top - new_idx * (d["row_h"] + d["row_gap"]) - d["row_h"] / 2
            anims.append(c.animate.move_to([COL_X, y, 0]))
            new_t = Text(f"{new_idx + 1:02d}", font_size=d["font"],
                         color=C_TEXT, weight=BOLD)
            new_t.move_to(c.body.get_left()
                          + RIGHT * (0.10 + d["row_w"] * 0.03)
                          + (y - c.get_center()[1]) * UP)
            anims.append(Transform(c.rank_t, new_t))
        self.play(*anims, run_time=run_time)
        self.cards = new_cards
        self.scores = [self.scores[i] for i in perm]
        if len(self.bm25) >= len(perm):
            self.bm25 = [self.bm25[i] for i in perm]
        self.is_gold = [self.is_gold[i] for i in perm]
