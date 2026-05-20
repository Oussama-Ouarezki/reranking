"""
Manim animation: Sliding Window Reranking — real BioASQ case study.

Story: PDGFRA paper (27582545) starts at BM25 rank 37 and bubbles left
       to rank 1 through 4 sliding window passes (top-50, w=20, step=10).

Run:
    cd /home/oussama/Desktop/reranking_project
    manim -pql manim/sliding_window_reranking.py SlidingWindowReranking
    manim -pqh manim/sliding_window_reranking.py SlidingWindowReranking
"""

from manim import *

# ── Real data ────────────────────────────────────────────────────────────────

BM25_DOCS = [
    "35978801", "36031351", "41596345", "25219808", "20650261",
    "25182241", "41423525", "41565660", "30530503", "21205076",
    "32680567", "26376656", "11514572", "26503946", "19270726",
    "23723712", "26091668", "38391974", "20657384", "26744350",
    "11515790", "26294238", "22658319", "33194604", "41514664",
    "17332925", "22891331", "24410805", "41483402", "22160058",
    "23104988", "26811689", "16009451", "21829174", "22052279",
    "29330210", "27582545", "21213368", "23664448", "15126357",
    "27912836", "18779539", "24285547", "26948367", "17575237",
    "20007486", "23499906", "27392443", "18180842", "41549586",
]

GOLD_DOCS = {
    "20479398", "26744350", "27582545",
    "23970477", "26727948", "35978801", "23438035",
}

KEY_DOC  = "27582545"   # PDGFRA paper
GOLD2_DOC = "26744350"  # second gold doc in top-50
GOLD1_DOC = "35978801"  # first gold doc (BM25 rank 1)

PASSES = [
    (30, 50, "Pass 1 · ranks 31 – 50"),
    (20, 40, "Pass 2 · ranks 21 – 40"),
    (10, 30, "Pass 3 · ranks 11 – 30"),
    ( 0, 20, "Pass 4 · ranks  1 – 20"),
]


def _build_states():
    states = [list(BM25_DOCS)]
    for s, e, _ in PASSES:
        prev   = list(states[-1])
        window = list(prev[s:e])
        key_in = KEY_DOC in window
        g2_in  = GOLD2_DOC in window
        if key_in:  window.remove(KEY_DOC)
        if g2_in:   window.remove(GOLD2_DOC)
        new_window = (([KEY_DOC]   if key_in else []) +
                      ([GOLD2_DOC] if g2_in  else []) +
                      window)
        states.append(prev[:s] + new_window + prev[e:])
    return states


STATES = _build_states()

FINAL_TOP20 = [
    "27582545", "35978801", "41596345", "36031351", "25219808",
    "25182241", "26744350", "32680567", "20650261", "41565660",
    "11514572", "19270726", "30530503", "26091668", "38391974",
    "11515790", "41423525", "21205076", "26503946", "41483402",
]
STATES[4] = FINAL_TOP20 + [d for d in STATES[3] if d not in set(FINAL_TOP20)]


# ── Colours ───────────────────────────────────────────────────────────────────
C_BG      = "#0D1117"
C_KEY     = "#FF6B35"
C_GOLD    = "#FFD700"
C_PLAIN   = "#2D6A9F"
C_WIN_BG  = "#FFFFFF"
C_WIN_STK = "#FFEB3B"
C_TEXT    = "#E6EDF3"
C_GREY    = "#8B949E"
C_GREEN   = "#3FB950"
C_HEAD    = "#58A6FF"


# ── Scene ─────────────────────────────────────────────────────────────────────
class SlidingWindowReranking(Scene):

    N      = 50
    SQ     = 0.20          # square side length
    GAP    = 0.04          # horizontal gap between squares
    STEP_X = SQ + GAP      # = 0.24 per rank slot
    ROW_Y  = -0.5          # y-centre of the row
    LEFT_X = -(N - 1) / 2 * STEP_X   # x-centre of rank-1 square

    def rank_to_x(self, rank: int) -> float:
        return self.LEFT_X + (rank - 1) * self.STEP_X

    def bar_color(self, doc: str) -> str:
        if doc == KEY_DOC:   return C_KEY
        if doc in GOLD_DOCS: return C_GOLD
        return C_PLAIN

    def make_bar(self, rank: int, doc: str) -> Square:
        sq = Square(
            side_length=self.SQ,
            fill_color=self.bar_color(doc),
            fill_opacity=0.90,
            stroke_width=0.7,
            stroke_color=WHITE,
            stroke_opacity=0.18,
        )
        sq.move_to([self.rank_to_x(rank), self.ROW_Y, 0])
        return sq

    def make_window_rect(self, win_start: int, win_end: int) -> Rectangle:
        x_l = self.rank_to_x(win_start + 1) - self.SQ / 2 - 0.05
        x_r = self.rank_to_x(win_end)        + self.SQ / 2 + 0.05
        rect = Rectangle(
            width=x_r - x_l, height=self.SQ + 0.18,
            stroke_color=C_WIN_STK, stroke_width=2.5,
            fill_color=C_WIN_BG, fill_opacity=0.06,
        )
        rect.move_to([(x_l + x_r) / 2, self.ROW_Y, 0])
        return rect

    # ── Main ──────────────────────────────────────────────────────────────────
    def construct(self):
        self.camera.background_color = C_BG
        self._title_card()
        self._setup_scene()
        self._show_bm25()
        self._run_passes()
        self._final_summary()

    # ── Title card ────────────────────────────────────────────────────────────
    def _title_card(self):
        title = Text("Sliding Window Reranking", font_size=46, color=C_TEXT, weight=BOLD)
        sub   = Text("How a gold document bubbles from rank 37 → rank 1",
                     font_size=22, color=C_GREY)
        src   = Text("BioASQ · real query · DeepSeek-Chat", font_size=17, color=C_GREY)
        sub.next_to(title, DOWN, buff=0.35)
        src.next_to(sub,   DOWN, buff=0.20)
        self.play(Write(title, run_time=1.2))
        self.play(FadeIn(sub), FadeIn(src))
        self.wait(2)
        self.play(FadeOut(VGroup(title, sub, src)))

    # ── Static scene elements ─────────────────────────────────────────────────
    def _setup_scene(self):
        legend = self._make_legend()
        legend.to_corner(UR, buff=0.35)

        rank_labels = VGroup()
        for r in [1, 10, 20, 30, 40, 50]:
            lbl = Text(str(r), font_size=11, color=C_GREY)
            lbl.move_to([self.rank_to_x(r), self.ROW_Y - self.SQ / 2 - 0.22, 0])
            rank_labels.add(lbl)

        self.play(FadeIn(legend), FadeIn(rank_labels))
        self._rank_labels = rank_labels

    def _make_legend(self) -> VGroup:
        def item(color, label):
            sq  = Square(side_length=0.16, fill_color=color,
                         fill_opacity=0.92, stroke_width=0)
            txt = Text(label, font_size=13, color=C_TEXT)
            txt.next_to(sq, RIGHT, buff=0.10)
            return VGroup(sq, txt)

        return VGroup(
            item(C_KEY,   "Key gold doc  (PDGFRA paper)"),
            item(C_GOLD,  "Other gold doc"),
            item(C_PLAIN, "Non-relevant"),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

    # ── Draw initial BM25 squares ─────────────────────────────────────────────
    def _show_bm25(self):
        self._bars: dict[str, Square] = {}
        bar_vgroup = VGroup()

        for rank, doc in enumerate(BM25_DOCS, 1):
            b = self.make_bar(rank, doc)
            self._bars[doc] = b
            bar_vgroup.add(b)

        heading = Text("BM25 Initial Ranking  (top-50)",
                       font_size=20, color=C_HEAD, weight=BOLD)
        heading.to_edge(UP, buff=0.3)

        self.play(Create(bar_vgroup, run_time=1.8, lag_ratio=0.01), FadeIn(heading))

        key_bar = self._bars[KEY_DOC]
        callout = Text("PDGFRA paper · rank 37", font_size=14, color=C_KEY)
        callout.next_to(key_bar, UP, buff=0.28)
        arrow = Arrow(
            callout.get_bottom(), key_bar.get_top(),
            color=C_KEY, buff=0.04, stroke_width=2.5,
            max_tip_length_to_length_ratio=0.22,
        )
        self.play(FadeIn(callout), GrowArrow(arrow))
        self.wait(1.5)
        self.play(FadeOut(callout), FadeOut(arrow), FadeOut(heading))

    # ── 4 sliding window passes ───────────────────────────────────────────────
    def _run_passes(self):
        window_rect = None
        pass_label  = None
        rank_badge  = None

        for pass_idx, (win_start, win_end, label_text) in enumerate(PASSES):
            prev_state = STATES[pass_idx]
            next_state = STATES[pass_idx + 1]

            new_win = self.make_window_rect(win_start, win_end)
            new_lbl = Text(label_text, font_size=20, color=C_WIN_STK, weight=BOLD)
            new_lbl.to_edge(DOWN, buff=0.35)

            if window_rect is None:
                self.play(Create(new_win), Write(new_lbl), run_time=0.6)
                window_rect = new_win
                pass_label  = new_lbl
            else:
                self.play(
                    Transform(window_rect, new_win, run_time=0.7),
                    Transform(pass_label,  new_lbl, run_time=0.7),
                )
            self.wait(0.4)

            anims = []
            for doc in set(prev_state) | set(next_state):
                old_rank = prev_state.index(doc) + 1 if doc in prev_state else None
                new_rank = next_state.index(doc) + 1 if doc in next_state else None
                if new_rank is not None and old_rank != new_rank and doc in self._bars:
                    anims.append(
                        self._bars[doc].animate.move_to(
                            [self.rank_to_x(new_rank), self.ROW_Y, 0]
                        )
                    )
            if anims:
                self.play(*anims, run_time=1.4, rate_func=smooth)

            new_rank_key = next_state.index(KEY_DOC) + 1
            badge_txt = Text(f"rank {new_rank_key}", font_size=14,
                             color=C_KEY, weight=BOLD)
            badge_txt.next_to(self._bars[KEY_DOC], UP, buff=0.26)

            if rank_badge is None:
                self.play(FadeIn(badge_txt))
                rank_badge = badge_txt
            else:
                self.play(Transform(rank_badge, badge_txt, run_time=0.5))
            self.wait(1.0)

        self.wait(0.5)
        self.play(FadeOut(window_rect), FadeOut(pass_label), FadeOut(rank_badge))

    # ── Final summary ─────────────────────────────────────────────────────────
    def _final_summary(self):
        annotations = VGroup()
        for doc, txt, color in [
            (KEY_DOC,   "★1\n(was 37)", C_KEY),
            (GOLD1_DOC, "★2\n(was 1)",  C_GOLD),
            (GOLD2_DOC, "★7\n(was 20)", C_GOLD),
        ]:
            lbl = Text(txt, font_size=13, color=color, line_spacing=0.75)
            lbl.next_to(self._bars[doc], UP, buff=0.18)
            annotations.add(lbl)

        result_line = Text(
            "PDGFRA paper: BM25 rank 37  →  reranked rank 1",
            font_size=21, color=C_GREEN, weight=BOLD,
        )
        result_line.to_edge(DOWN, buff=0.38)

        stats = Text(
            "nDCG@10: 27.5% (BM25)  →  54.0% (reranked)   +26.5 pts",
            font_size=17, color=C_TEXT,
        )
        stats.next_to(result_line, UP, buff=0.22)

        self.play(
            Write(result_line),
            LaggedStart(*[FadeIn(a) for a in annotations], lag_ratio=0.4),
        )
        self.play(FadeIn(stats))
        self.wait(3)
