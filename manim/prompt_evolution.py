"""
Manim animation: Prompt Evolution — boxes only (no prompt text)
Stages:
  1 → Basic prompt
  2 → APEER-enhanced prompt
  3 → APEER + standalone title

Run with:
  manim -pql prompt_evolution.py PromptEvolution   # fast preview
  manim -pqh prompt_evolution.py PromptEvolution   # high quality
"""

from manim import *

# ─── Colour palette ────────────────────────────────────────────────────────────
BG       = "#0D1117"
CARD_BG  = "#161B22"
C1       = "#58A6FF"   # blue   – Stage 1
C2       = "#F78166"   # orange – Stage 2
C3       = "#3FB950"   # green  – Stage 3
TEXT_COL = "#E6EDF3"
DIM_COL  = "#8B949E"
GOLD     = "#FFD700"

STAGES = [
    {
        "tag":         "Stage 1",
        "title":       "Basic Prompt",
        "color":       C1,
        "bullets":     [
            "✗  No ranking criteria",
            "✗  BM25 treated as equal signal",
            "✗  No semantic override guidance",
        ],
        "bullet_color": RED_B,
        "panel_label":  "Limitations",
    },
    {
        "tag":         "Stage 2",
        "title":       "APEER-Enhanced",
        "color":       C2,
        "bullets":     [
            "✓  Explicit 4-level ranking criteria",
            "✓  BM25 demoted to weak prior",
            "✓  Semantic depth prioritised",
            "✓  Comparative ranking instruction",
        ],
        "bullet_color": GREEN_B,
        "panel_label":  "Key changes",
    },
    {
        "tag":         "Stage 3",
        "title":       "APEER + Title",
        "color":       C3,
        "bullets":     [
            "✓  BM25 re-elevated as meaningful signal",
            "✓  'When in doubt → prefer BM25' rule",
            "✓  Title fed as standalone field",
            "✓  truncate_text splits title / abstract",
        ],
        "bullet_color": GREEN_B,
        "panel_label":  "Key changes",
    },
]


# ─── Card: box + badge + title + two section placeholder bars ──────────────────
def make_card(tag, title, color, width=4.4, height=5.2):
    rect = RoundedRectangle(
        corner_radius=0.2, width=width, height=height,
        fill_color=CARD_BG, fill_opacity=1,
        stroke_color=color, stroke_width=3,
    )

    badge_bg = RoundedRectangle(
        corner_radius=0.1, width=1.5, height=0.32,
        fill_color=color, fill_opacity=1, stroke_width=0,
    )
    badge_txt = Text(tag, font="Monospace", font_size=13, color=BG, weight=BOLD)
    badge_txt.move_to(badge_bg)
    badge = VGroup(badge_bg, badge_txt)
    badge.next_to(rect.get_top(), DOWN, buff=0.25)

    title_mob = Text(title, font="Monospace", font_size=18, color=color, weight=BOLD)
    title_mob.next_to(badge, DOWN, buff=0.18)

    div = Line(
        rect.get_left() + RIGHT * 0.3,
        rect.get_right() + LEFT  * 0.3,
        stroke_color=color, stroke_width=1, stroke_opacity=0.35,
    ).next_to(title_mob, DOWN, buff=0.18)

    def section_bar(label, accent):
        bar = RoundedRectangle(
            corner_radius=0.08, width=width - 0.6, height=0.38,
            fill_color=accent, fill_opacity=0.15,
            stroke_color=accent, stroke_width=1.2,
        )
        lbl = Text(label, font="Monospace", font_size=11, color=accent)
        lbl.move_to(bar).align_to(bar, LEFT).shift(RIGHT * 0.15)
        hint_lines = VGroup(*[
            Line(ORIGIN, RIGHT * (width - 1.4 - 0.12 * i),
                 stroke_color=accent, stroke_width=0.7, stroke_opacity=0.22)
            for i in range(3)
        ]).arrange(DOWN, buff=0.07).next_to(bar, DOWN, buff=0.08).align_to(bar, LEFT)
        return VGroup(bar, lbl, hint_lines)

    sys_sec = section_bar("SYSTEM PROMPT", color)
    sys_sec.next_to(div, DOWN, buff=0.22).align_to(rect, LEFT).shift(RIGHT * 0.3)

    usr_sec = section_bar("USER PROMPT", DIM_COL)
    usr_sec.next_to(sys_sec, DOWN, buff=0.30).align_to(sys_sec, LEFT)

    return VGroup(rect, badge, title_mob, div, sys_sec, usr_sec)


# ─── Bullet panel ──────────────────────────────────────────────────────────────
def make_panel(lines, color, label):
    bullets = VGroup(*[
        Text(ln, font="Monospace", font_size=14, color=color) for ln in lines
    ]).arrange(DOWN, aligned_edge=LEFT, buff=0.18)

    panel_rect = SurroundingRectangle(
        bullets, color=color, stroke_width=1.5,
        fill_color=CARD_BG, fill_opacity=0.75,
        buff=0.22, corner_radius=0.12,
    )
    panel_lbl = Text(label, font="Monospace", font_size=12, color=color)
    panel_lbl.next_to(panel_rect, UP, buff=0.08)
    return VGroup(panel_rect, panel_lbl), bullets


# ─── Scene ─────────────────────────────────────────────────────────────────────
class PromptEvolution(Scene):
    def construct(self):
        self.camera.background_color = BG

        header = Text(
            "Prompt Evolution: Biomedical Reranker",
            font="Monospace", font_size=24, color=TEXT_COL, weight=BOLD,
        ).to_edge(UP, buff=0.32)
        sub = Text(
            "Basic  →  APEER  →  APEER + Standalone Title",
            font="Monospace", font_size=13, color=DIM_COL,
        ).next_to(header, DOWN, buff=0.08)
        self.play(FadeIn(header), FadeIn(sub))
        self.wait(0.4)

        transition_labels = [None, "APEER optimises the prompt", "Adding standalone title"]

        for idx, s in enumerate(STAGES):
            # transition arrow
            if transition_labels[idx]:
                lbl = Text(transition_labels[idx], font="Monospace", font_size=15, color=GOLD)
                lbl.move_to(ORIGIN)
                arr = Arrow(
                    lbl.get_right() + RIGHT * 0.1,
                    lbl.get_right() + RIGHT * 1.3,
                    color=GOLD, stroke_width=3,
                )
                self.play(FadeIn(lbl), GrowArrow(arr))
                self.wait(0.5)
                self.play(FadeOut(lbl), FadeOut(arr))

            card = make_card(s["tag"], s["title"], s["color"])
            card.move_to(ORIGIN + LEFT * 2.0)
            self.play(FadeIn(card, shift=UP * 0.25), run_time=0.9)
            self.wait(0.3)

            panel, bullets = make_panel(s["bullets"], s["bullet_color"], s["panel_label"])
            panel.next_to(card, RIGHT, buff=0.5).align_to(card, UP).shift(DOWN * 0.4)
            bullets.move_to(panel[0])

            self.play(FadeIn(panel))
            for b in bullets:
                self.play(FadeIn(b, shift=RIGHT * 0.12), run_time=0.3)
            self.wait(1.6)

            self.play(FadeOut(card), FadeOut(panel), FadeOut(bullets), run_time=0.6)

        # ── final comparison ──
        comp_title = Text(
            "Side-by-side Comparison",
            font="Monospace", font_size=20, color=GOLD, weight=BOLD,
        ).to_edge(UP, buff=0.28)
        self.play(Transform(header, comp_title), FadeOut(sub))

        positions = [LEFT * 4.3, ORIGIN, RIGHT * 4.3]
        cards = []
        for s, pos in zip(STAGES, positions):
            c = make_card(s["tag"], s["title"], s["color"], width=3.6, height=4.4)
            c.move_to(pos + DOWN * 0.35)
            c.scale(0.88)
            cards.append(c)

        self.play(LaggedStart(*[FadeIn(c, shift=UP * 0.18) for c in cards], lag_ratio=0.22))
        self.wait(0.6)

        arrows = []
        for i in range(len(cards) - 1):
            a = Arrow(
                cards[i].get_right(), cards[i + 1].get_left(),
                color=GOLD, stroke_width=2.5, buff=0.06,
            )
            self.play(GrowArrow(a), run_time=0.45)
            arrows.append(a)

        self.wait(2.8)

        self.play(
            *[FadeOut(c) for c in cards],
            *[FadeOut(a) for a in arrows],
            FadeOut(header),
            run_time=0.9,
        )

        end = VGroup(
            Text("APEER Prompt Engineering", font="Monospace",
                 font_size=26, color=GOLD, weight=BOLD),
            Text("Basic  →  Structured criteria  →  Calibrated BM25 + Title",
                 font="Monospace", font_size=14, color=DIM_COL),
        ).arrange(DOWN, buff=0.35).move_to(ORIGIN)
        self.play(FadeIn(end, shift=UP * 0.2))
        self.wait(2.5)
        self.play(FadeOut(end))
