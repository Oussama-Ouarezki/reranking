from manim import *
from manim_slides import Slide


class RerankerPresentation(Slide):
    def construct(self):

        # ── Slide 1 : Title ──────────────────────────────────────────────────
        title = Text("Neural Reranking Pipeline", font_size=52, weight=BOLD)
        subtitle = Text("BioASQ Passage Retrieval", font_size=30, color=BLUE_C)
        subtitle.next_to(title, DOWN, buff=0.45)
        author = Text("Oussama Ouarezki", font_size=22, color=GREY_B)
        author.next_to(subtitle, DOWN, buff=0.6)

        self.play(Write(title))
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.play(FadeIn(author))
        self.next_slide()

        self.play(FadeOut(VGroup(title, subtitle, author)))

        # ── Slide 2 : Problem statement ──────────────────────────────────────
        header = Text("The Problem", font_size=42, weight=BOLD).to_edge(UP)
        self.play(Write(header))

        bullets = VGroup(
            Text("• Biomedical QA needs precise document retrieval", font_size=26),
            Text("• BM25 misses semantic relevance", font_size=26, color=RED_C),
            Text("• Neural rerankers can fix this — but which one?", font_size=26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        bullets.move_to(ORIGIN + LEFT * 0.5)

        for bullet in bullets:
            self.play(FadeIn(bullet, shift=RIGHT * 0.3), run_time=0.7)
            self.next_slide()

        self.play(FadeOut(VGroup(header, bullets)))

        # ── Slide 3 : Pipeline diagram ────────────────────────────────────────
        header2 = Text("Two-Stage Retrieval", font_size=42, weight=BOLD).to_edge(UP)
        self.play(Write(header2))

        box_data = [
            ("BM25\nRetrieval", YELLOW),
            ("monoT5\nReranker", GREEN),
            ("Top-K\nResults", BLUE),
        ]
        boxes = VGroup(*[
            RoundedRectangle(corner_radius=0.2, width=2.8, height=1.1,
                             color=c, fill_color=c, fill_opacity=0.15)
            for _, c in box_data
        ]).arrange(RIGHT, buff=1.4).move_to(ORIGIN)

        labels = VGroup(*[
            Text(lbl, font_size=22, color=c).move_to(boxes[i])
            for i, (lbl, c) in enumerate(box_data)
        ])

        arrows = VGroup(*[
            Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.1)
            for i in range(len(boxes) - 1)
        ])

        self.play(LaggedStart(*[DrawBorderThenFill(b) for b in boxes], lag_ratio=0.3))
        self.play(LaggedStart(*[Write(l) for l in labels], lag_ratio=0.3))
        self.next_slide()

        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.4))
        self.next_slide()

        # Highlight BM25 → monoT5 handoff
        highlight = SurroundingRectangle(VGroup(boxes[0], boxes[1], arrows[0]),
                                         color=YELLOW, buff=0.2)
        caption = Text("100 candidates → reranked", font_size=20,
                        color=YELLOW).next_to(highlight, DOWN, buff=0.25)
        self.play(Create(highlight), FadeIn(caption))
        self.next_slide()

        self.play(FadeOut(VGroup(header2, boxes, labels, arrows, highlight, caption)))

        # ── Slide 4 : Reranker models ─────────────────────────────────────────
        header3 = Text("Reranker Models", font_size=42, weight=BOLD).to_edge(UP)
        self.play(Write(header3))

        model_rows = [
            ("monoT5", "Pointwise", YELLOW,
             "Query: {q}  Document: {d}  Relevant:"),
            ("duoT5",  "Pairwise",  GREEN,
             "Query: {q}  Doc A: {a}  Doc B: {b}  Prefer A:"),
            ("LiT5",   "Listwise",  BLUE,
             "Rank passages [1]…[N] jointly"),
        ]

        cards = VGroup()
        for name, kind, color, prompt in model_rows:
            card = VGroup(
                RoundedRectangle(corner_radius=0.15, width=11, height=1.0,
                                  color=color, fill_color=color, fill_opacity=0.1),
                Text(f"{name}  ({kind})", font_size=24, weight=BOLD, color=color),
                Text(prompt, font_size=16, color=GREY_A),
            )
            card[1].move_to(card[0]).shift(LEFT * 3)
            card[2].move_to(card[0]).shift(RIGHT * 1.2)
            cards.add(card)

        cards.arrange(DOWN, buff=0.45).move_to(ORIGIN + DOWN * 0.2)

        for card in cards:
            self.play(FadeIn(card[0]), Write(card[1]), run_time=0.7)
            self.play(FadeIn(card[2]), run_time=0.5)
            self.next_slide()

        self.play(FadeOut(VGroup(header3, cards)))

        # ── Slide 5 : Bar-chart results ───────────────────────────────────────
        header4 = Text("nDCG@10 Results — BioASQ", font_size=40, weight=BOLD).to_edge(UP)
        self.play(Write(header4))

        model_scores = [
            ("BM25",        0.50, GREY),
            ("monoT5-10k",  0.65, YELLOW),
            ("monoT5-100k", 0.71, GREEN),
            ("duoT5",       0.74, BLUE),
        ]
        bar_group = VGroup()
        baseline_y = -2.0

        for i, (model, score, color) in enumerate(model_scores):
            bar_h = score * 3.5
            bar = Rectangle(width=1.1, height=bar_h,
                             fill_color=color, fill_opacity=0.85, stroke_width=0)
            bar.move_to(np.array([-4.2 + i * 2.8, baseline_y + bar_h / 2, 0]))
            lbl = Text(model, font_size=17).next_to(bar, DOWN, buff=0.12)
            score_lbl = Text(f"{score:.2f}", font_size=18,
                              color=color).next_to(bar, UP, buff=0.08)
            bar_group.add(VGroup(bar, lbl, score_lbl))

        baseline = DashedLine(
            np.array([-5.5, baseline_y + 0.50 * 3.5, 0]),
            np.array([5.5,  baseline_y + 0.50 * 3.5, 0]),
            color=GREY, dash_length=0.18,
        )
        bl_label = Text("BM25 baseline", font_size=15,
                         color=GREY).next_to(baseline, RIGHT, buff=0.1)

        self.play(Create(baseline), FadeIn(bl_label))
        self.play(LaggedStart(*[GrowFromEdge(g[0], DOWN) for g in bar_group],
                               lag_ratio=0.25))
        self.play(LaggedStart(*[FadeIn(g[1:]) for g in bar_group], lag_ratio=0.25))
        self.next_slide()

        # Animate improvement arrow
        best_bar = bar_group[-1][0]
        arr = DoubleArrow(
            np.array([best_bar.get_right()[0] + 0.3,
                      baseline_y + 0.50 * 3.5, 0]),
            np.array([best_bar.get_right()[0] + 0.3,
                      best_bar.get_top()[1], 0]),
            buff=0, color=ORANGE,
        )
        gain = Text("+48%", font_size=20, color=ORANGE).next_to(arr, RIGHT, buff=0.1)
        self.play(GrowArrow(arr), FadeIn(gain))
        self.next_slide()

        self.play(FadeOut(VGroup(header4, bar_group, baseline, bl_label, arr, gain)))

        # ── Slide 6 : Conclusion ──────────────────────────────────────────────
        header5 = Text("Key Takeaways", font_size=42, weight=BOLD).to_edge(UP)
        self.play(Write(header5))

        takeaways = VGroup(
            Text("✓  Neural rerankers significantly outperform BM25", font_size=26, color=GREEN),
            Text("✓  monoT5-100k: best single-model trade-off",        font_size=26, color=GREEN),
            Text("✓  duoT5 cascade adds +3 nDCG@10 at extra cost",     font_size=26, color=YELLOW),
            Text("→  Fine-tuning on BioASQ yields further gains",       font_size=26, color=BLUE_C),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to(ORIGIN)

        for t in takeaways:
            self.play(FadeIn(t, shift=RIGHT * 0.3), run_time=0.6)
            self.next_slide()

        self.play(FadeOut(VGroup(header5, takeaways)))

        # ── Slide 7 : Thank you ───────────────────────────────────────────────
        thanks = Text("Thank you!", font_size=60, weight=BOLD,
                       gradient=(BLUE, GREEN))
        contact = Text("oussamaouarezki7@gmail.com", font_size=22,
                        color=GREY_B).next_to(thanks, DOWN, buff=0.5)
        self.play(Write(thanks))
        self.play(FadeIn(contact))
        self.next_slide()
