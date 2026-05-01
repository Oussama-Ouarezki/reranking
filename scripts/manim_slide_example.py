from manim import *


class RerankerSlide(Scene):
    def construct(self):
        # --- Slide 1: Title ---
        title = Text("Neural Reranking Pipeline", font_size=52, weight=BOLD)
        subtitle = Text("BioASQ Passage Retrieval", font_size=30, color=BLUE_C)
        subtitle.next_to(title, DOWN, buff=0.4)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle, shift=UP * 0.3))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        # --- Slide 2: Pipeline diagram ---
        header = Text("Two-Stage Retrieval", font_size=40, weight=BOLD)
        header.to_edge(UP)
        self.play(Write(header))

        boxes = VGroup(
            RoundedRectangle(corner_radius=0.2, width=2.8, height=1.0, color=YELLOW),
            RoundedRectangle(corner_radius=0.2, width=2.8, height=1.0, color=GREEN),
            RoundedRectangle(corner_radius=0.2, width=2.8, height=1.0, color=BLUE),
        )
        boxes.arrange(RIGHT, buff=1.2)
        boxes.move_to(ORIGIN)

        labels = VGroup(
            Text("BM25\nRetrieval", font_size=22, color=YELLOW),
            Text("monoT5\nReranker", font_size=22, color=GREEN),
            Text("Top-K\nResults", font_size=22, color=BLUE),
        )
        for label, box in zip(labels, boxes):
            label.move_to(box.get_center())

        arrows = VGroup(
            Arrow(boxes[0].get_right(), boxes[1].get_left(), buff=0.1),
            Arrow(boxes[1].get_right(), boxes[2].get_left(), buff=0.1),
        )

        self.play(LaggedStart(*[DrawBorderThenFill(b) for b in boxes], lag_ratio=0.3))
        self.play(LaggedStart(*[Write(l) for l in labels], lag_ratio=0.3))
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.4))
        self.wait(1.5)
        self.play(FadeOut(VGroup(header, boxes, labels, arrows)))

        # --- Slide 3: Metric bar chart ---
        header2 = Text("nDCG@10 Results", font_size=40, weight=BOLD)
        header2.to_edge(UP)
        self.play(Write(header2))

        models = ["BM25", "monoT5\n10k", "monoT5\n100k", "duoT5"]
        scores = [0.50, 0.65, 0.71, 0.74]
        colors = [GREY, YELLOW, GREEN, BLUE]

        bar_width = 1.0
        bar_group = VGroup()
        for i, (model, score, color) in enumerate(zip(models, scores, colors)):
            bar_height = score * 4
            bar = Rectangle(width=bar_width, height=bar_height, fill_color=color, fill_opacity=0.85, stroke_width=0)
            bar.move_to(np.array([-4.5 + i * 2.5, -2.5 + bar_height / 2, 0]))

            label = Text(model, font_size=18).next_to(bar, DOWN, buff=0.15)
            score_label = Text(f"{score:.2f}", font_size=20, color=color).next_to(bar, UP, buff=0.1)
            bar_group.add(VGroup(bar, label, score_label))

        baseline = DashedLine(
            start=np.array([-5.5, -2.5 + 0.50 * 4, 0]),
            end=np.array([5.5, -2.5 + 0.50 * 4, 0]),
            color=GREY, dash_length=0.15,
        )
        baseline_label = Text("BM25 baseline", font_size=16, color=GREY).next_to(baseline, RIGHT, buff=0.1)

        self.play(LaggedStart(*[GrowFromEdge(g[0], DOWN) for g in bar_group], lag_ratio=0.2))
        self.play(LaggedStart(*[FadeIn(g[1:]) for g in bar_group], lag_ratio=0.2))
        self.play(Create(baseline), FadeIn(baseline_label))
        self.wait(2)
        self.play(FadeOut(VGroup(header2, bar_group, baseline, baseline_label)))

        # --- Slide 4: Closing ---
        closing = Text("Thank you!", font_size=56, weight=BOLD, gradient=(BLUE, GREEN))
        self.play(Write(closing))
        self.wait(1.5)
