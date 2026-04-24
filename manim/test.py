from manim import *

class TestScene(Scene):
    def construct(self):
        # Create a circle
        circle = Circle(radius=1, color=BLUE)

        # Create a square
        square = Square(side_length=2, color=GREEN)

        # Create some text
        title = Text("Manim is Working!", font_size=48, color=YELLOW)
        title.to_edge(UP)

        # Animate everything
        self.play(Write(title))
        self.play(Create(circle))
        self.wait(0.5)
        self.play(Transform(circle, square))
        self.wait(0.5)
        self.play(circle.animate.set_color(RED).scale(0.5))
        self.wait(1)
        self.play(FadeOut(circle), FadeOut(title))
        self.wait(0.5)
