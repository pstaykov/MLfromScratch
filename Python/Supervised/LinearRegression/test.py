from manim import *
import numpy as np

# Configuration based on the requested palette
config.background_color = "#000000"
TEXT_COLOR = "#ffffff"
ACCENT_BLUE = "#60b1cd"
ACCENT_YELLOW = "#d4d531"


class LinearRegressionViz(Scene):
    def construct(self):
        self.scene_1_intro()
        self.scene_2_hypothesis()
        self.scene_3_cost_function()
        self.scene_4_gradient_descent()
        self.scene_5_putting_it_together()
        self.scene_6_conclusion()

    def scene_1_intro(self):
        title = Tex("Linear Regression", color=TEXT_COLOR).scale(1.5)
        subtitle = Tex("The Math Behind the Magic", color=ACCENT_BLUE).scale(0.8)
        subtitle.next_to(title, DOWN)

        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": TEXT_COLOR}
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")

        points = VGroup()
        data_coords = [(1, 1.5), (2, 2.8), (3, 3.2), (4, 4.9), (5, 5.1), (6, 6.8), (7, 7.5), (8, 8.2)]
        for x, y in data_coords:
            points.add(Dot(axes.c2p(x, y), color=ACCENT_BLUE))

        self.play(Create(axes), Write(axes_labels))
        self.play(LaggedStart(*[Create(p) for p in points], lag_ratio=0.1))
        self.wait(1)

        line = axes.plot(lambda x: x, color=ACCENT_YELLOW, x_range=[0, 9])
        self.play(Create(line), run_time=2)
        self.wait(1)
        self.play(FadeOut(axes), FadeOut(points), FadeOut(line), FadeOut(axes_labels))

    def scene_2_hypothesis(self):
        eq1 = MathTex("y = mx + b", color=TEXT_COLOR).scale(1.5)
        self.play(Write(eq1))
        self.wait(1)

        eq2 = MathTex("f(x) = wx + b", color=TEXT_COLOR).scale(1.5)
        self.play(Transform(eq1, eq2))
        self.wait(1)

        definitions = VGroup(
            Tex("x: Input (Size)", color=ACCENT_BLUE),
            Tex("f(x): Prediction (Price)", color=ACCENT_BLUE),
            Tex("w: Weight (Slope)", color=ACCENT_YELLOW),
            Tex("b: Bias (Intercept)", color=ACCENT_YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8)
        definitions.next_to(eq1, DOWN, buff=1)

        self.play(FadeIn(definitions))
        self.wait(3)
        self.play(FadeOut(eq1), FadeOut(definitions))

    def scene_3_cost_function(self):
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": TEXT_COLOR}
        )
        self.play(Create(axes))

        points = VGroup()
        data_coords = [(2, 2), (4, 4), (6, 6), (8, 8)]
        for x, y in data_coords:
            points.add(Dot(axes.c2p(x, y), color=ACCENT_BLUE))
        self.play(LaggedStart(*[Create(p) for p in points], lag_ratio=0.1))

        bad_line = axes.plot(lambda x: 0.5 * x + 4, color=ACCENT_YELLOW, x_range=[0, 9])
        self.play(Create(bad_line))

        residuals = VGroup()
        for x, y in data_coords:
            y_pred = 0.5 * x + 4
            res = DashedLine(axes.c2p(x, y), axes.c2p(x, y_pred), color=ACCENT_YELLOW, stroke_width=2)
            residuals.add(res)

        self.play(LaggedStart(*[Create(r) for r in residuals], lag_ratio=0.1))
        self.wait(1)

        mse_text = MathTex(
            "J(w,b) = ", "\\frac{1}{n}", "\\sum", "(y_i - (wx_i + b))^2",
            color=TEXT_COLOR
        ).scale(0.8).to_corner(UR).set_stroke(color=BLACK, width=3, background=True)

        self.play(FadeIn(mse_text))
        self.wait(3)
        self.play(FadeOut(axes), FadeOut(points), FadeOut(bad_line), FadeOut(residuals), FadeOut(mse_text))

    def scene_4_gradient_descent(self):
        axes_gd = Axes(
            x_range=[-1, 5, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=4,
            axis_config={"color": TEXT_COLOR}
        ).shift(LEFT)

        x_label = MathTex("w", color=TEXT_COLOR).next_to(axes_gd.x_axis, RIGHT)
        y_label = Tex("Cost", color=TEXT_COLOR).next_to(axes_gd.y_axis, UP)

        parabola = axes_gd.plot(lambda x: (x - 2) ** 2, color=ACCENT_BLUE, x_range=[-0.5, 4.5])
        dot = Dot(axes_gd.c2p(4.5, (4.5 - 2) ** 2), color=ACCENT_YELLOW)

        self.play(Create(axes_gd), Write(x_label), Write(y_label))
        self.play(Create(parabola))
        self.play(FadeIn(dot))

        update_rule = MathTex(
            "w_{new} = w_{old} - \\alpha \\frac{\\partial J}{\\partial w}",
            color=TEXT_COLOR
        ).scale(0.7).next_to(axes_gd, RIGHT, buff=0.5)
        self.play(FadeIn(update_rule))

        steps = [4.5, 3.5, 2.8, 2.3, 2.0]
        for i in range(len(steps) - 1):
            end = steps[i + 1]
            self.play(
                dot.animate.move_to(axes_gd.c2p(end, (end - 2) ** 2)),
                run_time=1,
                path_arc=-0.3
            )

        self.wait(2)
        self.play(FadeOut(axes_gd), FadeOut(parabola), FadeOut(dot), FadeOut(update_rule), FadeOut(x_label),
                  FadeOut(y_label))

    def scene_5_putting_it_together(self):
        # Left side: Cost
        axes_cost = Axes(
            x_range=[0, 2, 0.5],
            y_range=[0, 10, 2],
            x_length=4,
            y_length=3,
            axis_config={"color": TEXT_COLOR}
        ).to_edge(LEFT, buff=0.5).shift(UP * 0.5)

        parabola = axes_cost.plot(lambda x: 10 * (x - 1) ** 2, color=ACCENT_BLUE, x_range=[0.2, 1.8])
        cost_dot = Dot(axes_cost.c2p(0.2, 10 * (0.2 - 1) ** 2), color=ACCENT_YELLOW)

        # Right side: Data
        axes_data = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 10, 2],
            x_length=4,
            y_length=3,
            axis_config={"color": TEXT_COLOR}
        ).to_edge(RIGHT, buff=0.5).shift(UP * 0.5)

        data_coords = [(1, 1.5), (3, 3.2), (5, 5.1), (7, 7.5), (9, 8.8)]
        points = VGroup(*[Dot(axes_data.c2p(x, y), color=ACCENT_BLUE) for x, y in data_coords])

        # Initial Line
        line = axes_data.plot(lambda x: 0.2 * x + 1, color=ACCENT_YELLOW, x_range=[0, 9])

        self.play(
            Create(axes_cost), Create(parabola), FadeIn(cost_dot),
            Create(axes_data), Create(points), Create(line)
        )

        label_cost = Tex("Cost Function", color=TEXT_COLOR).scale(0.6).next_to(axes_cost, UP)
        label_data = Tex("Data Space", color=TEXT_COLOR).scale(0.6).next_to(axes_data, UP)
        self.play(Write(label_cost), Write(label_data))

        w_steps = [0.2, 0.5, 0.8, 0.95, 1.0]
        b_steps = [1.0, 0.8, 0.6, 0.5, 0.5]

        for i in range(len(w_steps) - 1):
            w_next = w_steps[i + 1]
            b_next = b_steps[i + 1]

            new_line = axes_data.plot(lambda x: w_next * x + b_next, color=ACCENT_YELLOW, x_range=[0, 9])

            self.play(
                cost_dot.animate.move_to(axes_cost.c2p(w_next, 10 * (w_next - 1) ** 2)),
                ReplacementTransform(line, new_line),
                run_time=0.8
            )
            line = new_line

        self.wait(2)
        self.play(FadeOut(Group(*self.mobjects)))  # Clean clear

    def scene_6_conclusion(self):
        text = Tex("Linear Regression", color=TEXT_COLOR).scale(1.5)
        sub = Tex("Hypothesis + Cost Function + Gradient Descent", color=ACCENT_BLUE).scale(0.8)
        sub.next_to(text, DOWN)

        self.play(FadeIn(text), FadeIn(sub))
        self.wait(3)
        self.play(FadeOut(text), FadeOut(sub))