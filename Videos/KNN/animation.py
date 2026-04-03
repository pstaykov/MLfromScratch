from manim import *


class KNNAnimation(Scene):
    def construct(self):
        # Set background color
        self.camera.background_color = "#000000"

        # Color palette
        text_color = "#ffffff"
        accent_color_1 = "#60b1cd"
        accent_color_2 = "#d4d531"

        # Scene 1: Introduction - Party Scene
        self.intro_party_scene(text_color, accent_color_1, accent_color_2)

        # Scene 2: Explanation of KNN - Scatter Plot
        self.explanation_knn_scene(text_color, accent_color_1, accent_color_2)

        # Scene 3: Distance Calculation - Euclidean Distance
        self.distance_calculation_scene(text_color, accent_color_1, accent_color_2)

        # Scene 4: Choosing K - Small vs Large K
        self.choosing_k_scene(text_color, accent_color_1, accent_color_2)

        # Scene 5: Example - Age vs Income Scatter Plot
        self.example_scene(text_color, accent_color_1, accent_color_2)

        # Scene 6: Advantages and Disadvantages - Balance Scale
        self.advantages_disadvantages_scene(text_color, accent_color_1, accent_color_2)

        # Scene 7: Real-World Applications
        self.real_world_applications_scene(text_color, accent_color_1, accent_color_2)

        # Final wait
        self.wait(10)

    def intro_party_scene(self, text_color, accent_color_1, accent_color_2):
        # Title
        title = Text("K-Nearest Neighbors (KNN)", color=text_color, font_size=36)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Create party scene
        party_people = VGroup()
        preferences = ["Pop", "Rock", "Jazz", "Classical", "Hip-Hop"]
        colors = [accent_color_1, accent_color_2, "#ff6b6b", "#4ecdc4", "#ffe66d"]

        for i in range(5):
            person = Circle(radius=0.3, color=colors[i], fill_opacity=0.5)
            label = Text(preferences[i], color=text_color, font_size=18).next_to(person, UP)
            person_group = VGroup(person, label)
            jitter = np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                0.0,
            ])
            person_group.shift(i * RIGHT * 2 + jitter)
            party_people.add(person_group)

        self.play(LaggedStart(*[Create(person) for person in party_people], lag_ratio=0.5))
        self.wait(2)

        # Highlight new person
        new_person = Circle(radius=0.3, color=WHITE, fill_opacity=0.5)
        new_person_label = Text("New Person", color=text_color, font_size=18).next_to(new_person, UP)
        new_person_group = VGroup(new_person, new_person_label)
        new_person_group.shift(LEFT * 3 + UP * 2)

        self.play(Create(new_person_group))
        self.wait(1)

        # Draw arrows to nearby people
        arrows = VGroup()
        for i in range(2):
            arrow = Arrow(
                start=new_person.get_center(),
                end=party_people[i].get_center(),
                color=accent_color_1,
                buff=0.3
            )
            arrows.add(arrow)

        self.play(LaggedStart(*[Create(arrow) for arrow in arrows], lag_ratio=0.5))
        self.wait(2)

        # Fade out party scene
        self.play(FadeOut(party_people), FadeOut(new_person_group), FadeOut(arrows))
        self.wait(1)

    def explanation_knn_scene(self, text_color, accent_color_1, accent_color_2):
        # Title
        title = Text("How KNN Works", color=text_color, font_size=36)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Create scatter plot
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": text_color},
            x_length=7,
            y_length=7,
        )
        axes_labels = axes.get_axis_labels(x_label="Feature 1", y_label="Feature 2")

        self.play(Create(axes), Write(axes_labels))
        self.wait(1)

        # Add data points
        np.random.seed(42)
        class_1 = np.random.randn(20, 2) * 0.5 + np.array([3, 3])
        class_2 = np.random.randn(20, 2) * 0.5 + np.array([7, 7])

        dots_class_1 = VGroup(*[Dot(axes.c2p(x, y), color=accent_color_1, radius=0.1) for x, y in class_1])
        dots_class_2 = VGroup(*[Dot(axes.c2p(x, y), color=accent_color_2, radius=0.1) for x, y in class_2])

        self.play(LaggedStart(*[Create(dot) for dot in dots_class_1], lag_ratio=0.1))
        self.play(LaggedStart(*[Create(dot) for dot in dots_class_2], lag_ratio=0.1))
        self.wait(1)

        # Add new point
        new_point = Dot(axes.c2p(5, 5), color=WHITE, radius=0.1)
        new_point_label = Text("New Point", color=text_color, font_size=18).next_to(new_point, UP)

        self.play(Create(new_point), Write(new_point_label))
        self.wait(1)

        # Show K=3 circle
        circle_k3 = Circle(radius=1.5, color=RED, stroke_width=2).move_to(new_point.get_center())
        k3_label = Text("K=3", color=RED, font_size=24).next_to(circle_k3, UP)

        self.play(Create(circle_k3), Write(k3_label))
        self.wait(2)

        # Show K=7 circle
        circle_k7 = Circle(radius=2.5, color=GREEN, stroke_width=2).move_to(new_point.get_center())
        k7_label = Text("K=7", color=GREEN, font_size=24).next_to(circle_k7, UP)

        self.play(ReplacementTransform(circle_k3, circle_k7), ReplacementTransform(k3_label, k7_label))
        self.wait(2)

        # Fade out
        self.play(FadeOut(axes), FadeOut(axes_labels), FadeOut(dots_class_1), FadeOut(dots_class_2),
                  FadeOut(new_point), FadeOut(new_point_label), FadeOut(circle_k7), FadeOut(k7_label))
        self.wait(1)

    def distance_calculation_scene(self, text_color, accent_color_1, accent_color_2):
        # Title
        title = Text("Euclidean Distance", color=text_color, font_size=36)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Create axes
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": text_color},
            x_length=7,
            y_length=7,
        )
        axes_labels = axes.get_axis_labels(x_label="X", y_label="Y")

        self.play(Create(axes), Write(axes_labels))
        self.wait(1)

        # Add points
        point_p = Dot(axes.c2p(2, 3), color=accent_color_1, radius=0.1)
        point_q = Dot(axes.c2p(6, 7), color=accent_color_2, radius=0.1)

        label_p = Text("P(2, 3)", color=accent_color_1, font_size=20).next_to(point_p, LEFT)
        label_q = Text("Q(6, 7)", color=accent_color_2, font_size=20).next_to(point_q, RIGHT)

        self.play(Create(point_p), Write(label_p))
        self.play(Create(point_q), Write(label_q))
        self.wait(1)

        # Draw line between points
        line = Line(point_p.get_center(), point_q.get_center(), color=WHITE)
        self.play(Create(line))
        self.wait(1)

        # Show Euclidean distance formula
        formula = MathTex(
            "d(P, Q) = \\sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}",
            color=text_color,
            font_size=36
        ).shift(DOWN * 2)

        self.play(Write(formula))
        self.wait(2)

        # Animate calculation steps
        step1 = MathTex(
            "d(P, Q) = \\sqrt{(6 - 2)^2 + (7 - 3)^2}",
            color=text_color,
            font_size=36
        ).shift(DOWN * 2)

        self.play(ReplacementTransform(formula, step1))
        self.wait(2)

        step2 = MathTex(
            "d(P, Q) = \\sqrt{4^2 + 4^2}",
            color=text_color,
            font_size=36
        ).shift(DOWN * 2)

        self.play(ReplacementTransform(step1, step2))
        self.wait(2)

        step3 = MathTex(
            "d(P, Q) = \\sqrt{16 + 16}",
            color=text_color,
            font_size=36
        ).shift(DOWN * 2)

        self.play(ReplacementTransform(step2, step3))
        self.wait(2)

        step4 = MathTex(
            "d(P, Q) = \\sqrt{32}",
            color=text_color,
            font_size=36
        ).shift(DOWN * 2)

        self.play(ReplacementTransform(step3, step4))
        self.wait(2)

        step5 = MathTex(
            "d(P, Q) \\approx 5.66",
            color=text_color,
            font_size=36
        ).shift(DOWN * 2)

        self.play(ReplacementTransform(step4, step5))
        self.wait(2)

        # Fade out
        self.play(FadeOut(axes), FadeOut(axes_labels), FadeOut(point_p), FadeOut(point_q),
                  FadeOut(label_p), FadeOut(label_q), FadeOut(line), FadeOut(step5))
        self.wait(1)

    def choosing_k_scene(self, text_color, accent_color_1, accent_color_2):
        # Title
        title = Text("Choosing the Right K", color=text_color, font_size=36)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Create scatter plot
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": text_color},
            x_length=7,
            y_length=7,
        )
        axes_labels = axes.get_axis_labels(x_label="Feature 1", y_label="Feature 2")

        self.play(Create(axes), Write(axes_labels))
        self.wait(1)

        # Add data points
        np.random.seed(42)
        class_1 = np.random.randn(20, 2) * 0.5 + np.array([3, 3])
        class_2 = np.random.randn(20, 2) * 0.5 + np.array([7, 7])

        dots_class_1 = VGroup(*[Dot(axes.c2p(x, y), color=accent_color_1, radius=0.1) for x, y in class_1])
        dots_class_2 = VGroup(*[Dot(axes.c2p(x, y), color=accent_color_2, radius=0.1) for x, y in class_2])

        self.play(LaggedStart(*[Create(dot) for dot in dots_class_1], lag_ratio=0.1))
        self.play(LaggedStart(*[Create(dot) for dot in dots_class_2], lag_ratio=0.1))
        self.wait(1)

        # Add new point
        new_point = Dot(axes.c2p(5, 5), color=WHITE, radius=0.1)
        self.play(Create(new_point))
        self.wait(1)

        # Small K (K=1)
        circle_k1 = Circle(radius=0.5, color=RED, stroke_width=2).move_to(new_point.get_center())
        k1_label = Text("K=1", color=RED, font_size=24).next_to(circle_k1, UP)

        self.play(Create(circle_k1), Write(k1_label))
        self.wait(2)

        # Highlight noise sensitivity
        noise_text = Text("Noise Sensitive!", color=RED, font_size=24).shift(DOWN * 2)
        self.play(Write(noise_text))
        self.wait(2)
        self.play(FadeOut(noise_text))

        # Large K (K=15)
        circle_k15 = Circle(radius=3.5, color=GREEN, stroke_width=2).move_to(new_point.get_center())
        k15_label = Text("K=15", color=GREEN, font_size=24).next_to(circle_k15, UP)

        self.play(ReplacementTransform(circle_k1, circle_k15), ReplacementTransform(k1_label, k15_label))
        self.wait(2)

        # Highlight over-generalization
        general_text = Text("Over-Generalized!", color=GREEN, font_size=24).shift(DOWN * 2)
        self.play(Write(general_text))
        self.wait(2)
        self.play(FadeOut(general_text))

        # Optimal K (K=5)
        circle_k5 = Circle(radius=2.0, color=BLUE, stroke_width=2).move_to(new_point.get_center())
        k5_label = Text("K=5", color=BLUE, font_size=24).next_to(circle_k5, UP)

        self.play(ReplacementTransform(circle_k15, circle_k5), ReplacementTransform(k15_label, k5_label))
        self.wait(2)

        optimal_text = Text("Optimal K", color=BLUE, font_size=24).shift(DOWN * 2)
        self.play(Write(optimal_text))
        self.wait(2)

        # Fade out
        self.play(FadeOut(axes), FadeOut(axes_labels), FadeOut(dots_class_1), FadeOut(dots_class_2),
                  FadeOut(new_point), FadeOut(circle_k5), FadeOut(k5_label), FadeOut(optimal_text))
        self.wait(1)

    def example_scene(self, text_color, accent_color_1, accent_color_2):
        # Title
        title = Text("KNN Example", color=text_color, font_size=36)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Create scatter plot
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            axis_config={"color": text_color},
            x_length=7,
            y_length=7,
        )
        axes_labels = axes.get_axis_labels(x_label="Age", y_label="Income")

        self.play(Create(axes), Write(axes_labels))
        self.wait(1)

        # Add data points (Buys and Does not buy)
        buys = np.array([[2, 3], [3, 5], [4, 2], [5, 6], [6, 4], [7, 7]])
        no_buys = np.array([[1, 8], [2, 7], [3, 8], [4, 9], [5, 8], [6, 9]])

        dots_buys = VGroup(*[Dot(axes.c2p(x, y), color=accent_color_1, radius=0.1) for x, y in buys])
        dots_no_buys = VGroup(*[Dot(axes.c2p(x, y), color=accent_color_2, radius=0.1) for x, y in no_buys])

        self.play(LaggedStart(*[Create(dot) for dot in dots_buys], lag_ratio=0.3))
        self.play(LaggedStart(*[Create(dot) for dot in dots_no_buys], lag_ratio=0.3))
        self.wait(1)

        # Add new point
        new_point = Dot(axes.c2p(5, 5), color=WHITE, radius=0.1)
        new_point_label = Text("New Person", color=text_color, font_size=18).next_to(new_point, UP)

        self.play(Create(new_point), Write(new_point_label))
        self.wait(1)

        # Find K=3 nearest neighbors
        circle_k3 = Circle(radius=2.0, color=BLUE, stroke_width=2).move_to(new_point.get_center())
        k3_label = Text("K=3", color=BLUE, font_size=24).next_to(circle_k3, UP)

        self.play(Create(circle_k3), Write(k3_label))
        self.wait(2)

        # Highlight neighbors
        neighbors = VGroup()
        for dot in dots_buys[3:6]:
            neighbors.add(dot)
        for dot in dots_no_buys[3:4]:
            neighbors.add(dot)

        self.play(LaggedStart(*[ApplyMethod(dot.scale, 1.5) for dot in neighbors], lag_ratio=0.3))
        self.wait(2)

        # Show majority class
        majority_text = Text("Majority: Buys", color=accent_color_1, font_size=24).shift(DOWN * 2)
        self.play(Write(majority_text))
        self.wait(2)

        # Fade out
        self.play(FadeOut(axes), FadeOut(axes_labels), FadeOut(dots_buys), FadeOut(dots_no_buys),
                  FadeOut(new_point), FadeOut(new_point_label), FadeOut(circle_k3),
                  FadeOut(k3_label), FadeOut(majority_text))
        self.wait(1)

    def advantages_disadvantages_scene(self, text_color, accent_color_1, accent_color_2):
        # Title
        title = Text("Pros and Cons of KNN", color=text_color, font_size=36)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Create balance scale
        scale = VGroup()
        base = Line(UP * 0.5, DOWN * 0.5, color=text_color, stroke_width=5)
        base.move_to(ORIGIN)

        left_pan = Polygon(
            LEFT * 2 + UP * 0.5,
            RIGHT * 0.5 + UP * 0.5,
            ORIGIN + DOWN * 1,
            LEFT * 2 + DOWN * 1,
            color=accent_color_1,
            fill_opacity=0.5
        )

        right_pan = Polygon(
            RIGHT * 2 + UP * 0.5,
            LEFT * 0.5 + UP * 0.5,
            ORIGIN + DOWN * 1,
            RIGHT * 2 + DOWN * 1,
            color=accent_color_2,
            fill_opacity=0.5
        )

        scale.add(base, left_pan, right_pan)
        scale.shift(UP * 1)

        self.play(Create(scale))
        self.wait(1)

        # Add pros
        pros_title = Text("Advantages", color=accent_color_1, font_size=24).next_to(left_pan, UP)
        pros = BulletedList(
            "Simple to understand",
            "No assumptions about data",
            "Non-parametric",
            color=text_color,
            font_size=20
        ).next_to(pros_title, DOWN)

        self.play(Write(pros_title), Write(pros))
        self.wait(2)

        # Add cons
        cons_title = Text("Disadvantages", color=accent_color_2, font_size=24).next_to(right_pan, UP)
        cons = BulletedList(
            "Computationally expensive",
            "Sensitive to scale",
            "Requires feature scaling",
            color=text_color,
            font_size=20
        ).next_to(cons_title, DOWN)

        self.play(Write(cons_title), Write(cons))
        self.wait(2)

        # Fade out
        self.play(FadeOut(scale), FadeOut(pros_title), FadeOut(pros), FadeOut(cons_title), FadeOut(cons))
        self.wait(1)

    def real_world_applications_scene(self, text_color, accent_color_1, accent_color_2):
        # Title
        title = Text("Real-World Applications", color=text_color, font_size=36)
        self.play(Write(title))
        self.wait(2)
        self.play(FadeOut(title))

        # Create icons for applications
        apps = VGroup()

        # Recommendation Systems
        rec_sys = VGroup()
        rec_icon = Circle(radius=0.5, color=accent_color_1, fill_opacity=0.5)
        rec_text = Text("Recommendation\\nSystems", color=text_color, font_size=18).next_to(rec_icon, DOWN)
        rec_sys.add(rec_icon, rec_text)
        rec_sys.shift(LEFT * 4)
        apps.add(rec_sys)

        # Image Recognition
        img_rec = VGroup()
        img_icon = Square(side_length=1, color=accent_color_2, fill_opacity=0.5)
        img_text = Text("Image\\nRecognition", color=text_color, font_size=18).next_to(img_icon, DOWN)
        img_rec.add(img_icon, img_text)
        img_rec.shift(ORIGIN)
        apps.add(img_rec)

        # Medical Diagnosis
        med_diag = VGroup()
        med_icon = Triangle(color=WHITE, fill_opacity=0.5).scale(0.5)
        med_text = Text("Medical\\nDiagnosis", color=text_color, font_size=18).next_to(med_icon, DOWN)
        med_diag.add(med_icon, med_text)
        med_diag.shift(RIGHT * 4)
        apps.add(med_diag)

        self.play(LaggedStart(*[Create(app) for app in apps], lag_ratio=0.5))
        self.wait(3)

        # Fade out
        self.play(FadeOut(apps))
        self.wait(1)