from manim import *
from visualizations.utils import *

class ResidualConnection(Scene):
    def construct(self):
        # Create title
        title = Text("Dropouts, then Residuals, then Layers Normalized", font_size=36)
        title.to_edge(UP, buff=0.5)

        # Create embedding tables
        q_original = EmbeddingTable(rows=5, columns=4, color=BLUE, title="Q Original", scale=0.5)
        q = EmbeddingTable(rows=5, columns=4, color=BLUE_E, title="Output", scale=0.5, black_squares=[3, 6, 14])
        k_original = EmbeddingTable(rows=5, columns=4, color=GREEN, title="K Original", scale=0.5)
        k = EmbeddingTable(rows=5, columns=4, color=GREEN_E, title="Output", scale=0.5, black_squares=[3, 6, 14])
        v_original = EmbeddingTable(rows=5, columns=4, color=RED, title="V Original", scale=0.5)
        v = EmbeddingTable(rows=5, columns=4, color=RED_E, title="Output", scale=0.5, black_squares=[3, 6, 14])

        # Arrange tables
        tables = VGroup(q, q_original, k, k_original, v, v_original)
        tables.arrange_in_grid(rows=1, cols=6, buff=0.7)
        tables.next_to(title, DOWN, buff=1)

        # Create plus signs
        plus_signs = VGroup(*[Text("+", font_size=36) for _ in range(3)])
        for i, plus in enumerate(plus_signs):
            plus.move_to((tables[2*i].get_right() + tables[2*i+1].get_left()) / 2)

        # Create vertical lines
        vertical_lines = VGroup()
        for i in range(1, 3):
            line = Line(
                start=tables[2*i-1].get_top() + UP * 0.5,
                end=tables[2*i-1].get_bottom() + DOWN * 0.5,
                color=WHITE
            )
            line.move_to((tables[2*i-1].get_right() + tables[2*i].get_left()) / 2)
            vertical_lines.add(line)

        # Add everything to the scene
        self.play(
            Write(title),
            *[Create(table) for table in tables],
            *[Write(plus) for plus in plus_signs],
            *[Create(line) for line in vertical_lines]
        )

        self.wait(2)
        self.wait(15)  # Added wait time