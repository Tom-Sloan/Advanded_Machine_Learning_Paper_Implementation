from visualizations.utils import *
from manim import *

class MergingKeyQueryValue(Scene):
    def construct(self):
        # Create label for Key
        key_label = Text("Output", font_size=36)

        # Create half tables for key with surrounding rectangles
        k1_table = EmbeddingTable(rows=6, columns=2, color=GREEN, title="O1", scale=1)
        k1_outline = SurroundingRectangle(k1_table, buff=0.1, color=RED)
        k1_group = VGroup(k1_table, k1_outline)

        k2_table = EmbeddingTable(rows=6, columns=2, color=GREEN, title="O2", scale=1)
        k2_outline = SurroundingRectangle(k2_table, buff=0.1, color=RED)
        k2_group = VGroup(k2_table, k2_outline)

        # Position the tables on opposite sides of the scene
        k1_group.move_to(LEFT * 3)
        k2_group.move_to(RIGHT * 3)
        
        key_group = VGroup(k1_group, k2_group)
        
        # Position the title on top of the scene
        key_label.to_edge(UP, buff=0.5)
        
        # Animation
        self.play(
            Create(k1_group), Create(k2_group)
        )
        
        # Remove rectangles and titles
        self.play(
            FadeOut(k1_outline), FadeOut(k2_outline),
            k1_table.fade_out_title(), k2_table.fade_out_title()
        )
        # Set the opacity of the titles to 0
        k1_table.get_label().set_opacity(0)
        k2_table.get_label().set_opacity(0)
        
        # Animate the tables meeting in the center
        self.play(
            Write(key_label),
            k1_table.animate.move_to(ORIGIN + LEFT * k1_table.width / 2 + LEFT * 0.1),
            k2_table.animate.move_to(ORIGIN + RIGHT * k2_table.width / 2 + RIGHT * 0.1)
        )
        
        self.wait(5)
        self.wait(15)  # Added wait time