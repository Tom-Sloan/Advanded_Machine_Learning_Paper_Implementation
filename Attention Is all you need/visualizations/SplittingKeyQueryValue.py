from visualizations.utils import *
from manim import *

class SplittingKeyQueryValue(Scene):
    def construct(self):
        # Create half tables for query, key, and value
        q1_table = EmbeddingTable(rows=6, columns=2, color=BLUE, title="Q1", scale=0.8)
        q2_table = EmbeddingTable(rows=6, columns=2, color=BLUE, title="Q2", scale=0.8)
        k1_table = EmbeddingTable(rows=6, columns=2, color=GREEN, title="K1", scale=0.8)
        k2_table = EmbeddingTable(rows=6, columns=2, color=GREEN, title="K2", scale=0.8)
        v1_table = EmbeddingTable(rows=6, columns=2, color=RED, title="V1", scale=0.8)
        v2_table = EmbeddingTable(rows=6, columns=2, color=RED, title="V2", scale=0.8)
        
        # Arrange the half tables with added buff
        q1_table.next_to(q2_table, LEFT, buff=q1_table.buff)
        k1_table.next_to(k2_table, LEFT, buff=k1_table.buff)
        v1_table.next_to(v2_table, LEFT, buff=v1_table.buff)
        
        query_group = VGroup(q1_table, q2_table).move_to(LEFT * 4)
        key_group = VGroup(k1_table, k2_table).move_to(ORIGIN)
        value_group = VGroup(v1_table, v2_table).move_to(RIGHT * 4)
        
        # Create labels for Query, Key, and Value
        query_label = Text("Query", font_size=36).next_to(query_group, UP, buff=0.5)
        key_label = Text("Key", font_size=36).next_to(key_group, UP, buff=0.5)
        value_label = Text("Value", font_size=36).next_to(value_group, UP, buff=0.5)
        
        # Create text for number of heads
        num_heads_text = Text("Number of Heads = 2", font_size=36).to_edge(DOWN)
        
        q1_table.get_label().set_opacity(0)
        q2_table.get_label().set_opacity(0)
        k1_table.get_label().set_opacity(0)
        k2_table.get_label().set_opacity(0)
        v1_table.get_label().set_opacity(0)
        v2_table.get_label().set_opacity(0)

        # Animation
        self.play(
            Create(q1_table), Create(q2_table),
            Create(k1_table), Create(k2_table),
            Create(v1_table), Create(v2_table),
            Write(query_label),
            Write(key_label),
            Write(value_label)
        )

        self.play(Write(num_heads_text))

        q1_move = q1_table.animate.next_to(q2_table, LEFT, buff=0.5)
        k1_move = k1_table.animate.next_to(k2_table, LEFT, buff=0.5)
        v1_move = v1_table.animate.next_to(v2_table, LEFT, buff=0.5)

        # Play the animations simultaneously for a smooth effect
        self.play(q1_move, k1_move, v1_move, run_time=1.5, rate_func=smooth)
        
        # Create surrounding rectangles for each half table
        q1_outline = SurroundingRectangle(q1_table, buff=0.1, color=RED)
        q2_outline = SurroundingRectangle(q2_table, buff=0.1, color=RED)
        k1_outline = SurroundingRectangle(k1_table, buff=0.1, color=RED)
        k2_outline = SurroundingRectangle(k2_table, buff=0.1, color=RED)
        v1_outline = SurroundingRectangle(v1_table, buff=0.1, color=RED)
        v2_outline = SurroundingRectangle(v2_table, buff=0.1, color=RED)
        
        self.play(
            Create(q1_outline), Create(q2_outline),
            Create(k1_outline), Create(k2_outline),
            Create(v1_outline), Create(v2_outline),
            q1_table.get_label().animate.set_opacity(1),
            q2_table.get_label().animate.set_opacity(1),
            k1_table.get_label().animate.set_opacity(1),
            k2_table.get_label().animate.set_opacity(1),
            v1_table.get_label().animate.set_opacity(1),
            v2_table.get_label().animate.set_opacity(1),
            run_time=1.5
        )

        self.wait(2)
        self.wait(15)  # Added wait time