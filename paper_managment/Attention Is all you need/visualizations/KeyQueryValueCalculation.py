from visualizations.utils import *
from manim import *
class KeyQueryValueCalculation(Scene):
    def construct(self):
        # Create a reusable embedding table
        embedding_table = EmbeddingTable(rows=6, columns=4, color=YELLOW, title="Embedding + Position Table", scale=0.8)
        query_table = EmbeddingTable(rows=4, columns=4, color=BLUE, title="Query Linear NN", scale=0.6)
        key_table = EmbeddingTable(rows=4, columns=4, color=GREEN, title="Key Linear NN", scale=0.6)
        value_table = EmbeddingTable(rows=4, columns=4, color=RED, title="Value Linear NN", scale=0.6)
        
        # Move the embedding table to the left quarter of the screen
        embedding_table.move_to(LEFT * 4)
        
        # Arrange query, key, value tables on the right side with vertical offset
        query_table.move_to(RIGHT * 2 + UP * 2)
        key_table.move_to(RIGHT * 5)
        value_table.move_to(RIGHT * 2 + DOWN * 2)
        
        # Create red box outlines for query, key, and value tables
        query_outline = SurroundingRectangle(query_table, buff=0.1, color=RED)
        key_outline = SurroundingRectangle(key_table, buff=0.1, color=RED)
        value_outline = SurroundingRectangle(value_table, buff=0.1, color=RED)
        
        # Create a big X in the center of the scene
        big_x = Text("X", font_size=75, color=WHITE)
        big_x.move_to(ORIGIN)
        
        # Animation
        self.play(Create(embedding_table))
        self.play(Create(query_table), Create(key_table), Create(value_table))
        self.play(Create(query_outline), Create(key_outline), Create(value_outline))
        self.play(Create(big_x))
        
        self.wait(10)
        self.wait(15)  # Added wait time