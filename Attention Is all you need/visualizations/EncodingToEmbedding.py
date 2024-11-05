from manim import *
from visualizations.utils import *
import random

class EncodingToEmbedding(Scene):
    def construct(self):
        # Define the encoding vector
        encoding = [0, 1, 2, 3]
        N = 5  # Embedding dimension
        words = ["The", "cat", "sat", "on", "the", "mat"]

        # Create the encoding vector visualization as a 1-column embedding table
        encoding_table = EmbeddingTable(rows=6, columns=1, color=BLUE, title="Encoding Vector", scale=0.8, indexed=True)
        
        # Create the embedding table
        embedding_table = EmbeddingTable(rows=6, columns=5, color=YELLOW, title="Embedding Table", scale=0.8)
        
        # Position the encoding table and embedding table
        encoding_table.move_to(LEFT * 3)
        embedding_table.move_to(RIGHT * 2)
        
        # Align the top edges of encoding_table and embedding_table
        embedding_table.align_to(encoding_table, UP)

        # Add labels for encoding_table rows
        encoding_row_labels = VGroup()
        for i, word in enumerate(words):
            label = Text(word, font_size=24)
            label.next_to(encoding_table.table_group[i], LEFT, buff=0.5)
            label.align_to(encoding_table.table_group[i].get_center(), UP)
            encoding_row_labels.add(label)
        
        # Add dot dot dot below encoding_table and embedding_table
        dots_encoding = VGroup(*[Text(".", font_size=36) for _ in range(3)])
        dots_encoding.arrange(DOWN, buff=0.1).next_to(encoding_table, DOWN, buff=0.3)
        
        dots_embedding = VGroup(*[Text(".", font_size=36) for _ in range(3)])
        dots_embedding.arrange(DOWN, buff=0.1).next_to(embedding_table, DOWN, buff=0.3)
        
        # Group all elements
        all_elements = VGroup(encoding_table, embedding_table,
                              dots_encoding, dots_embedding, encoding_row_labels)
        
        # Center the entire scene
        all_elements.move_to(ORIGIN)
        
        # Animation
        self.play( Create(encoding_table), Create(encoding_row_labels))
        self.play( Create(embedding_table))
        
        # Show the mapping with arrows (increased speed)
        arrows = VGroup(*[Arrow(enc_row.get_right(), emb_row.get_left(), buff=0.2, color=YELLOW) 
                          for enc_row, emb_row in zip(encoding_table.table_group, embedding_table.table_group)])
        self.play(Create(arrows), run_time=1)  # Create all arrows at once with reduced run_time
        
        # Add the dot dot dot animations
        self.play(
            Write(dots_encoding),
            Write(dots_embedding),
            run_time=1
        )
        
        self.wait(2)
        self.wait(15)  # Added wait time