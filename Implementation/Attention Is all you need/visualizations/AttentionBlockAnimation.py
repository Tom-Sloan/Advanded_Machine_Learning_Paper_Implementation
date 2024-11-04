from manim import *
from visualizations.utils import *

class AttentionBlockAnimation(Scene):
    def construct(self):
        # Title
        title = Text("Self-Attention Block", font_size=36)
        title.to_edge(UP, buff=0.5)

        # Input text
        text = "The cat sat on the mat"
        words = text.split()

        # Create horizontal text
        horizontal_text = VGroup(*[Text(word, font_size=24) for word in words])
        horizontal_text.arrange(RIGHT, buff=0.3)
        horizontal_text.next_to(title, DOWN, buff=0.5)

        # Create vertical text
        vertical_text = VGroup(*[Text(word, font_size=24) for word in words])
        vertical_text.arrange(DOWN, buff=0.3)
        vertical_text.to_edge(LEFT, buff=0.5)

        # Create attention block using EmbeddingTable
        num_words = len(words)
        attention_block = EmbeddingTable(rows=num_words, columns=num_words, color=BLUE, title="", scale=1, indexed=False)
        attention_block.next_to(vertical_text, RIGHT, buff=1)
        attention_block.next_to(horizontal_text, DOWN, buff=1)

        # Add explanation text
        explanation = Text("Animation shown sequentially, all in parallel in reality", font_size=24)
        explanation.to_edge(DOWN, buff=0.5)

        # Animation
        self.play(Write(title), Write(horizontal_text), Write(vertical_text), Write(explanation))
        self.play(Create(attention_block))
        
        # Highlight attention for each word pair
        horizontal_arrow = None
        vertical_arrow = None
        for i in range(num_words):
            for j in range(num_words):
                highlight = attention_block.table_group[i][0][j].copy().set_fill(YELLOW, opacity=0.8)
                
                # Create or move arrows
                if horizontal_arrow is None:
                    horizontal_arrow = Arrow(start=horizontal_text[j].get_bottom(), end=highlight.get_top(), buff=0.1, color=YELLOW)
                    vertical_arrow = Arrow(start=vertical_text[i].get_right(), end=highlight.get_left(), buff=0.1, color=YELLOW)
                    arrow_creation = Create(horizontal_arrow), Create(vertical_arrow)
                else:
                    arrow_creation = (
                        horizontal_arrow.animate.put_start_and_end_on(horizontal_text[j].get_bottom(), highlight.get_top()),
                        vertical_arrow.animate.put_start_and_end_on(vertical_text[i].get_right(), highlight.get_left())
                    )
                
                self.play(
                    FadeIn(highlight),
                    horizontal_text[j].animate.set_color(YELLOW),
                    vertical_text[i].animate.set_color(YELLOW),
                    *arrow_creation,
                    run_time=0.5
                )
                self.play(
                    FadeOut(highlight),
                    horizontal_text[j].animate.set_color(WHITE),
                    vertical_text[i].animate.set_color(WHITE),
                    run_time=0.5
                )


        self.wait(15)