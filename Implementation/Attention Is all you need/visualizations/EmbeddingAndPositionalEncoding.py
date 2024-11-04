import random
from manim import *
from manim.utils.color import color_to_rgb

class EmbeddingAndPositionalEncoding(Scene):
    def construct(self):
        max_width = config.frame_width
        N, num_rows = 4, 6
        words = ["The", "cat", "sat", "on", "the", "mat"]

        # Create and display initial tables
        tables, labels = self.create_and_display_initial_tables(N, num_rows, words)

        # Scale down and shift tables, fade out labels
        self.scale_and_shift_tables(tables, labels, max_width)

        # Create and display combined table with arrows
        self.create_and_display_combined_table_with_arrows(tables, N, num_rows)

        self.wait(2)
        self.wait(15)  # Added wait time

    def create_and_display_initial_tables(self, N, num_rows, words):
        tables = [self.create_embedding_table(N, num_rows, words) for _ in range(2)]
        labels = ["Embedding Table", "Positional Encoding Table"]
        label_texts = VGroup()

        for i, (table, label) in enumerate(zip(tables, labels)):
            direction = LEFT if i == 0 else RIGHT
            centered_group = self.center_table(table, direction * (config.frame_width / 4))
            self.adjust_word_positions(table, centered_group)
            
            label_text = Text(label, font_size=36).next_to(centered_group, UP, buff=0.5)
            label_texts.add(label_text)
            
            self.play(Write(label_text), Create(table))

        return tables, label_texts

    def scale_and_shift_tables(self, tables, labels, max_width):
        left_shift = LEFT * (max_width / 4 - tables[0].width / 2)
        right_shift = RIGHT * (max_width * 3 / 10 - tables[0].width / 2)
        self.play(
            tables[0].animate.scale(0.7).shift(left_shift),
            tables[1].animate.scale(0.7).shift(right_shift),
            *[FadeOut(label) for label in labels],
            run_time=1.5
        )
    def get_contrasting_color(self, bg_color):
        # Convert ManimColor to RGB values (0 to 1 scale)
        r, g, b = color_to_rgb(bg_color)
        # Calculate luminance
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return BLACK if luminance > 0.5 else WHITE
    
    def create_and_display_combined_table_with_arrows(self, tables, N, num_rows):
        combined_table = self.create_combined_table(tables[0], tables[1], N, num_rows)
        
        combined_label = Text("Combined Table", font_size=36)
        combined_label.next_to(combined_table, UP, buff=0.5)
        combined_label.move_to(combined_table.get_top() + UP * 0.5)
        self.play(Write(combined_label))

        for i in range(num_rows):
            if i < 3:
                # Create and display the block with arrows
                self.create_block_with_arrows(tables, combined_table, i)
            else:
                # Create the remaining blocks without arrows
                self.play(Create(combined_table[i]), run_time=0.5)

    def create_block_with_arrows(self, tables, combined_table, i):
        emb_num = tables[0][i][3][0]
        pos_num = tables[1][i][3][0]
        combined_num = combined_table[i][3][0]

        emb_arrow = Arrow(emb_num.get_center(), combined_num.get_center(), buff=0.1, color=BLUE)
        pos_arrow = Arrow(pos_num.get_center(), combined_num.get_center(), buff=0.1, color=RED)

        self.play(
            Create(combined_table[i]),
            Create(emb_arrow),
            Create(pos_arrow),
            run_time=1
        )
        self.wait(0.5)
        self.play(FadeOut(emb_arrow), FadeOut(pos_arrow), run_time=0.5)

    def create_combined_table(self, embedding_table, positional_table, N, num_rows):
        combined_table = VGroup()
        max_word_width = max(Text(word, font_size=24).width for word in ["The", "cat", "sat", "on", "the", "mat"])
        
        for i in range(num_rows):
            # Create row squares
            row = VGroup(*[
                Square(side_length=0.5, fill_opacity=1, fill_color=GREEN, stroke_color=GREEN) 
                for _ in range(N)
            ])
            row.arrange(RIGHT, buff=0.2)
            
            # Create combined numbers
            combined_numbers = VGroup()
            for j in range(N):
                embed_num = embedding_table[i][3][j].text
                pos_num = positional_table[i][3][j].text
                combined_value = int(pos_num) + int(embed_num)
                combined_text = Text(f"{combined_value}", font_size=16, color=self.get_contrasting_color(GREEN)).move_to(row[j].get_center())
                combined_numbers.add(combined_text)

            # Create word text and container (use words from embedding table)
            word = embedding_table[i][1].text
            word_text = Text(word, font_size=24)
            word_container = Rectangle(width=max_word_width, height=word_text.height, fill_opacity=0, stroke_opacity=0)
            word_container.next_to(row, LEFT, buff=0.5)
            word_text.move_to(word_container.get_center())

            # Group all elements into a single row
            row_group = VGroup(word_container, word_text, row, combined_numbers)
            combined_table.add(row_group)
        
        combined_table.arrange(DOWN, buff=0.3)
        return combined_table
    
    def center_table(self, table, position):
        # Extract blocks and numbers for centering
        blocks = VGroup(*[group[2] for group in table])
        numbers = VGroup(*[group[3] for group in table])
        
        # Group blocks and numbers together
        centered_group = VGroup(blocks, numbers)
        centered_group.move_to(position)
        return centered_group

    def adjust_word_positions(self, table, centered_group):
        blocks = centered_group[0]  # The blocks are the first item in the centered_group
        for i, group in enumerate(table):
            if i < len(blocks):
                word_container, word = group[:2]
                word_container.next_to(blocks[i], LEFT, buff=0.5)
                word.move_to(word_container)
            else:
                print(f"Warning: Mismatch in table rows and blocks. Table row {i} has no corresponding block.")

    def create_embedding_table(self, N, num_rows, words):
        embedding_table = VGroup()
        max_word_width = max(Text(word, font_size=24).width for word in words)
        
        for i in range(num_rows):
            row = self.create_row(N, i, words, max_word_width)
            embedding_table.add(row)
        
        embedding_table.arrange(DOWN, buff=0.3)
        return embedding_table

    def create_row(self, N, i, words, max_word_width):
        squares = VGroup(*[Square(side_length=0.5, fill_opacity=1, fill_color=YELLOW, stroke_color=YELLOW) for _ in range(N)])
        squares.arrange(RIGHT, buff=0.2)
        
        numbers = VGroup(*[
            Text(f"{random.randint(-9, 9)}", font_size=16, color=self.get_contrasting_color(YELLOW)).move_to(square.get_center()) 
            for square in squares
        ])

        word = words[i] if i < len(words) else ""
        word_text = Text(word, font_size=24)
        word_container = Rectangle(width=max_word_width, height=word_text.height, fill_opacity=0, stroke_opacity=0)
        word_container.next_to(squares, LEFT, buff=0.5)
        word_text.move_to(word_container.get_center())

        return VGroup(word_container, word_text, squares, numbers)