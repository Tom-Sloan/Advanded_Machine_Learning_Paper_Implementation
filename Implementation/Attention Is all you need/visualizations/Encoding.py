from manim import *
import string

class Encoding(Scene):
    def construct(self):
        # Define the encoding mapping (word-level embeddings)
        encoding_map = {
            'The': 1, 'cat': 2, 'sat': 3, 'on': 4, 'the': 5, 'mat': 6,
            '<PAD>': 0, '<UNK>': 7
        }

        # Example sentence to encode
        sentence = "The cat sat on the mat"

        # Create a container for all elements
        container = VGroup()

        # Display the "Sentence:" title
        sentence_title = Text("Sentence:")
        container.add(sentence_title)

        # Prepare the words of the sentence
        words = sentence.split()
        word_mobs = [Text(word) for word in words]

        # Arrange words next to the "Sentence:" title with a gap, aligning their bottoms
        word_group = VGroup(*word_mobs).arrange(RIGHT, buff=0.4, aligned_edge=DOWN)
        word_group.next_to(sentence_title, RIGHT, buff=1.0, aligned_edge=DOWN)
        container.add(word_group)

        # Create the "Encoding:" title
        encoding_title = Text("Encoding:")
        encoding_title.next_to(sentence_title, DOWN, buff=1).align_to(sentence_title, LEFT)
        container.add(encoding_title)

        # Create the encoded numbers and place them vertically below the words
        encoded_group = VGroup()
        arrows = VGroup()
        for word_mob in word_mobs:
            word = word_mob.text
            encoded_value = encoding_map.get(word, encoding_map['<UNK>'])
            encoded_text = Text(str(encoded_value))
            box = SurroundingRectangle(encoded_text, color=WHITE)
            encoded_mob = VGroup(encoded_text, box)
            encoded_mob.next_to(word_mob, DOWN, buff=1.0)
            encoded_group.add(encoded_mob)
            
            arrow = Arrow(start=word_mob.get_bottom(), end=box.get_top(), buff=0.1, color=RED)
            arrows.add(arrow)

        # Align the bottom of the encoded numbers with the bottom of the encoding title
        encoded_group.align_to(encoding_title, DOWN)
        container.add(encoded_group, arrows)

        # Create the "Decoding:" title
        decoding_title = Text("Decoding:")
        decoding_title.next_to(encoding_title, DOWN, buff=1).align_to(encoding_title, LEFT)
        container.add(decoding_title)

        # Create the decoded words
        decoded_group = VGroup()
        decoding_arrows = VGroup()
        for word_mob, encoded_mob in zip(word_mobs, encoded_group):
            word = word_mob.text
            decoded_text = Text(word)
            decoded_text.move_to(word_mob.get_center())
            decoded_text.align_to(decoding_title, DOWN)
            decoded_group.add(decoded_text)
            
            arrow = Arrow(start=encoded_mob[1].get_bottom(), end=decoded_text.get_top(), buff=0.1, color=GREEN)
            decoding_arrows.add(arrow)

        container.add(decoded_group, decoding_arrows)

        # Center the entire container
        container.center()

        # Animate the elements
        self.play(Write(sentence_title))
        self.wait(0.5)
        self.play(Write(word_group, run_time=0.3))
        self.wait(0.2)
        self.play(Write(encoding_title))
        self.wait(0.5)
        self.play(Write(encoded_group), Create(arrows))
        self.wait(0.5)
        self.play(Write(decoding_title))
        self.wait(0.5)
        self.play(Write(decoded_group), Create(decoding_arrows))
        self.wait(2)
        self.wait(15)  # Added wait time