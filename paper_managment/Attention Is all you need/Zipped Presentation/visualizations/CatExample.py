from manim import *
class CatExample(Scene):
    def construct(self):
        shift = 3
        # Create two text fields for input and output
        text1 = Text("Input", font_size=36)
        text2 = Text("Output", font_size=36)
        
        input_text = Text("The cat sat on the mat", font_size=36)
        output_text = "<.> It sounds like you're quoting a classic simple sentence!"
        
        
        # Move texts to bottom
        text1.to_edge(DOWN, buff=1).shift(shift*LEFT)
        text2.to_edge(DOWN, buff=1).shift(shift*RIGHT)
        
        # Create boxes with text for encoder and decoder
        encoder_box = Rectangle(height=1.5, width=3)
        encoder_text = Text("Encoder", font_size=30)
        encoder_group = VGroup(encoder_box, encoder_text).next_to(text1, UP, buff=2)
        
        decoder_box = Rectangle(height=2.2, width=3)
        decoder_text = Text("Decoder", font_size=30)
        decoder_group = VGroup(decoder_box, decoder_text).next_to(text2, UP, buff=2)
        
        # Create arrows
        arrow1 = Arrow(start=text1.get_top(), end=encoder_group.get_bottom(), buff=0.5)
        arrow2 = Arrow(start=text2.get_top(), end=decoder_group.get_bottom(), buff=0.5)
        arrow3 = Arrow(start=decoder_group.get_top()+ LEFT*0.5, end=decoder_group.get_top() + UP * 2 +LEFT*0.5, buff=0.2)
        arrow4 = Arrow(start=encoder_group.get_right(), end=decoder_group.get_left(), buff=0.5)
        
        # Position input_text below text1
        input_text.next_to(text1, DOWN, buff=0.5)
        
        # Create animations
        self.play(Write(text1), Write(text2), Create(encoder_group), Create(decoder_group))
        self.play(Create(arrow1), Create(arrow2), Create(arrow3), Create(arrow4))
        self.play(Write(input_text))
        self.wait(1)
        
        # Animate input_text moving from below text1 to above encoder_group
        self.play(input_text.animate.next_to(encoder_group, UP, buff=0.5))
        self.wait(2)
        

        # Animate output_text word by word
        output_words = output_text.split()
        output_texts = [Text(word, font_size=16) for word in output_words]
        
        # Create a group for output text with buffer
        output_text_group = VGroup()
        for i, word_text in enumerate(output_texts):
            if i == 0:
                output_text_group.add(word_text)
            else:
                word_text.next_to(output_text_group, RIGHT, buff=0.1)
                output_text_group.add(word_text)

        # Position the group below text2
        output_text_group.next_to(text2, DOWN, buff=0.5)

        # Animate each word appearing
        for i, word_text in enumerate(output_text_group):
            if i > 0:
                word_text.set_opacity(0)
            self.play(Write(word_text), run_time=0.1)

        # Move the entire group above the decoder
        # self.play(output_text_group.animate.next_to(decoder_group, UP, buff=0.5).shift(2.5*RIGHT))

        for word_text in output_text_group:
            self.play(word_text.animate.set_opacity(1), run_time=0.5)
            self.play(output_text_group.animate.next_to(text2, DOWN, buff=0.5), run_time=0.5)
            self.play(output_text_group.animate.next_to(decoder_group, UP, buff=0.5).shift(2.5*RIGHT))
        self.wait(2)
        self.wait(15)  # Added wait time
