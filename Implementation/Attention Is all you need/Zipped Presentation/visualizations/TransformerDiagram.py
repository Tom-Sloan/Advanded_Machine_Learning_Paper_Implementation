from manim import *
from visualizations.utils import ConnectedBlocks, BlockWithText
class TransformerDiagram(Scene):
    def construct(self):

        # Colors for the different blocks
        attn_color = BLUE
        ff_color = GREEN
        add_norm_color = YELLOW
        embedding_color = PINK
        pos_encoding_color = ORANGE
        output_color = GREY
        left_right_shift = 2

        block_types = {
            "Multi-Head Attention": {
                "color": attn_color,
                "height": 0.5,
                "width": 3.2
            },
            "Feed Forward": {
                "color": ff_color,
                "height": 0.5,
                "width": 3.2
            },
            "Add & Norm": {
                "color": add_norm_color,
                "height": 0.5,
                "width": 1.7,
                "residual": True
            },
            "Masked Multi-Head Attention": {
                "color": attn_color,
                "height": 0.5,
                "width": 3.2
            },
            "Linear": {
                "color": attn_color,
                "height": 0.5,
                "width": 3.2
            },
            "Softmax": {
                "color": attn_color,
                "height": 0.5,
                "width": 3.2
            }
        }

        # Create blocks representing the different components
        # Encoder Blocks
        encoder_blocks = ConnectedBlocks([
            "Multi-Head Attention",
            "Add & Norm",
            "Feed Forward",
            "Add & Norm"
        ], block_types, scale=0.6)
        encoder_blocks.shift(LEFT * left_right_shift + DOWN * 0.5 + 0.15*RIGHT)
        
        # Decoder Blocks
        decoder_blocks = ConnectedBlocks([
            "Masked Multi-Head Attention",
            "Add & Norm",
            "Multi-Head Attention",
            "Add & Norm",
            "Feed Forward",
            "Add & Norm"
        ], block_types, scale=0.6)
        decoder_blocks.shift(RIGHT * left_right_shift + 0.15*RIGHT)
        
        # Adding text labels
        encoder_text = Text("Encoder", font_size=24).move_to(LEFT * left_right_shift*5/2 + UP * 3)
        decoder_text = Text("Decoder", font_size=24).move_to(RIGHT * left_right_shift*5/2 + UP * 3)
        input_text = Text("Input", font_size=20).move_to(LEFT * left_right_shift + DOWN * 3.5)
        output_text = Text("Output", font_size=20).move_to(RIGHT * left_right_shift + DOWN * 3.5)

        # Embeddings
        input_embedding = BlockWithText("Input Embedding", font_size=14, height=0.3, width=2, color=embedding_color).next_to(input_text, UP, buff=0.5)
        output_embedding = BlockWithText("Output Embedding", font_size=14, height=0.3, width=2, color=embedding_color).next_to(output_text, UP, buff=0.5)
        
        # Positional Encoding
        position_encoder = BlockWithText("Positional\nEncoding", font_size=14, height=0.6, width=1.2, color=pos_encoding_color).next_to(input_embedding, LEFT * 0.7, buff=0.5)
        position_decoder = BlockWithText("Positional\nEncoding", font_size=14, height=0.6, width=1.2, color=pos_encoding_color).next_to(output_embedding, RIGHT * 0.7, buff=0.5)
        position_encoder.shift(UP * 0.5)
        position_decoder.shift(UP * 0.5)

        # Addition circle
        addition_circle_encoder = Circle(radius=0.1, color=RED, fill_opacity=1).next_to(input_embedding, UP*1, buff=0.5)
        addition_circle_decoder = Circle(radius=0.1, color=RED, fill_opacity=1).next_to(output_embedding, UP*1, buff=0.5)
        
        # Add "+" text to the center of the circles
        plus_text_encoder = Text("+", font_size=20, color=WHITE).move_to(addition_circle_encoder.get_center())
        plus_text_decoder = Text("+", font_size=20, color=WHITE).move_to(addition_circle_decoder.get_center())
        
        # Group the circle and text for each addition
        addition_encoder = VGroup(addition_circle_encoder, plus_text_encoder)
        addition_decoder = VGroup(addition_circle_decoder, plus_text_decoder)
        
        # Add arrows from text to blocks
        # Define consistent stroke width and max_tip_length_to_length_ratio for all arrows
        arrow_stroke_width = 3
        arrow_tip_ratio = 0.1

        input_arrow = Arrow(start=input_text.get_top(), end=input_embedding.get_bottom(), buff=0.1, stroke_width=arrow_stroke_width)
        output_arrow = Arrow(start=output_text.get_top(), end=output_embedding.get_bottom(), buff=0.1, stroke_width=arrow_stroke_width)
        
        # Shift the encoder and decoder blocks up
        encoder_blocks.shift(UP * 0.5)
        decoder_blocks.shift(UP * 0.5)

        # Add arrows from positional encoding to addition circle
        pos_to_add_encoder = Arrow(start=position_encoder.get_right(), end=addition_encoder.get_left(), buff=0.1, stroke_width=arrow_stroke_width, max_tip_length_to_length_ratio=arrow_tip_ratio)
        pos_to_add_decoder = Arrow(start=position_decoder.get_left(), end=addition_decoder.get_right(), buff=0.1, stroke_width=arrow_stroke_width, max_tip_length_to_length_ratio=arrow_tip_ratio)

        # Add arrows from embedding to addition circle
        emb_to_add_encoder = Arrow(start=input_embedding.get_top(), end=addition_encoder.get_bottom(), buff=0, stroke_width=arrow_stroke_width)
        emb_to_add_decoder = Arrow(start=output_embedding.get_top(), end=addition_decoder.get_bottom(), buff=0, stroke_width=arrow_stroke_width)

        # Add arrows from addition circle to bottom of ConnectedBlock
        add_to_block_encoder = Arrow(start=addition_encoder.get_top(), end=encoder_blocks.get_bottom(), buff=0.1, stroke_width=arrow_stroke_width)
        add_to_block_decoder = Arrow(start=addition_decoder.get_top(), end=decoder_blocks.get_bottom(), buff=0.1, stroke_width=arrow_stroke_width)

        # Add the softmax and linear using connected blocks
        softmax_linear = ConnectedBlocks(["Softmax", "Linear"], block_types, has_outline=False, scale=0.6)
        softmax_linear.next_to(decoder_blocks.get_top(), UP*1.1, buff=0.2)
        softmax_linear.shift(LEFT * 0.1)
        # add arrow from top of decoder blocks to softmax linear
        softmax_linear_arrow = Arrow(start=decoder_blocks.get_top() + LEFT * 0.1, end=softmax_linear.get_bottom(), buff=0, stroke_width=arrow_stroke_width)

        # Create a rounded arrow from encoder to decoder
        start_point = encoder_blocks.get_top() + UP * 0.1
        end_point = decoder_blocks.get_block_at_index(2).get_bottom()
        
        # Define control points for the curved path
        control1 = start_point + UP * 0.5
        control2 = start_point + UP * 0.5 + RIGHT * 2
        control3 = end_point + DOWN * 0.1 + LEFT * 2
        control4 = end_point + DOWN * 0.1 + LEFT * 0.2  # Slightly below the target block
        
        # Create a custom path
        custom_path = VMobject()
        custom_path.set_points_smoothly([start_point, control1, control2, control3, control4])
        
        # Create the arrow
        encoder_to_decoder_arrow = Arrow(
            start=custom_path.get_start(),
            end=end_point,
            path_arc=0,  # This ensures a straight arrow tip
            buff=0,
            stroke_width=arrow_stroke_width,
            tip_length=0.05
        )
        
        # Set the path of the arrow to follow the custom path
        encoder_to_decoder_arrow.put_start_and_end_on(custom_path.get_start(), custom_path.get_end())
        encoder_to_decoder_arrow.set_points(custom_path.points)

        # Create a rounded arrow from encoder to decoder
        start_point = softmax_linear.get_top() + UP * 0.1
        end_point = output_text.get_right() + RIGHT * 0.1
        
        # Define control points for the curved path
        control1 = start_point
        control2 = start_point + UP*0.3
        control3 = start_point + RIGHT * 2
        control4 = end_point + RIGHT * 3
        control5 = end_point  
        
        # Create a custom path
        custom_path = VMobject()
        custom_path.set_points_as_corners([start_point, control1, control2, control3, control4, control5])
        
        # Create the arrow
        output_arrow = Arrow(
            start=custom_path.get_start(),
            end=end_point + RIGHT * 0.5 - DOWN * 0.5,
            path_arc=1,  # This ensures a straight arrow tip
            buff=0,
            stroke_width=arrow_stroke_width,
            tip_length=0.1
        )
        
        # Set the path of the arrow to follow the custom path
        output_arrow.put_start_and_end_on(custom_path.get_start(), custom_path.get_end())
        output_arrow.points = custom_path.points

        # Make a dotted line vertically to divid the scene in half
        dotted_line = DashedVMobject(Line(start=UP * 4, end=DOWN * 4, color=GRAY), dashed_ratio=0.2)

        # Add everything to the scene
        self.play(
            Create(encoder_blocks),
            Create(decoder_blocks),
            Create(input_embedding),
            Create(output_embedding),
            Create(position_encoder),
            Create(position_decoder),
            Create(addition_encoder),
            Create(addition_decoder),
            Create(pos_to_add_encoder),
            Create(pos_to_add_decoder),
            Create(emb_to_add_encoder),
            Create(emb_to_add_decoder),
            Create(add_to_block_encoder),
            Create(add_to_block_decoder),
            Create(encoder_to_decoder_arrow),
            Create(softmax_linear),
            Create(softmax_linear_arrow),
            Create(input_arrow),
            Create(output_arrow),
            Write(input_text),
            Write(output_text),
            Write(encoder_text), 
            Write(decoder_text),
            Write(dotted_line)
        )

        self.wait(2)
        self.wait(15)  # Added wait time