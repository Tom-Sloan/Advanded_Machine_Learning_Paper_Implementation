from manim import *
from visualizations.utils import *

class LinearLayers2D(Scene):
    def construct(self):
        # Add title to the top left corner
        title = Text("Multi-Head\nAttention Block", font_size=36)
        title.to_corner(UL, buff=0.5)

        # Define consistent arrow style
        ARROW_STROKE_WIDTH = 5
        ARROW_MAX_TIP_LENGTH_RATIO = 0.1
        ARROW_BUFF = 0.1

        # Create three blocks with "Linear" text
        block1 = BlockWithText("Linear", width=2, height=0.6, color=BLUE)
        block2 = BlockWithText("Linear", width=2, height=0.6, color=BLUE)
        block3 = BlockWithText("Linear", width=2, height=0.6, color=BLUE)

        # Arrange blocks horizontally and position them 1 up from the bottom
        blocks = VGroup(block1, block2, block3).arrange(RIGHT, buff=1)
        blocks.to_edge(DOWN, buff=1.5)

        # Create Text objects for Q, K, V
        text_q = Text("Q", font_size=24).next_to(block1, DOWN, buff=1)
        text_k = Text("K", font_size=24).next_to(block2, DOWN, buff=1)
        text_v = Text("V", font_size=24).next_to(block3, DOWN, buff=1)

        # Create arrows from text to blocks with consistent style
        arrow_q = Arrow(text_q.get_top(), block1.get_bottom(), buff=ARROW_BUFF, stroke_width=ARROW_STROKE_WIDTH, max_tip_length_to_length_ratio=ARROW_MAX_TIP_LENGTH_RATIO)
        arrow_k = Arrow(text_k.get_top(), block2.get_bottom(), buff=ARROW_BUFF, stroke_width=ARROW_STROKE_WIDTH, max_tip_length_to_length_ratio=ARROW_MAX_TIP_LENGTH_RATIO)
        arrow_v = Arrow(text_v.get_top(), block3.get_bottom(), buff=ARROW_BUFF, stroke_width=ARROW_STROKE_WIDTH, max_tip_length_to_length_ratio=ARROW_MAX_TIP_LENGTH_RATIO)

        # Create the "Scaled Dot Product Attention" block using ConnectedBlocks
        block_names = ["Scaled Dot Product Attention", "Concat", "Linear"]
        block_types = {
            "Scaled Dot Product Attention": {"width": 5, "height": 0.8, "color": GREEN},
            "Concat": {"width": 3, "height": 0.6, "color": YELLOW},
            "Linear": {"width": 3, "height": 0.6, "color": BLUE}
        }
        attention_blocks = ConnectedBlocks(block_names, block_types, vertical_buff=1.4, scale=0.8, has_outline=False)
        attention_blocks.next_to(blocks, UP, buff=1)  # Position it 0.5 units above the blocks

        # Create dashed rectangles for the attention blocks
        dashed_rects = VGroup()
        for i in range(5):
            rect = RoundedRectangle(
                width=attention_blocks.get_block_at_index(0).width,
                height=attention_blocks.get_block_at_index(0).height,
                corner_radius=0.2,
                color=GREEN,
                stroke_width=2
            )
            rect.move_to(attention_blocks.get_block_at_index(0).get_center())
            rect.shift(UP * 0.15 * i + RIGHT * 0.15 * i)
            dashed_rect = DashedVMobject(rect, num_dashes=1, dashed_ratio=0.5, dash_offset=0.95)
            dashed_rects.add(dashed_rect)

        # Create dashed rectangles for block1, block2, and block3
        linear_dashed_rects = VGroup()
        for block in [block1, block2, block3]:
            for i in range(5):
                rect = RoundedRectangle(
                    width=block.width,
                    height=block.height,
                    corner_radius=0.2,
                    color=BLUE,
                    stroke_width=2
                )
                rect.move_to(block.get_center())
                rect.shift(UP * 0.1 * i + RIGHT * 0.1 * i)
                dashed_rect = DashedVMobject(rect, num_dashes=1, dashed_ratio=0.5, dash_offset=0.95)
                linear_dashed_rects.add(dashed_rect)

        number_of_heads = Text("h", font_size=18)
        number_of_heads.move_to(dashed_rects[1].get_right() + RIGHT * 1 + DOWN * 0.2)

        # Adding curly brace
        brace = Brace(
            VGroup(dashed_rects[4], dashed_rects[0]),
            direction=RIGHT,
            buff=0.1,
            color=RED
        )
        brace.set_stroke(width=ARROW_STROKE_WIDTH)
        brace.rotate(-30 * DEGREES)
        brace.shift(DOWN * 0.2)

        # Create arrows from Linear blocks to the Attention block with consistent style
        arrow1 = Arrow(block1.get_top(), attention_blocks.get_bottom(), buff=ARROW_BUFF*2, stroke_width=ARROW_STROKE_WIDTH, max_tip_length_to_length_ratio=0.05)
        arrow2 = Arrow(block2.get_top(), attention_blocks.get_bottom(), buff=ARROW_BUFF, stroke_width=ARROW_STROKE_WIDTH, max_tip_length_to_length_ratio=ARROW_MAX_TIP_LENGTH_RATIO)
        arrow3 = Arrow(block3.get_top(), attention_blocks.get_bottom(), buff=ARROW_BUFF*2, stroke_width=ARROW_STROKE_WIDTH, max_tip_length_to_length_ratio=0.05)

        arrow4 = Arrow(attention_blocks.get_top(), attention_blocks.get_top() + UP * 1, buff=ARROW_BUFF*2, stroke_width=ARROW_STROKE_WIDTH, max_tip_length_to_length_ratio=0.05)
        # Group all elements
        all_elements = VGroup(blocks, text_q, text_k, text_v, arrow_q, arrow_k, arrow_v, dashed_rects, number_of_heads, linear_dashed_rects)
        # Add new elements to the all_elements group
        all_elements.add(attention_blocks, arrow1, arrow2, arrow3, arrow4, brace, title)

        # Animate the creation of all elements
        self.play(Create(all_elements))
        self.wait(1)
        self.wait(15)  # Added wait time