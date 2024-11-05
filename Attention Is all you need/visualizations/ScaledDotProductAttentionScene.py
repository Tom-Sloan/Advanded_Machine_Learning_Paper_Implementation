from manim import *
from visualizations.utils import BlockWithText, ConnectedBlocks, ScaledDotProductAttention
class ScaledDotProductAttentionScene(Scene):

    def construct(self):
        attention = ScaledDotProductAttention()

        # Create MatMul block above attention
        matmul_block = BlockWithText("MatMul", width=2, height=0.6, color=BLUE)
        matmul_block.scale(0.8)
        matmul_block.next_to(attention.get_top(), UP + RIGHT * 2, buff=0.5*0.8)

        # Create arrow from MatMul block to attention
        matmul_arrow = Arrow(
            end=matmul_block.get_bottom() + DOWN * 0.1,
            start=attention.get_top(),
            buff=0.1,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.1
        )

        # Create Q, K, V text at the bottom
        self.qkv_text = VGroup(
            Text("Q", font_size=24),
            Text("K", font_size=24),
            Text("V", font_size=24)
        ).arrange(RIGHT, buff=1)
        self.qkv_text.next_to(attention.get_bottom(), DOWN * 4)


        # Create arrows from text to bottom of attention
        self.arrows = VGroup()
        for text in self.qkv_text[:2]:
            arrow = Arrow(
                start=text.get_top(),
                end=attention.get_bottom(),
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.1
            )
            self.arrows.add(arrow)

        # Add value arrow
        value_arrow = Arrow(
            start=self.qkv_text[2].get_top(),
            end=matmul_block.get_bottom(),
            buff=0.1,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.03
        )

        # add arrow above matmul block
        matmul_arrow_finish = Arrow(
            start=matmul_block.get_top(),
            end=matmul_block.get_top() + UP * 1,
            buff=0.1,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.1
        )

        # add value arrow to self.arrows
        self.arrows.add(value_arrow)
        self.arrows.add(matmul_arrow)
        self.arrows.add(matmul_arrow_finish)

        # add matul block to attention
        attention.add(matmul_block)
        


        self.play(
            Create(attention),
            Create(self.qkv_text),
            Create(self.arrows),
            run_time=2
        )
        self.wait(2)
        self.wait(15)  # Added wait time