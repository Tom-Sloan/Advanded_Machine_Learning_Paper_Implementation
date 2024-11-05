from manim import *
from visualizations.utils import ConnectedBlocks, BlockWithText

class FeedForwardNetwork(Scene):
    def construct(self):
        # Create title
        title = Text("Feed Forward Network", font_size=36)
        title.to_edge(UP, buff=0.5)

        # Define the blocks for the feed-forward network
        block_names = ["Linear", "ReLU", "Linear"]
        block_types = {
            "Linear": {"width": 2, "height": 0.6, "color": BLUE},
            "ReLU": {"width": 2, "height": 0.6, "color": GREEN},
        }
        
        # Create the ConnectedBlocks
        ffn_blocks = ConnectedBlocks(block_names, block_types, vertical_buff=0.3, scale=0.8, has_outline=False)
        
        # Top Arrow
        top_arrow = Arrow(start=ffn_blocks.get_top(), end=ffn_blocks.get_top() + UP * 1, color=WHITE)
        bottom_arrow = Arrow(end=ffn_blocks.get_bottom(), start=ffn_blocks.get_bottom() + DOWN * 1, color=WHITE)
        # Center the diagram
        ffn_blocks.move_to(ORIGIN)
        
        # Add everything to the scene
        self.play(Write(title), run_time=1)
        self.play(Create(ffn_blocks), Create(top_arrow), Create(bottom_arrow), run_time=2)
        self.wait(2)
        self.wait(15)  # Added wait time