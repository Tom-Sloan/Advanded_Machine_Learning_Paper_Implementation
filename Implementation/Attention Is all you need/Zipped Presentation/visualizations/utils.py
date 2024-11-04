from manim import *
import random
from manim.utils.color import color_to_rgb

class BlockWithText(VGroup):
    def __init__(self, text, width=3, height=1, color=BLUE, stroke_opacity=1, font_size=18, corner_radius=0.2, **kwargs):
        super().__init__(**kwargs)
        
        # Create a rectangle block
        self.block = RoundedRectangle(width=width, height=height, color=color, stroke_opacity=stroke_opacity, corner_radius=corner_radius)
        
        # Create a text label
        self.label = Text(text, font_size=font_size).move_to(self.block.get_center())
        
        # Add both the block and the label to the VGroup
        self.add(self.block, self.label)
    
    def get_rectangle(self):
        return self.block
    
    def get_label(self):
        return self.label
    
    def get_right(self):
        return self.block.get_right()
    def get_bottom(self):
        return self.block.get_bottom()

class ConnectedBlocks(VGroup):
    def __init__(self, block_names, block_types, vertical_buff=0.5, scale=1, has_outline=True, **kwargs):
        super().__init__(**kwargs)
        
        self.blocks = []
        for block_name in block_names:
            block_type = block_types.get(block_name, {})
            block = BlockWithText(
                block_name,
                width=block_type.get("width", 3),
                height=block_type.get("height", 1),
                color=block_type.get("color", BLUE)
            )
            self.blocks.append(block)
        
        # Position blocks vertically
        for i in range(1, len(self.blocks)):
            self.blocks[i].next_to(self.blocks[i-1], UP, buff=vertical_buff)
        
        # Create arrows connecting blocks
        self.arrows = []
        for i in range(len(self.blocks) - 1):
            arrow = Arrow(self.blocks[i].get_top(), self.blocks[i+1].get_bottom(), buff=0.05)
            self.arrows.append(arrow)
        
        for i, block_name in enumerate(block_names):
            if block_types[block_name].get("residual", False):
                start_point = self.blocks[i].get_right() + RIGHT * 0.1
                right_point = self.blocks[i-1].get_right() + RIGHT * self.blocks[i].get_height()
                bottom_point = self.blocks[i-1].get_bottom() + DOWN * 0.2
                mid_point = right_point + DOWN * 0.5
                path = VGroup(
                    Line(mid_point, bottom_point, color=WHITE, stroke_width=3),
                    Line(right_point, mid_point, color=WHITE, stroke_width=3),
                    Arrow(right_point, start_point, color=WHITE, buff=0, stroke_width=3),
                )
                self.blocks[i].add(path)

        # Add all elements to the group
        self.add(*self.blocks, *self.arrows)
        
        # Create a surrounding rectangle
        if has_outline:
            self.surrounding_rect = SurroundingRectangle(self, buff=0.2, color=WHITE)
            self.dashed_surrounding_rect = DashedVMobject(self.surrounding_rect, dashed_ratio=0.4, dash_offset=-0.2)
            self.add(self.dashed_surrounding_rect)
        
        # Apply scaling
        self.scale(scale)
        
        # Center vertically
        self.center()

    def get_top(self):
        return self.blocks[-1].get_top()

    def get_bottom(self):
        return self.blocks[0].get_bottom()

    def get_num_blocks(self):
        return len(self.blocks)

    def get_block_at_index(self, index):
        try:
            return self.blocks[index]
        except IndexError:
            print(f"Index {index} is out of range. The ConnectedBlocks has {len(self.blocks)} blocks.")
            return None
class EmbeddingTable(VGroup):
    def __init__(self, rows, columns, color=YELLOW, title="Embedding Table", scale=1.0, indexed=False, tril=False, black_squares=None, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows
        self.columns = columns
        self.color = color
        self.title = title
        self.scale = scale
        self.indexed = indexed
        self.tril = tril
        self.buff = 0.2*self.scale
        self.black_squares = black_squares if black_squares is not None else []
        self.create_table()
        self.add_label()

    def create_table(self):
        self.table_group = VGroup()
        all_squares = []
        all_numbers = []
        count = 1
        for i in range(self.rows):
            row = VGroup(*[Square(side_length=0.5*self.scale, fill_opacity=0.8, fill_color=self.color, stroke_color=self.color) for _ in range(self.columns)])
            row.arrange(RIGHT, buff=0.2*self.scale)
            all_squares.extend(row)
            if self.indexed:
                row_numbers = VGroup(*[Text(f"{count}", font_size=16*self.scale, color=self.get_contrasting_color(self.color)).move_to(square) for square in row])
                count += self.columns
            else:
                row_numbers = VGroup()
                for j in range(self.columns):
                    if self.tril and j > i:
                        number = Text("-∞", font_size=20*self.scale, color=self.get_contrasting_color(self.color)).move_to(row[j])
                    else:
                        number = Text(f"{random.randint(-9, 9)}", font_size=16*self.scale, color=self.get_contrasting_color(self.color)).move_to(row[j])
                    row_numbers.add(number)
            all_numbers.extend(row_numbers)
            self.table_group.add(VGroup(row, row_numbers))
        self.table_group.arrange(DOWN, buff=0.3*self.scale)
        
        # Set specified squares to be black
        if self.black_squares:
            for index in self.black_squares:
                if 0 <= index < len(all_squares):
                    all_squares[index].set_opacity(0)
                    all_numbers[index].set_opacity(0)
        
        self.add(self.table_group)

    def add_label(self):
        self.label = Text(self.title, font_size=36*self.scale).next_to(self, UP, buff=0.5*self.scale)
        self.add(self.label)

    def fade_out_title(self):
        return FadeOut(self.label)
    
    def get_label(self):
        return self.label

    def get_table_center(self):
        return self.table_group.get_center()

    def get_contrasting_color(self, bg_color):
        # Convert ManimColor to RGB values (0 to 1 scale)
        r, g, b = color_to_rgb(bg_color)
        # Calculate luminance
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return BLACK if luminance > 0.5 else WHITE
    
class ScaledDotProductAttention(VGroup):
    def __init__(self, scale=0.8, **kwargs):
        super().__init__(**kwargs)
        
        # Define the blocks for the attention mechanism
        block_names = ["MatMul", "Scale", "Mask (opt)", "Softmax"]
        block_types = {
            "MatMul": {"width": 2, "height": 0.6, "color": BLUE},
            "Scale": {"width": 2, "height": 0.6, "color": GREEN},
            "Mask (opt)": {"width": 2, "height": 0.6, "color": YELLOW},
            "Softmax": {"width": 2, "height": 0.6, "color": RED},
        }
        
        # Create the ConnectedBlocks
        self.attention_blocks = ConnectedBlocks(block_names, block_types, vertical_buff=0.3, scale=scale, has_outline=False)

        # Add all elements to the group
        self.add(self.attention_blocks)
    
    def get_top(self):
        return self.attention_blocks.get_top()
    
    def get_bottom(self):
        return self.attention_blocks.get_bottom()
    
    def get_left(self):
        return self.attention_blocks.get_block_at_index(0).get_left()
    
    def get_right(self):
        return self.attention_blocks.get_block_at_index(0).get_right()