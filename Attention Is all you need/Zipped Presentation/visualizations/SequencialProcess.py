from manim import *

config.pixel_height = 720  # Set the desired pixel height
config.pixel_width = 1280  # Set the desired pixel width

class SequentialProcessing(Scene):
    def construct(self):
        # Add title
        title = Text("Recurrent Neural Network", font_size=36)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Animation speed control
        node_spawn_time = 0.1
        arrow_spawn_time = 0.1
        box_move_time = 2  # Slower box movement
        fade_time = 0.5    # Time for fading arrows in/out
        node_width = 1    # Width of each node
        node_space = 2

        # Create the sentence
        sentence = "The cat sat on the mat"
        words = sentence.split()

        # Create nodes representing RNN steps with increased spacing and centered
        total_width = (len(words) - 1) * node_space  # Increased spacing between nodes

        start_x = -total_width / 2
        rnn_nodes = []
        for i in range(len(words)):
            circle = Circle(radius=node_width/2, color=YELLOW, stroke_width=6)
            text = Text(words[i], font_size=24).move_to(circle.get_center())
            node = VGroup(circle, text)
            node.shift(RIGHT * (start_x + i * node_space))
            rnn_nodes.append(node)

        # Create sequential arrows (straight arrows connecting nodes in order)
        sequential_arrows = []
        for i in range(len(rnn_nodes) - 1):
            
            arrow = Arrow(
                start=RIGHT * (start_x + i * node_space + node_width/4),
                end=RIGHT * (start_x + (i + 1) * node_space - node_width/4),
                color=YELLOW,
                stroke_width=6  # Set fixed thin stroke width
            )
            sequential_arrows.append(arrow)

        # Animate the sequential processing through each step
        for i, node in enumerate(rnn_nodes):
            # Draw the current RNN step node
            self.play(Create(node), run_time=node_spawn_time)
            # Draw the connecting arrow if not the last node
            if i < len(rnn_nodes) - 1:
                self.play(Create(sequential_arrows[i]), run_time=arrow_spawn_time)

        # Define connections between words as per the provided information
        # Using indices corresponding to the words list
        # 0: "The", 1: "Cat", 2: "Sat", 3: "On", 4: "The", 5: "Mat"
        attention_connections = {
            0: [1, 2],       # "The" -> "Cat", "Sat"
            1: [2, 4],       # "Cat" -> "Sat", "The"
            2: [1, 3, 4],    # "Sat" -> "Cat", "On", "The"
            3: [5, 2],       # "On" -> "Mat", "Sat"
            4: [5],          # "The" -> "Mat"
            5: [0, 3, 2]      # "Mat" -> "The", "On", "Sat"
        }

        # Categorize attention arrows into forward and backward
        forward_attention_arrows = VGroup()
        backward_attention_arrows = VGroup()
        attention_arrow_mapping = {}  # To map arrows to their connections

        for source_idx, target_indices in attention_connections.items():
            for target_idx in target_indices:
                start_node = rnn_nodes[source_idx]
                end_node = rnn_nodes[target_idx]

                if target_idx > source_idx:
                    start_point = start_node.get_bottom()
                    end_point = end_node.get_bottom()
                    # Forward connection: position below nodes
                    arrow = CurvedArrow(
                        start_point=start_point + DOWN * 0.1,  # Slight offset to prevent overlap
                        end_point=end_point + DOWN * 0.1,
                        color=BLUE,
                        angle=PI/4,
                        stroke_width=4  # Thin lines
                    )
                    
                    forward_attention_arrows.add(arrow)
                    attention_arrow_mapping[arrow] = (source_idx, target_idx, 'forward', len(attention_arrow_mapping))
                else:
                    start_point = start_node.get_top()
                    end_point = end_node.get_top()
                    # Backward connection: position above nodes
                    arrow = CurvedArrow(
                        start_point=start_point + UP * 0.1,  # Slight offset to prevent overlap
                        end_point=end_point + UP * 0.1,
                        color=GREEN,
                        angle=PI/4,
                        stroke_width= 4  # Thin lines
                    )
                    
                    backward_attention_arrows.add(arrow)
                    attention_arrow_mapping[arrow] = (source_idx, target_idx, 'backward', len(attention_arrow_mapping))

        
        # Create a flat group of all attention arrows for proper subtraction
        all_attention_arrows = VGroup(*forward_attention_arrows, *backward_attention_arrows)

        # Define the steps of nodes covered by the box
        in_box_per_step = [
            [0],            # Step 1: Only the first word
            [0, 1],         # Step 2: First and second words
            [0, 1, 2],      # Step 3: First, second, and third words
            [1, 2, 3],      # Step 4: Second, third, and fourth words
            [2, 3, 4],      # Step 5: Third, fourth, and fifth words
            [3, 4, 5],      # Step 6: Fourth, fifth, and sixth words
        ]
        
        # Determine active arrows (both source and target within step_nodes)
        active_arrows = VGroup()
        index_list = []
        # Define the box movement steps based on in_box_per_step
        for i, step_nodes in enumerate(in_box_per_step):
            if i == 0:
                # Create the box
                box_width = total_width/2 + node_space/2
                box = Rectangle(height=4, width=box_width, color=RED, fill_opacity=0)
                box.move_to((rnn_nodes[0].get_right() + rnn_nodes[1].get_left())/2 - (box_width/2)*RIGHT)  
                self.play(Create(box))
            else:
                self.play(box.animate.move_to(box.get_center() + RIGHT * node_space), run_time=box_move_time)
            

            for arrow, (src, tgt, direction, arrow_idx) in attention_arrow_mapping.items():
                if src in step_nodes and tgt in step_nodes:
                    if arrow_idx not in index_list:
                        active_arrows.add(arrow)
                        index_list.append(arrow_idx)
                        self.play(Create(arrow))
                else:
                    if arrow in active_arrows:
                        self.play(FadeOut(arrow), run_time=0.1)
                        active_arrows.remove(arrow)
            
            
        self.wait(2)
        self.wait(15)  # Added wait time