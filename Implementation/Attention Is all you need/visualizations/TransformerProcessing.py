from manim import *

class TransformerProcessing(Scene):
    def construct(self):
        title = Text("Transformer", font_size=36)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        # Animation speed control
        node_spawn_time = 0.1
        arrow_spawn_time = 0.1
        fade_time = 0.5    # Time for fading arrows in/out
        node_width = 1    # Width of each node
        node_space = 2

        # Create the sentence
        sentence = "The cat sat on the mat"
        words = sentence.split()

        # Create nodes representing words with increased spacing and centered
        total_width = (len(words) - 1) * node_space  # Increased spacing between nodes
        start_x = -total_width / 2
        rnn_nodes = []
        for i in range(len(words)):
            circle = Circle(radius=node_width/2, color=YELLOW, stroke_width=6)
            text = Text(words[i], font_size=24).move_to(circle.get_center())
            node = VGroup(circle, text)
            node.shift(RIGHT * (start_x + i * node_space))
            rnn_nodes.append(node)

        # Animate the creation of word nodes
        for node in rnn_nodes:
            self.play(Create(node), run_time=node_spawn_time)

        # Define connections between words and their strengths
        # 0: "The", 1: "Cat", 2: "Sat", 3: "On", 4: "The", 5: "Mat"
        attention_connections = {
            0: {1: 0.8, 2: 0.4},       # "The" -> "Cat" (strong), "Sat" (weak)
            1: {2: 0.9, 4: 0.3},       # "Cat" -> "Sat" (very strong), "The" (weak)
            2: {1: 0.7, 3: 0.6, 4: 0.5},    # "Sat" -> "Cat" (strong), "On" (medium), "The" (medium)
            3: {5: 0.8, 2: 0.4},       # "On" -> "Mat" (strong), "Sat" (weak)
            4: {5: 0.9},               # "The" -> "Mat" (very strong)
            5: {0: 0.3, 3: 0.7, 2: 0.5}      # "Mat" -> "The" (weak), "On" (strong), "Sat" (medium)
        }

        # Create attention arrows
        attention_arrows = VGroup()
        for source_idx, targets in attention_connections.items():
            for target_idx, strength in targets.items():
                start_node = rnn_nodes[source_idx]
                end_node = rnn_nodes[target_idx]

                if target_idx > source_idx:
                    start_point = start_node.get_bottom()
                    end_point = end_node.get_bottom()
                    color = BLUE
                    start_point += DOWN * 0.1 
                    end_point += DOWN * 0.1 
                else:
                    start_point = start_node.get_top()
                    end_point = end_node.get_top()
                    color = GREEN
                    start_point += UP * 0.1 
                    end_point += UP * 0.1 
                
                # Adjust stroke width based on connection strength
                stroke_width = 1 + 5 * strength  # Scale from 1 to 6 based on strength

                arrow = CurvedArrow(
                    start_point=start_point,
                    end_point=end_point,
                    color=color,
                    angle=PI/4,
                    stroke_width=stroke_width * 2.5
                )
                attention_arrows.add(arrow)

        # Create a large box around the entire scene
        padding = 1  # Add some padding around the scene
        scene_width = total_width + 2 * padding
        scene_height = 6  # Adjust this value based on your scene's height
        scene_box = Rectangle(
            width=scene_width,
            height=scene_height,
            color=RED,
            stroke_width=2,
            fill_opacity=0
        )
        scene_box.move_to(ORIGIN)

        # Show the scene box
        self.play(Create(scene_box))

        # Show all attention arrows
        for arrow in attention_arrows:
            self.play(Create(arrow), run_time=arrow_spawn_time)

        self.wait(2)
        self.wait(15)  # Added wait time