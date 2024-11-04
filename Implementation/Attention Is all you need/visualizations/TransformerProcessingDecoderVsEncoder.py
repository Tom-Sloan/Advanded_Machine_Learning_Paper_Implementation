from manim import *

class TransformerProcessingCreator:
    def __init__(self, scene, position, scale=1, has_forward=True):
        self.scene = scene
        self.position = position
        self.scale = scale
        self.has_forward = has_forward
        # Animation speed control
        self.node_spawn_time = 0.1 * scale
        self.arrow_spawn_time = 0.1 * scale
        self.fade_time = 0.5 * scale
        self.node_width = 1 * scale
        self.node_space = 2 * scale

        self.sentence = "The cat sat on the mat"
        self.words = self.sentence.split()

        self.create_nodes()
        self.create_attention_connections()
        self.create_attention_arrows()

    def create_nodes(self):
        total_width = (len(self.words) - 1) * self.node_space
        start_x = -total_width / 2
        self.rnn_nodes = []
        for i in range(len(self.words)):
            circle = Circle(radius=self.node_width/2, color=YELLOW, stroke_width=6*self.scale)
            text = Text(self.words[i], font_size=24*self.scale).move_to(circle.get_center())
            node = VGroup(circle, text)
            node.shift(RIGHT * (start_x + i * self.node_space))
            self.rnn_nodes.append(node)

    def create_attention_connections(self):
        self.attention_connections = {
            0: {1: 0.8, 2: 0.4},
            1: {2: 0.9, 4: 0.3},
            2: {1: 0.7, 3: 0.6, 4: 0.5},
            3: {5: 0.8, 2: 0.4},
            4: {5: 0.9},
            5: {0: 0.3, 3: 0.7, 2: 0.5}
        }

    def create_attention_arrows(self):
        self.attention_arrows = VGroup()
        for source_idx, targets in self.attention_connections.items():
            for target_idx, strength in targets.items():
                start_node = self.rnn_nodes[source_idx]
                end_node = self.rnn_nodes[target_idx]

                if target_idx > source_idx:
                    if self.has_forward:
                        start_point = start_node.get_bottom()
                        end_point = end_node.get_bottom()
                        color = BLUE
                        start_point += DOWN * 0.1 * self.scale
                        end_point += DOWN * 0.1 * self.scale
                    else:
                        continue
                else:
                    start_point = start_node.get_top()
                    end_point = end_node.get_top()
                    color = GREEN
                    start_point += UP * 0.1 * self.scale
                    end_point += UP * 0.1 * self.scale
                
                stroke_width = (1 + 5 * strength) * self.scale

                arrow = CurvedArrow(
                    start_point=start_point,
                    end_point=end_point,
                    color=color,
                    angle=PI/4,
                    stroke_width=stroke_width * 2.5
                )
                self.attention_arrows.add(arrow)

    def get_animations(self):
        # Scale and position all elements
        all_elements = VGroup(*self.rnn_nodes, self.attention_arrows)
        all_elements.scale(self.scale).move_to(self.position)

        animations = []
        # Animate nodes
        for node in self.rnn_nodes:
            animations.append(Create(node))

        # Animate arrows
        for arrow in self.attention_arrows:
            animations.append(Create(arrow))

        return animations

class TransformerProcessingDecoderVsEncoder(Scene):
    def construct(self):
        shift = 3.5
        title = Text("Transformer: Encoder vs Decoder", font_size=36)
        title.to_edge(UP, buff=0.5)

        # Create Encoder title
        encoder_title = Text("Encoder", font_size=24)

        # Create Decoder title
        decoder_title = Text("Decoder", font_size=24)

        encoder = TransformerProcessingCreator(self, LEFT * shift, scale=0.7)
        decoder = TransformerProcessingCreator(self, RIGHT * shift, scale=0.7, has_forward=False)
        encoder_title.next_to(LEFT * shift + UP * 1, UP)
        decoder_title.next_to(RIGHT * shift + UP * 1, UP)

        # Scale and position decoder elements
        decoder_elements = VGroup(*decoder.rnn_nodes, decoder.attention_arrows)
        decoder_elements.scale(0.7).move_to(RIGHT * shift + UP * 0.2)

        # Play animations for title and encoder
        self.play(
            Write(title),
            Write(encoder_title),
            Write(decoder_title),
            *encoder.get_animations()
        )

        # Play animations for decoder, one element at a time
        decoder_animations = [Create(node) for node in decoder.rnn_nodes] + [Create(arrow) for arrow in decoder.attention_arrows]
        
        # Move the 6 animation to the 3 spot in the list
        decoder_animations.insert(3, decoder_animations.pop(6))
        decoder_animations.insert(5, decoder_animations.pop(7))
        

        for i, animation in enumerate(decoder_animations):
            if i < len(decoder_animations) - 3:
                self.play(animation)
            else:
                break
        
        self.play(*decoder_animations[-3:])

        self.wait(2)
        self.wait(15)  # Added wait time