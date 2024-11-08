# Import necessary modules from manim
from manim import *

# Import necessary modules for torch
import torch
import torch.nn as nn
import torch.optim as optim


#  recurrent neural network animation
class VanishingGradient(Scene):
    def construct(self):
        # Define speed controls
        node_creation_speed = 0.5
        arrow_creation_speed = 0.5
        text_writing_speed = 0.5
        arrow_fading_speed = 0.5

        # Set a larger scene size
        self.camera.frame_width = 16
        self.camera.frame_height = 9

        # Create the nodes of the RNN with increased spacing, centered in the scene
        nodes = [Circle(radius=0.3, color=WHITE).shift(LEFT * 4 + RIGHT * 2 * i) for i in range(5)]

        # Draw the nodes
        for node in nodes:
            self.play(Create(node), run_time=1/node_creation_speed)

        # Create arrows representing connections (gradients) between nodes
        arrows = [Arrow(start=nodes[i].get_center(), end=nodes[i+1].get_center(), buff=0.1) for i in range(len(nodes)-1)]

        # Draw the arrows with initial colors representing the gradients
        for i, arrow in enumerate(arrows):
            arrow.set_color(interpolate_color(YELLOW, RED, i / len(arrows)))
            self.play(Create(arrow), run_time=1/arrow_creation_speed)

        # Create gradient magnitude text for each node with smaller font size
        gradient_texts = [Text(f"Grad: {0.9**i:.2f}", font_size=20).next_to(nodes[i], UP) for i in range(len(nodes))]

        # Display the gradient values
        for text in gradient_texts:
            self.play(Write(text), run_time=1/text_writing_speed)

        # Animate the arrows to show the vanishing effect
        for i, arrow in enumerate(arrows):
            self.play(arrow.animate.set_opacity(1 - i / len(arrows)), run_time=1/arrow_fading_speed)

        self.wait(15)  # Added wait time