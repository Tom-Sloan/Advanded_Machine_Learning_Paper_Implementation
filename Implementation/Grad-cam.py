import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

    def generate_cam(self, class_idx):
        # Get the gradients and activations
        gradients = self.gradients.data.numpy()
        activations = self.activations.data.numpy()

        # Compute the weights
        weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        # Create the class activation map
        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        # Apply ReLU to the CAM
        cam = np.maximum(cam, 0)

        # Normalize the CAM
        cam = cam / np.max(cam)
        return cam

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def overlay_cam_on_image(original_image, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed_image = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
    return overlayed_image

# Example usage
if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    model.eval()

    # Specify the target layer (for ResNet, it's usually the last convolutional layer)
    target_layer = model.layer4[2].conv2

    grad_cam = GradCAM(model, target_layer)

    # Load and preprocess the image
    image_path = 'path_to_your_image.jpg'
    input_image = preprocess_image(image_path)

    # Forward pass
    output = grad_cam.forward(input_image)

    # Backward pass for the class index you want to visualize
    class_idx = 243  # Example: 'bull mastiff' in ImageNet
    one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
    one_hot_output[0][class_idx] = 1
    output.backward(gradient=one_hot_output)

    # Generate the CAM
    cam = grad_cam.generate_cam(class_idx)

    # Load the original image for overlay
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Overlay the CAM on the original image
    overlayed_image = overlay_cam_on_image(original_image, cam)

    # Display the result
    plt.imshow(overlayed_image)
    plt.axis('off')
    plt.show()
