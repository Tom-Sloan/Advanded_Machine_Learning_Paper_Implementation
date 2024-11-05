import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn as nn

# Import the shared data loading function
from src.data_management.dataset import load_image_pairs

class GradCAM:
    """GradCAM implementation for visualizing model decisions"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

    def generate_cam(self, class_idx):
        gradients = self.gradients.data.numpy()
        activations = self.activations.data.numpy()
        weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-10)  # Added small epsilon to prevent division by zero
        return cam

def preprocess_image(image):
    """Preprocess image for model input"""
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
        
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def overlay_cam_on_image(original_image, cam):
    """Create heatmap overlay on original image"""
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)
    
    overlayed_image = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
    return overlayed_image

def run_gradcam_visualization(data_dir='@/data/Stoneflies', num_samples=10, save_path=None):
    """Run GradCAM visualization on sample images"""
    try:
        # Load and prepare data using the shared function
        X, y = load_image_pairs(data_dir)
        print(f"Loaded {len(X)} images from {len(np.unique(y))} classes")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize model
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_classes = len(np.unique(y_train))
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.eval()
        
        # Initialize GradCAM
        target_layer = model.layer4[2].conv2
        grad_cam = GradCAM(model, target_layer)
        
        # Get random samples
        sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Create visualization grid
        rows = (num_samples + 4) // 5  # Ceiling division
        cols = min(5, num_samples)
        plt.figure(figsize=(4*cols, 4*rows))
        
        for idx, sample_idx in enumerate(sample_indices):
            # Process image
            input_image = preprocess_image(X_test[sample_idx])
            original_image = X_test[sample_idx].astype(np.uint8)
            
            # Get model prediction
            output = grad_cam.forward(input_image)
            pred_class = torch.argmax(output).item()
            
            # Generate GradCAM
            one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
            one_hot_output[0][pred_class] = 1
            output.backward(gradient=one_hot_output)
            
            cam = grad_cam.generate_cam(pred_class)
            overlayed_image = overlay_cam_on_image(original_image, cam)
            
            # Plot
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(overlayed_image)
            plt.title(f'Predicted: {y_test[sample_idx]}')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error running GradCAM visualization: {e}")

if __name__ == "__main__":
    # Run visualization
    save_dir = '@/visualizations/gradcam'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'gradcam_visualization.png')
    
    run_gradcam_visualization(
        data_dir='@/data/Stoneflies',
        num_samples=10,
        save_path=save_path
    )
