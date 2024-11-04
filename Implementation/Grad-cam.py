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

def load_image_pairs(data_dir, target_size=(224, 224)):
    stonefly_dir = os.path.join(data_dir, 'stonefly')
    images = []
    labels = []
    
    for bug_type in tqdm(os.listdir(stonefly_dir), desc="Loading images"):
        bug_type_path = os.path.join(stonefly_dir, bug_type)
        if os.path.isdir(bug_type_path):
            for set_num in os.listdir(bug_type_path):
                set_path = os.path.join(bug_type_path, set_num)
                if os.path.isdir(set_path):
                    for image_file in os.listdir(set_path):
                        if image_file.endswith('.jpg'):
                            # Load and preprocess stonefly image
                            stonefly_image_path = os.path.join(set_path, image_file)
                            img = Image.open(stonefly_image_path)
                            # Convert to RGB if not already
                            img = img.convert('RGB')
                            # Resize image
                            img = img.resize(target_size)
                            # Convert to numpy array
                            img_array = np.array(img)
                            images.append(img_array)
                            labels.append(bug_type)
    
    return np.array(images), np.array(labels)

# Load the data
data_dir = './Data/Stoneflies'
X, y = load_image_pairs(data_dir)
print(f"Number of images: {len(X)}")
print(f"Number of labels: {len(y)}")
print(f"Unique classes: {np.unique(y)}")
print(f"Image shape: {X[0].shape}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

def preprocess_image(image):
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
        
    # Resize to 224x224 (standard size for ResNet)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return img

def overlay_cam_on_image(original_image, cam):
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Convert CAM to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure original_image is uint8 and has correct shape
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)
    
    # Create overlay
    overlayed_image = cv2.addWeighted(original_image, 0.5, heatmap, 0.5, 0)
    return overlayed_image

# Example usage with your stonefly dataset
if __name__ == "__main__":
    # Load pretrained model and modify final layer for your classes
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)  # Updated weights parameter
    num_classes = len(np.unique(y_train))  # Number of stonefly classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()

    # Specify the target layer
    target_layer = model.layer4[2].conv2
    grad_cam = GradCAM(model, target_layer)

    # Process a sample image from your test set
    # Get 10 random sample indices
    sample_indices = np.random.choice(len(X_test), 10, replace=False)
    
    # Create figure with 2 rows and 5 columns
    plt.figure(figsize=(20, 8))
    
    for idx, sample_idx in enumerate(sample_indices):
        # Preprocess image
        input_image = preprocess_image(X_test[sample_idx])
        original_image = X_test[sample_idx].astype(np.uint8)

        # Forward pass
        output = grad_cam.forward(input_image)
        pred_class = torch.argmax(output).item()
        
        # Backward pass
        one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float32)
        one_hot_output[0][pred_class] = 1
        output.backward(gradient=one_hot_output)

        # Generate overlay
        cam = grad_cam.generate_cam(pred_class)
        overlayed_image = overlay_cam_on_image(original_image, cam)

        # Plot in grid
        plt.subplot(2, 5, idx + 1)
        plt.imshow(overlayed_image)
        plt.title(f'Class: {y_test[sample_idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
