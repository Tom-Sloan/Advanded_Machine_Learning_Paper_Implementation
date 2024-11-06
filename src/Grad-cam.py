import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

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
import argparse
from torchvision import transforms

# Import shared data loading and augmentation functions
from data_management.dataset import (
    get_dataset,
    create_data_loaders,
    get_training_transforms,
    get_eval_transforms
)
from training.trainer import Trainer

class GradCAM:
    """GradCAM implementation for visualizing model decisions"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def forward(self, x):
        return self.model(x)

    def generate_cam(self, class_idx):
        gradients = self.gradients.data.cpu().numpy()
        activations = self.activations.data.cpu().numpy()
        weights = np.mean(gradients, axis=(2, 3))[0, :]
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-10)  # Added small epsilon to prevent division by zero
        return cam

def preprocess_image(image, device):
    """Preprocess image for model input"""
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    # Use the standard eval transforms
    transform = get_eval_transforms()
    
    # Convert to tensor and add batch dimension
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(device)

def overlay_cam_on_image(original_image, cam):
    """Create heatmap overlay on original image"""
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)
    
    overlayed_image = cv2.addWeighted(original_image, 0.7, heatmap, 0.3, 0)
    return overlayed_image

def ensure_model_dir():
    """Ensure the model directory exists"""
    model_dir = os.path.join('./trained_models', 'gradcam')
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def get_model_path():
    """Get the standard model path"""
    return os.path.join('./trained_models', 'gradcam', 'gradcam_resnet50_model.pt')

def train(args):
    """Train and evaluate the model"""
    model_path = get_model_path()
    ensure_model_dir()
    
    print(f"Model will be saved as: {model_path}")
    
    # Load dataset using the new interface
    dataset = get_dataset('stoneflies', args.data_dir)
    print(f"Loaded {len(dataset.images)} images from {len(np.unique(dataset.labels))} classes")
    
    # Get standard transforms
    train_transform = get_training_transforms()
    eval_transform = get_eval_transforms()
    
    # Get data loaders using the new function
    train_loader, val_loader, test_loader, unique_labels = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        train_transform=train_transform,
        eval_transform=eval_transform
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_classes = len(unique_labels)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Fine-tune the model
    print("\nFine-tuning the model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    trainer = Trainer(model, criterion, optimizer, device)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_acc = trainer.train_epoch(train_loader)
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        
        # Validate
        val_acc, is_best, _ = trainer.evaluate(val_loader)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        
        if is_best:
            # Test on test set
            test_acc, _, _ = trainer.evaluate(test_loader)
            print(f"Test Accuracy: {test_acc*100:.2f}%")
            
            # Save model with additional information
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc,
                'unique_labels': unique_labels,
                'num_classes': num_classes,
                'eval_transform': eval_transform.state_dict() if hasattr(eval_transform, 'state_dict') else None
            }
            torch.save(save_dict, model_path)
            print(f"New best model saved! Validation Accuracy: {val_acc*100:.2f}%")
            best_val_acc = val_acc
    
    return model, unique_labels

def gradcam_overlay(model, img, pred_class):
    """Create GradCAM overlay for an image"""
    # Initialize GradCAM
    target_layer = model.layer4[2].conv2
    grad_cam = GradCAM(model, target_layer)
    
    # Process image
    device = next(model.parameters()).device
    if isinstance(img, Image.Image):
        img = np.array(img)
    original_image = img
    
    # Use consistent transform
    input_tensor = get_eval_transforms()(img).unsqueeze(0).to(device)
    output = grad_cam.forward(input_tensor)
    one_hot_output = torch.zeros((1, output.size()[-1]), dtype=torch.float32, device=device)
    one_hot_output[0][pred_class] = 1
    output.backward(gradient=one_hot_output)
    
    cam = grad_cam.generate_cam(pred_class)
    return overlay_cam_on_image(original_image, cam)

def evaluate(args):
    """Load model and run GradCAM visualization"""
    model_path = get_model_path()
    
    if not os.path.exists(model_path):
        print(f"No saved model found at {model_path}")
        print("Please train the model first using 'Train and Evaluate' option")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Load dataset using new interface
    dataset = get_dataset('stoneflies', args.data_dir)
    eval_transform = get_eval_transforms()
    
    # Get data loaders using the new function
    _, _, test_loader, unique_labels = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        eval_transform=eval_transform
    )
    
    # Initialize model with correct number of classes
    num_classes = len(unique_labels)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load model state
    checkpoint = torch.load(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    best_acc = checkpoint.get('best_acc', 0.0)
    print(f"Loaded model with validation accuracy: {best_acc*100:.2f}%")
    
    # Use dataset's visualization method
    fig, accuracy = dataset.visualize_predictions(
        model=model,
        num_samples=args.num_samples,
        device=device,
        title="GradCAM Visualizations",
        overlay_func=gradcam_overlay
    )
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate GradCAM model')
    parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'],
                      help='train or eval mode')
    parser.add_argument('--data_dir', type=str, default='./data/Stoneflies',
                      help='path to dataset')
    parser.add_argument('--epochs', type=int, default=5,
                      help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='learning rate')
    parser.add_argument('--num_samples', type=int, default=20,
                      help='number of samples to visualize')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    evaluate(args)  # Always run evaluation after training or by itself

if __name__ == "__main__":
    main()
