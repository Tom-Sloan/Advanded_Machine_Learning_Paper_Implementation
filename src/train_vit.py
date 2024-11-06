import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

from data_management.dataset import (
    get_dataset,
    create_data_loaders,
    get_training_transforms,
    get_eval_transforms
)
from data_management.augmentation import augment_data
from models.vit import create_vit_base
from training.trainer import Trainer

def ensure_model_dir(model_path):
    """Ensure the model directory exists"""
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

def get_model_path():
    """Get the standard model path"""
    return os.path.join('./trained_models', 'vit_stonefly_model.pt')

def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Use standard model path
    args.model_path = get_model_path()
    
    # Ensure model directory exists
    ensure_model_dir(args.model_path)
    
    print(f"Model will be saved as: {args.model_path}")
    
    # Load dataset using the new interface
    dataset = get_dataset('stoneflies', args.data_dir)
    print(f"Original dataset size: {len(dataset.images)}")
    print(f"Original image shape: {dataset.images[0].shape}")
    
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
    
    # Print dataset information
    print("\nDataset splits:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Check shapes of first batch
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    test_batch = next(iter(test_loader))
    
    print("\nBatch shapes:")
    print(f"Training batch shape: {train_batch[0].shape}")
    print(f"Validation batch shape: {val_batch[0].shape}")
    print(f"Test batch shape: {test_batch[0].shape}")
    
    # Print class distribution
    print("\nClass distribution:")
    for label in unique_labels:
        train_count = sum(dataset.labels == label)
        print(f"{label}: {train_count} images")
    
    # Initialize model
    num_classes = len(unique_labels)
    print(f"\nNumber of classes: {num_classes}")
    
    # Set learning rate only ONCE and increase it
    args.learning_rate = 1e-3  # Increased initial learning rate
    
    # Initialize model and move to device first
    model, get_layer_lr_decay = create_vit_base()
    model.head = nn.Linear(model.head.in_features, num_classes)
    model = model.to(device)
    
    # Setup optimizer with layer-wise learning rates
    param_groups = get_layer_lr_decay(model, args.learning_rate)
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Setup loss with class weights
    class_counts = np.bincount([np.where(unique_labels == label)[0][0] for label in dataset.labels])
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Setup schedulers
    num_epochs = args.epochs
    warmup_epochs = 2  # Reduced warmup period
    
    # Create warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs * len(train_loader)
    )
    
    # Main scheduler starts after warmup
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(num_epochs - warmup_epochs) * len(train_loader),
        eta_min=1e-6
    )
    
    # Create trainer
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        device,
        warmup_scheduler=warmup_scheduler,  # Pass both schedulers
        main_scheduler=main_scheduler,
        grad_clip_value=1.0
    )
    
    # Training loop
    best_val_acc = 0.0
    no_improve = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_acc = trainer.train_epoch(train_loader, epoch < warmup_epochs)
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        
        # Validate
        val_acc, is_best, _ = trainer.evaluate(val_loader)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        
        if is_best:
            no_improve = 0
            # Test on test set
            test_acc, _, _ = trainer.evaluate(test_loader)
            print(f"Test Accuracy: {test_acc*100:.2f}%")
            
            save_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'warmup_scheduler_state_dict': warmup_scheduler.state_dict() if warmup_scheduler else None,
                'main_scheduler_state_dict': main_scheduler.state_dict() if main_scheduler else None,
                'best_acc': best_val_acc,
                'unique_labels': unique_labels,
                'num_classes': num_classes,
                'epoch': epoch
            }
            torch.save(save_dict, args.model_path)
            print(f"New best model saved! Validation Accuracy: {val_acc*100:.2f}%")
            best_val_acc = val_acc
        else:
            no_improve += 1
            if no_improve >= 10:  # Early stopping after 10 epochs without improvement
                print("Early stopping triggered")
                break

def evaluate(args):
    # Use standard model path
    args.model_path = get_model_path()
    
    if not os.path.exists(args.model_path):
        print(f"No saved model found at {args.model_path}")
        print("Please train the model first using 'Train and Evaluate' option")
        return
    
    print(f"Loading model from: {args.model_path}")
    
    # Load dataset using new interface
    dataset = get_dataset('stoneflies', args.data_dir)
    eval_transform = get_eval_transforms()
    
    # Get data loaders using the new function
    _, val_loader, test_loader, unique_labels = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        eval_transform=eval_transform
    )
    
    # Load saved model and metadata
    checkpoint = torch.load(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize model with correct number of classes
    model, _ = create_vit_base()  # Unpack only the model, ignore the lr_decay function
    model.head = nn.Linear(model.head.in_features, len(unique_labels))
    
    # Load saved model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    print(f"Loaded model with best accuracy: {checkpoint['best_acc']*100:.2f}%")
    
    # Evaluate on both validation and test sets
    trainer = Trainer(model, None, None, device)
    
    print("\nValidation Results:")
    val_acc, _, (val_predictions, val_labels) = trainer.evaluate(val_loader)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    
    # Convert numeric predictions and labels back to original species names
    val_pred_species = [unique_labels[pred] for pred in val_predictions]
    val_true_species = [unique_labels[label] for label in val_labels]
    
    print("\nValidation Classification Report:")
    print(classification_report(val_true_species, val_pred_species))
    
    print("\nTest Results:")
    test_acc, _, (test_predictions, test_labels) = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    
    print("\nTest Classification Report:")
    test_pred_species = [unique_labels[pred] for pred in test_predictions]
    test_true_species = [unique_labels[label] for label in test_labels]
    print(classification_report(test_true_species, test_pred_species))
    
    # Use dataset's visualization method directly
    fig, accuracy = dataset.visualize_predictions(
        model=model,
        num_samples=20,
        device=device,
        title="ViT Predictions"
    )
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate ViT model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                      help='train or eval mode')
    parser.add_argument('--data_dir', type=str, default='./data/Stoneflies',
                      help='path to dataset')
    parser.add_argument('--epochs', type=int, default=10,
                      help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='learning rate')
    parser.add_argument('--augment', action='store_true',
                      help='use data augmentation')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == "__main__":
    main()