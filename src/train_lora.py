import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from models.lora import create_lora_model, add_lora_layers, get_lora_params
from data_management.dataset import (
    get_dataset,
    create_data_loaders,
    get_training_transforms,
    get_eval_transforms
)
from training.trainer import Trainer

def evaluate_model(model, test_loader, device, unique_labels, title="Model Predictions"):
    """Evaluate model and return accuracy and predictions"""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Ensure both inputs and model are on same device and type
            inputs = inputs.to(device, dtype=torch.float32)  # Explicitly set dtype
            labels = labels.to(device)
            
            # Move model to same device and type if needed
            if next(model.parameters()).dtype != inputs.dtype:
                model = model.to(dtype=inputs.dtype)
            
            outputs = model(inputs)
            if isinstance(outputs, dict):  # Handle HuggingFace model output
                outputs = outputs.logits
                
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100. * correct / total
    print(f"\n{title}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Print classification report
    pred_species = [unique_labels[pred] for pred in all_preds]
    true_species = [unique_labels[label] for label in all_labels]
    print("\nClassification Report:")
    print(classification_report(true_species, pred_species))
    
    return accuracy, all_preds, all_labels

def train(args):
    # Create necessary directories
    os.makedirs('./trained_models/lora', exist_ok=True)
    os.makedirs('./models/pretrained', exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get dataset and create data loaders
    dataset = get_dataset('stoneflies', args.data_dir)
    train_loader, val_loader, test_loader, unique_labels = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        train_transform=get_training_transforms(),
        eval_transform=get_eval_transforms()
    )
    
    # Create pretrained model with LoRA
    model = create_lora_model(
        model_name=args.model_name,
        num_classes=len(unique_labels),
        cache_dir=args.cache_dir
    )
    
    # Move model to device and set dtype
    model = model.to(device, dtype=torch.float32)
    
    # Evaluate pretrained model before LoRA
    print("\nEvaluating pretrained model before LoRA adaptation:")
    pre_accuracy, pre_preds, pre_labels = evaluate_model(
        model, test_loader, device, unique_labels, 
        title="Pretrained Model (Before LoRA)"
    )
    
    # Add LoRA layers
    model = add_lora_layers(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha
    )
    model = model.to(device)
    
    # Get LoRA parameters for training
    lora_params = get_lora_params(model)
    
    # Setup training
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer and train
    trainer = Trainer(model, criterion, optimizer, device)
    
    # Load checkpoint if specified
    start_epoch = 0
    if hasattr(args, 'checkpoint') and args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    best_val_acc = 0.0
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        trainer.train_epoch(train_loader)
        
        # Evaluate
        val_acc, is_best, _ = trainer.evaluate(val_loader)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        
        if is_best:
            best_val_acc = val_acc
            # Test on test set
            test_acc, _, _ = trainer.evaluate(test_loader)
            print(f"Test Accuracy: {test_acc*100:.2f}%")
            
            # Save best model
            save_path = os.path.join('./trained_models/lora', 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_val_acc,
                'pre_lora_accuracy': pre_accuracy
            }, save_path)
            print(f"New best model saved! Validation Accuracy: {val_acc*100:.2f}%")
    
    # Final evaluation after LoRA
    print("\nEvaluating model after LoRA adaptation:")
    post_accuracy, post_preds, post_labels = evaluate_model(
        model, test_loader, device, unique_labels,
        title="Model After LoRA"
    )
    
    # Print improvement
    print(f"\nAccuracy Improvement: {post_accuracy - pre_accuracy:.2f}%")

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load dataset
    dataset = get_dataset('stoneflies', args.data_dir)
    _, _, test_loader, unique_labels = create_data_loaders(
        dataset,
        batch_size=args.batch_size,
        eval_transform=get_eval_transforms()
    )
    
    # Create model and load weights
    model = create_lora_model(
        model_name=args.model_name,
        num_classes=len(unique_labels),
        cache_dir=args.cache_dir
    )
    model = add_lora_layers(model)
    
    # Load checkpoint
    checkpoint = torch.load('./trained_models/lora/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print original accuracy
    pre_lora_accuracy = checkpoint.get('pre_lora_accuracy', 0.0)
    print(f"\nAccuracy before LoRA: {pre_lora_accuracy:.2f}%")
    
    # Evaluate current model
    accuracy, preds, labels = evaluate_model(
        model, test_loader, device, unique_labels,
        title="LoRA Adapted Model"
    )
    
    # Print improvement
    print(f"Accuracy Improvement: {accuracy - pre_lora_accuracy:.2f}%")
    
    # Use dataset's visualization method
    fig, _ = dataset.visualize_predictions(
        model=model,
        num_samples=20,
        device=device,
        title="LoRA Model Predictions"
    )
    plt.show()

def main():
    # Create necessary directories
    os.makedirs('./trained_models/lora', exist_ok=True)
    os.makedirs('./models/pretrained', exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description='Train or evaluate LoRA (Low-Rank Adaptation) model on image classification'
    )
    
    # Mode argument
    parser.add_argument(
        '--mode', 
        type=str,
        default='train',
        choices=['train', 'eval'],
        help='Operation mode: train (train model from scratch or resume training) or eval (evaluate trained model)'
    )
    
    # Model selection
    parser.add_argument(
        '--model_name',
        type=str,
        default='google/vit-large-patch16-224',
        help='Pretrained model name from HuggingFace'
    )
    
    # Cache directory
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='./models/pretrained',
        help='Directory to cache pretrained models'
    )
    
    # Data directory
    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data/Stoneflies',
        help='Path to dataset directory containing the images'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for training and evaluation'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Learning rate for optimizer'
    )
    
    # LoRA-specific parameters
    parser.add_argument(
        '--lora_rank',
        type=int,
        default=4,
        help='Rank of LoRA adaptation matrices (r in the paper). Lower rank = fewer parameters'
    )
    
    parser.add_argument(
        '--lora_alpha',
        type=float,
        default=1,
        help='LoRA scaling factor (alpha in the paper). Higher values = stronger adaptation'
    )
    
    # Checkpoint handling
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file for resuming training or evaluation'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == '__main__':
    main() 