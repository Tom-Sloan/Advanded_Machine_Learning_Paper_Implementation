import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report

from src.data_management.dataset import load_image_pairs, get_data_loaders
from src.data_management.augmentation import augment_data
from models.vit import create_vit_base
from training.trainer import Trainer

def train(args):
    # Load and prepare the dataset
    X, y = load_image_pairs(args.data_dir)
    print(f"Number of images: {len(X)}")
    print(f"Unique classes: {np.unique(y)}")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if args.augment:
        X_train, y_train = augment_data(X_train, y_train, augmentations_per_image=5)
        print(f"Number of training images after augmentation: {len(X_train)}")

    # Convert labels to numeric
    unique_labels = np.unique(y)
    y_train = np.array([np.where(unique_labels == label)[0][0] for label in y_train])
    y_test = np.array([np.where(unique_labels == label)[0][0] for label in y_test])

    # Get data loaders
    train_loader, test_loader = get_data_loaders(X_train, X_test, y_train, y_test, batch_size=args.batch_size)

    # Initialize model
    num_classes = len(unique_labels)
    model = create_vit_base()
    model.head = nn.Linear(model.head.in_features, num_classes)

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create trainer
    trainer = Trainer(model, criterion, optimizer, device)
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_acc = trainer.train_epoch(train_loader)
        test_acc, is_best, _ = trainer.evaluate(test_loader)
        
        print(f"Train Accuracy: {train_acc*100:.2f}%")
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        
        # Save if best model
        if is_best:
            trainer.save_model(args.model_path)
            print(f"New best model saved! Accuracy: {test_acc*100:.2f}%")

def evaluate(args):
    # Load dataset
    X, y = load_image_pairs(args.data_dir)
    unique_labels = np.unique(y)
    
    # Prepare test data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_test = np.array([np.where(unique_labels == label)[0][0] for label in y_test])
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = create_vit_base()
    model.head = nn.Linear(model.head.in_features, len(unique_labels))
    
    # Load saved model
    model, best_acc = Trainer.load_model(model, args.model_path, device)
    model = model.to(device)
    print(f"Loaded model with best accuracy: {best_acc*100:.2f}%")
    
    # Create test loader
    _, test_loader = get_data_loaders(X_test, X_test, y_test, y_test, batch_size=args.batch_size)
    
    # Evaluate
    trainer = Trainer(model, None, None, device)
    test_acc, _, (predictions, labels) = trainer.evaluate(test_loader)
    
    # Print detailed metrics
    print("\nTest Results:")
    print(f"Accuracy: {test_acc*100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(labels, predictions, target_names=unique_labels))

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate ViT model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                      help='train or eval mode')
    parser.add_argument('--data_dir', type=str, default='./data/Stoneflies',
                      help='path to dataset')
    parser.add_argument('--model_path', type=str, default='@/trained_models/vit_stonefly/best_model.pt',
                      help='path to save/load model')
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