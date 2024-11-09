import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from torchvision.transforms import (
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    RandomAffine,
    ColorJitter,
    GaussianBlur,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomEqualize,
    RandomPerspective,
    Compose,
    ToTensor,
    Normalize,
    RandomApply,
    RandomGrayscale
)
import random
from pathlib import Path
from datasets import load_dataset

class BaseDataset(ABC):
    """Abstract base class for all datasets"""
    @abstractmethod
    def load_data(self):
        """Load the dataset"""
        pass
    
    @abstractmethod
    def get_sample(self, num_samples=5):
        """Get a random sample from the dataset"""
        pass

class StoneflyDataset(Dataset, BaseDataset):
    """Dataset class for Stonefly images with optional segmentation masks"""
    def __init__(self, data_dir=None, images=None, labels=None, segmentations=None, transform=None, target_size=(224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        
        if data_dir is not None:
            # Load data directly in init if data_dir is provided
            self.images = []
            self.labels = []
            
            stonefly_dir = os.path.join(self.data_dir, 'stonefly')
            for bug_type in tqdm(os.listdir(stonefly_dir), desc="Loading images"):
                if bug_type.startswith('.'): 
                    continue
                    
                bug_type_path = os.path.join(stonefly_dir, bug_type)
                if os.path.isdir(bug_type_path):
                    for set_num in os.listdir(bug_type_path):
                        set_path = os.path.join(bug_type_path, set_num)
                        if os.path.isdir(set_path):
                            for image_file in os.listdir(set_path):
                                if image_file.endswith('.jpg'):
                                    img_path = os.path.join(set_path, image_file)
                                    img = Image.open(img_path).convert('RGB')
                                    img = img.resize(self.target_size)
                                    self.images.append(np.array(img))
                                    self.labels.append(bug_type)
            
            self.images = np.array(self.images)
            self.labels = np.array(self.labels)
        else:
            self.images = images
            self.labels = labels
            
        self.segmentations = segmentations
    
    def load_data(self):
        """Implement load_data to satisfy BaseDataset, but not used"""
        return self.images, self.labels
    
    def get_sample(self, num_samples=5):
        """Get random samples from the dataset with visualization"""
        if len(self.images) == 0:
            return []
        
        # Create a figure with subplots
        fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        # Get random indices
        indices = np.random.choice(len(self.images), num_samples, replace=False)
        
        samples = []
        for idx, ax in zip(indices, axes):
            img = self.images[idx]
            label = self.labels[idx]
            
            # Display image
            ax.imshow(img)
            ax.set_title(f'Label: {label}')
            ax.axis('off')
            
            samples.append((img, label))
        
        plt.tight_layout()
        return fig, samples
    
    def visualize_predictions(self, model, num_samples=20, device=None, title="Model Predictions", 
                            overlay_func=None):
        """Visualize model predictions on random samples"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Create visualization
        rows = int(np.ceil(num_samples / 5))
        fig = plt.figure(figsize=(10, 1.5*rows))
        sample_indices = np.random.choice(len(self.images), num_samples, replace=False)
        
        correct_count = 0
        unique_labels = np.unique(self.labels)
        
        for idx, sample_idx in enumerate(sample_indices):
            plt.subplot(rows, 5, idx + 1)
            
            # Get the image and prepare it for the model
            img = self.images[sample_idx]
            true_label = self.labels[sample_idx]
            
            # Transform for model input
            if self.transform:
                input_tensor = self.transform(img).unsqueeze(0).to(device)
            else:
                input_tensor = get_eval_transforms()(img).unsqueeze(0).to(device)
            
            # Get model prediction
            with torch.no_grad():
                output = model(input_tensor)
                # Handle HuggingFace model output
                if hasattr(output, 'logits'):
                    output = output.logits
                pred_idx = torch.argmax(output).item()
            
            # Create visualization
            if overlay_func is not None:
                display_img = overlay_func(model, img, pred_idx)
            else:
                display_img = img
            
            # Get labels
            pred_label = unique_labels[pred_idx]
            true_label_idx = np.where(unique_labels == true_label)[0][0]
            
            plt.imshow(display_img)
            plt.title(f'Pred: {pred_label}\nTrue: {true_label}', fontsize=8)
            plt.axis('off')
            
            correct_count += (pred_idx == true_label_idx)
        
        # Calculate and display accuracy
        accuracy = (correct_count / num_samples) * 100
        plt.suptitle(f'{title}\nAccuracy: {accuracy:.2f}%', y=0.95)
        plt.tight_layout()
        return fig, accuracy
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        if self.segmentations is not None:
            seg = self.segmentations[idx]
            if self.transform:
                seg = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor()
                ])(seg)
            return image, seg, label
            
        return image, label

class ShakespeareCharDataset(BaseDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.load_data()
        
    def load_data(self):
        with open(os.path.join(self.data_dir, 'input.txt'), 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    def get_sample(self, num_samples=5):
        """Get random sequences from the text"""
        if not self.data:
            return []
        
        sequences = []
        seq_length = 100  # Get 100 character sequences
        
        for _ in range(num_samples):
            start_idx = np.random.randint(0, len(self.data) - seq_length)
            sequence = self.data[start_idx:start_idx + seq_length]
            sequences.append(sequence)
            
        return sequences, seq_length

class ShakespeareWordDataset(BaseDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.load_data()
        
    def load_data(self):
        with open(os.path.join(self.data_dir, 'input.txt'), 'r', encoding='utf-8') as f:
            text = f.read().split()
        return text
    
    def get_sample(self, num_samples=5):
        """Get random word sequences from the text"""
        if not self.data:
            return []
        
        sequences = []
        seq_length = 20  # Get 20 word sequences
        
        for _ in range(num_samples):
            start_idx = np.random.randint(0, len(self.data) - seq_length)
            sequence = ' '.join(self.data[start_idx:start_idx + seq_length])
            sequences.append(sequence)
            
        return sequences, seq_length

class CityscapesDataset(Dataset, BaseDataset):
    """Dataset class for Cityscapes image-to-image translation with side-by-side format"""
    def __init__(self, data_dir, split='train', transform=None):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.images = self.load_data()
        
    def load_data(self):
        """Load Cityscapes dataset images"""
        # Get images from the appropriate split directory
        split_dir = os.path.join(self.data_dir, self.split)
        image_paths = []
        
        # Collect all images
        for img_name in sorted(os.listdir(split_dir)):
            if img_name.endswith(('.jpg', '.png')):
                image_path = os.path.join(split_dir, img_name)
                image_paths.append(image_path)
        
        print(f"Found {len(image_paths)} images in {self.split} set")
        return image_paths
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load the combined image
        img = Image.open(self.images[idx])
        w, h = img.size
        
        # Split image into real photo (left) and semantic map (right)
        real_img = img.crop((0, 0, w//2, h))
        semantic_img = img.crop((w//2, 0, w, h))
        
        # Resize to 512x512 if needed
        if real_img.size != (512, 512):
            real_img = real_img.resize((512, 512), Image.BICUBIC)
            semantic_img = semantic_img.resize((512, 512), Image.BICUBIC)
        
        # Convert to RGB
        real_img = real_img.convert('RGB')
        semantic_img = semantic_img.convert('RGB')
        
        # Apply transforms if specified
        if self.transform:
            # Use same random seed for both images to ensure consistent transforms
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            real_img = self.transform(real_img)
            
            torch.manual_seed(seed)
            semantic_img = self.transform(semantic_img)
        else:
            # Default transform to tensor and normalize
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            real_img = to_tensor(real_img)
            semantic_img = to_tensor(semantic_img)
        
        return semantic_img, real_img  # Return semantic map as input, real photo as target
    
    def get_sample(self, num_samples=5):
        """Get random samples from the dataset with visualization"""
        if len(self.images) == 0:
            return []
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
        
        # Get random indices
        indices = np.random.choice(len(self.images), num_samples, replace=False)
        
        samples = []
        for idx, i in enumerate(indices):
            # Load and split image
            img = Image.open(self.images[i])
            w, h = img.size
            
            real_img = img.crop((0, 0, w//2, h))
            semantic_img = img.crop((w//2, 0, w, h))
            
            # Resize if needed
            if real_img.size != (512, 512):
                real_img = real_img.resize((512, 512), Image.BICUBIC)
                semantic_img = semantic_img.resize((512, 512), Image.BICUBIC)
            
            # Display images
            axes[0, idx].imshow(semantic_img)
            axes[0, idx].set_title('Semantic Map')
            axes[0, idx].axis('off')
            
            axes[1, idx].imshow(real_img)
            axes[1, idx].set_title('Photo')
            axes[1, idx].axis('off')
            
            samples.append((semantic_img, real_img))
        
        plt.tight_layout()
        return fig, samples

class CelebAHQDataset(Dataset, BaseDataset):
    """Dataset class for CelebA-HQ 256x256 images"""
    def __init__(self, data_dir=None, transform=None):
        super().__init__()
        self.transform = transform
        self.data_dir = Path('./data/celeba-hq') if data_dir is None else Path(data_dir)
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Use HuggingFace datasets with local caching
        print("Loading CelebA-HQ dataset (this may take a while the first time)...")
        self.dataset = load_dataset(
            "korexyz/celeba-hq-256x256",
            cache_dir=str(self.data_dir),
            split='train'
        )
        print(f"Loaded {len(self.dataset)} images from CelebA-HQ dataset")
        print(f"Dataset cached at: {self.data_dir}")
    
    def load_data(self):
        """Implement load_data to satisfy BaseDataset"""
        return self.dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get image and label from HuggingFace dataset
        # Convert idx to Python int to avoid numpy.int64 issues
        sample = self.dataset[int(idx)]
        image = sample['image']
        label = sample['label']  # 0 for female, 1 for male
        
        # Convert to tensor and normalize to [-1, 1]
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image = transform(image)
        
        return image, label
    
    def get_sample(self, num_samples=5):
        """Get random samples from the dataset with visualization"""
        # Create figure with subplots
        fig, axes = plt.subplots(1, num_samples, figsize=(3*num_samples, 3))
        if num_samples == 1:
            axes = [axes]
        
        # Get random indices and convert to Python int
        indices = [int(i) for i in np.random.choice(len(self), num_samples, replace=False)]
        
        samples = []
        for idx, ax in zip(indices, axes):
            # Get image and label
            sample = self.dataset[idx]
            image = sample['image']
            label = sample['label']
            gender = 'Male' if label == 1 else 'Female'
            
            # Display image
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(f'{gender}', pad=10)
            
            samples.append((image, label))
        
        plt.suptitle('CelebA-HQ Samples', y=1.05)
        plt.tight_layout()
        return fig, samples

def get_dataset(dataset_name, data_dir=None, split='train', transform=None):
    """Factory function to get the appropriate dataset"""
    datasets = {
        'stoneflies': StoneflyDataset,
        'shakespeare_char': ShakespeareCharDataset,
        'shakespeare_word': ShakespeareWordDataset,
        'cityscapes': CityscapesDataset,
        'celeba-hq': CelebAHQDataset
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not found. Available datasets: {list(datasets.keys())}")
    
    if dataset_name == 'cityscapes':
        return datasets[dataset_name](data_dir, split, transform)
    elif dataset_name == 'celeba-hq':
        return datasets[dataset_name](transform=transform)
    else:
        return datasets[dataset_name](data_dir)

def get_training_transforms():
    """Get enhanced training transforms with stronger augmentation"""
    return Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),  # Wider scale range
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=180),  # Full rotation
        transforms.RandomAffine(
            degrees=30,
            translate=(0.2, 0.2),
            scale=(0.8, 1.2),
            shear=15
        ),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                hue=0.2
            )
        ], p=0.8),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.3),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3),  # Add random erasing
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def get_eval_transforms():
    """Get standard evaluation transforms"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def mixup_collate(batch):
    """Collate function that handles both regular and mixup batches"""
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    
    # Check if any item in the batch is a mixup sample
    if any(isinstance(label, tuple) for label in labels):
        # Convert all labels to mixup format for consistency
        processed_labels = []
        for label in labels:
            if isinstance(label, tuple):
                # Already a mixup sample
                processed_labels.append(label)
            else:
                # Convert regular label to mixup format (same label twice with lambda=1)
                processed_labels.append((label, label, 1.0))
        
        # Now all labels are in mixup format, we can safely unpack them
        label1 = torch.tensor([label[0] for label in processed_labels])
        label2 = torch.tensor([label[1] for label in processed_labels])
        lam = torch.tensor([label[2] for label in processed_labels])
        return images, (label1, label2, lam)
    else:
        # All regular labels
        labels = torch.tensor(labels)
        return images, labels

def create_data_loaders(dataset, batch_size=32, train_transform=None, eval_transform=None, val_split=0.15, test_split=0.15):
    """
    Create train, validation and test data loaders with improved splitting strategy
    """
    X, y = dataset.images, dataset.labels
    
    # First split out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_split,
        stratify=y,
        random_state=42
    )
    
    # Then split remaining data into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_split/(1-test_split),
        stratify=y_temp,
        random_state=42
    )
    
    # Get unique labels
    unique_labels = np.unique(y)
    
    # Convert labels to numeric indices directly using unique_labels
    y_train_idx = np.array([np.where(unique_labels == label)[0][0] for label in y_train])
    y_val_idx = np.array([np.where(unique_labels == label)[0][0] for label in y_val])
    y_test_idx = np.array([np.where(unique_labels == label)[0][0] for label in y_test])
    
    # Print split sizes
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Print class distribution
    for label in unique_labels:
        train_count = sum(y_train == label)
        val_count = sum(y_val == label)
        test_count = sum(y_test == label)
        print(f"\nClass {label} distribution:")
        print(f"  Train: {train_count} ({train_count/len(y_train)*100:.1f}%)")
        print(f"  Val: {val_count} ({val_count/len(y_val)*100:.1f}%)")
        print(f"  Test: {test_count} ({test_count/len(y_test)*100:.1f}%)")
    
    # Use default transforms if none provided
    if train_transform is None:
        train_transform = get_training_transforms()
    if eval_transform is None:
        eval_transform = get_eval_transforms()
    
    # Create datasets with mixup augmentation
    train_dataset = MixupDataset(X_train, y_train_idx, transform=train_transform)
    val_dataset = StoneflyDataset(images=X_val, labels=y_val_idx, transform=eval_transform)
    test_dataset = StoneflyDataset(images=X_test, labels=y_test_idx, transform=eval_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Changed from 4 to 0 for debugging
        pin_memory=True,
        collate_fn=mixup_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0 for debugging
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Changed from 4 to 0 for debugging
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, unique_labels

class MixupDataset(Dataset):
    """Dataset wrapper that applies mixup augmentation"""
    def __init__(self, images, labels, transform=None, alpha=0.4):  # Increased alpha
        self.images = images
        self.labels = labels
        self.transform = transform
        self.alpha = alpha
        
        # Add oversampling for minority classes
        class_counts = np.bincount(labels)
        max_count = max(class_counts)
        self.sample_weights = [max_count/count for count in class_counts]
        self.sample_weights = [self.sample_weights[label] for label in labels]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply mixup with probability 0.5
        if random.random() < 0.5:
            # Get another random sample
            mix_idx = random.randint(0, len(self.images)-1)
            mix_image = self.images[mix_idx]
            mix_label = self.labels[mix_idx]
            
            # Generate mixup weight
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Apply transforms to both images before mixing
            if self.transform:
                image = self.transform(image)
                mix_image = self.transform(mix_image)
            
            # Combine images
            image = lam * image + (1 - lam) * mix_image
            label = (int(label), int(mix_label), float(lam))  # Ensure proper types
        else:
            # Regular sample - just apply transform
            if self.transform:
                image = self.transform(image)
        
        return image, label

def print_data_info(train_loader, val_loader, test_loader, unique_labels):
    """
    Print information about the datasets including shapes and labels.
    
    Args:
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader: DataLoader for the test set.
        unique_labels: Array of unique labels in the dataset.
    """
    print("Dataset Information:")
    
    # Training set
    train_size = len(train_loader.dataset)
    print(f"Training set size: {train_size}")
    print(f"Training set labels: {unique_labels}")
    
    # Validation set
    val_size = len(val_loader.dataset)
    print(f"Validation set size: {val_size}")
    print(f"Validation set labels: {unique_labels}")
    
    # Test set
    test_size = len(test_loader.dataset)
    print(f"Test set size: {test_size}")
    print(f"Test set labels: {unique_labels}")
    
    # Print shapes of the first batch of each loader
    for loader_name, loader in zip(['Training', 'Validation', 'Test'], [train_loader, val_loader, test_loader]):
        inputs, labels = next(iter(loader))
        print(f"{loader_name} set input shape: {inputs.shape}, labels shape: {labels.shape}")

# Example usage in your train or evaluate function
# After creating the data loaders, you can call this function:
# print_data_info(train_loader, val_loader, test_loader, unique_labels)