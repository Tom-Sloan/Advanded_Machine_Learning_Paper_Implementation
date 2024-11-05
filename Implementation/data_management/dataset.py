import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class StoneflyDataset(Dataset):
    """Dataset class for Stonefly images"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def load_image_pairs(data_dir, target_size=(224, 224)):
    """
    Loads image pairs from the stonefly dataset directory
    
    Args:
        data_dir: Path to data directory
        target_size: Tuple of (height, width) to resize images to
        
    Returns:
        Tuple of (images array, labels array)
    """
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
                            stonefly_image_path = os.path.join(set_path, image_file)
                            img = Image.open(stonefly_image_path)
                            img = img.convert('RGB')
                            img = img.resize(target_size)
                            img_array = np.array(img)
                            images.append(img_array)
                            labels.append(bug_type)
    
    return np.array(images), np.array(labels)

def get_data_loaders(X_train, X_test, y_train, y_test, batch_size=32):
    """Creates and returns train and test data loaders"""
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = StoneflyDataset(X_train, y_train, transform=train_transform)
    test_dataset = StoneflyDataset(X_test, y_test, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader 