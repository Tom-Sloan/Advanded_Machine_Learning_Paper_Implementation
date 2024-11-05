import numpy as np
from torchvision import transforms
from tqdm import tqdm

def augment_data(images, labels, augmentations_per_image=2):
    """
    Augments the training data by creating multiple versions of each image with transformations.
    
    Args:
        images: Numpy array of images
        labels: Numpy array of corresponding labels 
        augmentations_per_image: Number of augmented versions to create per original image
        
    Returns:
        Tuple of augmented images and labels arrays
    """
    augmented_images = []
    augmented_labels = []
    
    augment_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    augmented_images.extend(list(images))
    augmented_labels.extend(list(labels))
    
    for i in tqdm(range(len(images)), desc="Augmenting images"):
        img = images[i]
        label = labels[i]
        
        for _ in range(augmentations_per_image):
            aug_img = augment_transforms(img)
            aug_img = np.array(aug_img)
            
            augmented_images.append(aug_img)
            augmented_labels.append(label)
            
    return np.array(augmented_images), np.array(augmented_labels) 