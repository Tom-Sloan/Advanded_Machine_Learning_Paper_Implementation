import warnings
warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.pix2pix import create_pix2pix_models
from data_management.dataset import get_dataset, get_training_transforms, get_eval_transforms

from contextlib import nullcontext

class Pix2PixDataset(Dataset):
    def __init__(self, input_images, target_images, transform=None):
        self.input_images = input_images
        self.target_images = target_images
        self.transform = transform
        
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):
        input_img = self.input_images[idx]
        target_img = self.target_images[idx]
        
        if self.transform:
            # Apply same transform to both images
            seed = np.random.randint(2147483647)
            
            torch.manual_seed(seed)
            input_img = self.transform(input_img)
            
            torch.manual_seed(seed)
            target_img = self.transform(target_img)
            
        return input_img, target_img

def save_samples(generator, dataset, epoch, device, save_dir='samples/pix2pix'):
    """Save sample generations from the model"""
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    
    with torch.no_grad():
        # Get a few samples from the dataset
        test_loader = DataLoader(dataset, batch_size=4, shuffle=True)
        input_images, target_images = next(iter(test_loader))
        input_images = input_images.to(device)
        
        # Generate fake images
        fake_images = generator(input_images)
        
        # Create figure
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        fig.suptitle(f'Epoch {epoch}')
        
        # Plot each sample
        for i in range(4):
            # Input semantic map
            axes[0, i].imshow(input_images[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
            axes[0, i].set_title('Input')
            axes[0, i].axis('off')
            
            # Generated image
            axes[1, i].imshow(fake_images[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
            axes[1, i].set_title('Generated')
            axes[1, i].axis('off')
            
            # Target image
            axes[2, i].imshow(target_images[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
            axes[2, i].set_title('Target')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
        plt.close()
    
    generator.train()

def train(args):
    # Check MPS availability and settings
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == "mps":
        print("MPS device properties:")
        print(f"MPS backend enabled: {torch.backends.mps.is_available()}")
        print(f"MPS device built: {torch.backends.mps.is_built()}")
        # Force sync to ensure MPS operations complete
        torch.mps.synchronize()
    
    # Create models and explicitly move to device
    generator, discriminator = create_pix2pix_models()
    
    # Initialize optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    
    # Load checkpoint if specified
    start_epoch = 0
    if hasattr(args, 'checkpoint') and args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Loss functions - move to device
    gan_loss = nn.BCEWithLogitsLoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    
    # Load and prepare data with pin_memory for faster transfer
    dataset = get_dataset('cityscapes', args.data_dir)
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True if device.type != "cpu" else False,
        num_workers=2  # Increase for faster data loading
    )
    
    # Create samples directory
    os.makedirs('samples/pix2pix', exist_ok=True)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        generator.train()
        discriminator.train()
        
        for i, (real_A, real_B) in enumerate(tqdm(train_loader)):
            # Ensure tensors are on the correct device and in the right format
            real_A = real_A.to(device, non_blocking=True)
            real_B = real_B.to(device, non_blocking=True)
            
            # Train Discriminator
            d_optimizer.zero_grad(set_to_none=True)  # Slightly more efficient
            
            # Generate fake image
            with torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext():
                fake_B = generator(real_A)
                fake_AB = torch.cat([real_A, fake_B], 1)
                real_AB = torch.cat([real_A, real_B], 1)
                
                pred_fake = discriminator(fake_AB.detach())
                pred_real = discriminator(real_AB)
                
                loss_d_fake = gan_loss(pred_fake, torch.zeros_like(pred_fake))
                loss_d_real = gan_loss(pred_real, torch.ones_like(pred_real))
                loss_d = (loss_d_fake + loss_d_real) * 0.5
            
            loss_d.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext():
                pred_fake = discriminator(fake_AB)
                loss_g_gan = gan_loss(pred_fake, torch.ones_like(pred_fake))
                loss_g_l1 = l1_loss(fake_B, real_B) * args.lambda_l1
                loss_g = loss_g_gan + loss_g_l1
            
            loss_g.backward()
            g_optimizer.step()
            
            # Force MPS synchronization periodically
            if device.type == "mps" and i % 10 == 0:
                torch.mps.synchronize()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{args.epochs}], Step [{i}/{len(train_loader)}]')
                print(f'Discriminator loss: {loss_d.item():.4f}, Generator loss: {loss_g.item():.4f}')
        
        # Save samples every 5 epochs
        if epoch % 5 == 0:
            save_samples(generator, dataset, epoch, device)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            # Ensure synchronization before saving
            if device.type == "mps":
                torch.mps.synchronize()
            
            save_path = f'trained_models/pix2pix/model_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'g_optimizer_state_dict': g_optimizer.state_dict(),
                'd_optimizer_state_dict': d_optimizer.state_dict(),
                'g_loss': loss_g.item(),
                'd_loss': loss_d.item(),
            }, save_path)
            print(f"Saved checkpoint to {save_path}")

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    generator, _ = create_pix2pix_models()
    checkpoint = torch.load('trained_models/pix2pix/model_latest.pt')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator = generator.to(device)
    generator.eval()
    
    # Load test data
    dataset = get_dataset('cityscapes', args.data_dir, split='test')
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Generate and display results
    with torch.no_grad():
        for i, (input_images, target_images) in enumerate(test_loader):
            if i >= args.num_samples:
                break
                
            input_images = input_images.to(device)
            fake_images = generator(input_images)
            
            # Display results
            fig, axs = plt.subplots(3, min(args.batch_size, 5), figsize=(15, 9))
            for j in range(min(args.batch_size, 5)):
                axs[0, j].imshow(input_images[j].cpu().numpy().transpose(1, 2, 0))
                axs[0, j].set_title('Input')
                axs[0, j].axis('off')
                
                axs[1, j].imshow(fake_images[j].cpu().numpy().transpose(1, 2, 0))
                axs[1, j].set_title('Generated')
                axs[1, j].axis('off')
                
                axs[2, j].imshow(target_images[j].cpu().numpy().transpose(1, 2, 0))
                axs[2, j].set_title('Target')
                axs[2, j].axis('off')
            
            plt.tight_layout()
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate Pix2Pix model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--data_dir', type=str, default='./data/cityscapes')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('trained_models/pix2pix', exist_ok=True)
    os.makedirs('samples/pix2pix', exist_ok=True)
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == '__main__':
    main() 