import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt

from models.ddpm import create_diffusion_model
from data_management.dataset import get_dataset
from training.trainer import Trainer

class DDPMTrainer(Trainer):
    def __init__(self, model, criterion, optimizer, device, num_timesteps=1000,
                 warmup_scheduler=None, main_scheduler=None, grad_clip_value=None):
        super().__init__(model, criterion, optimizer, device, 
                        warmup_scheduler, main_scheduler, grad_clip_value)
        self.num_timesteps = num_timesteps
        
        # Define beta schedule
        self.beta = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def train_epoch(self, train_loader, is_warmup=False):
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, images in enumerate(tqdm(train_loader, desc="Training")):
            if isinstance(images, (tuple, list)):
                images = images[0]  # Some datasets return (image, label)
            
            images = images.to(self.device)
            batch_size = images.shape[0]
            
            # Sample t uniformly
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)
            
            # Get noise and noisy image
            noise = torch.randn_like(images)
            noisy_images = self.q_sample(images, t, noise)
            
            # Predict noise
            self.optimizer.zero_grad()
            predicted_noise = self.model(noisy_images, t)
            loss = self.criterion(predicted_noise, noise)
            
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            
            self.optimizer.step()
            
            if is_warmup and self.warmup_scheduler:
                self.warmup_scheduler.step()
            elif not is_warmup and self.main_scheduler:
                self.main_scheduler.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f'Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        return running_loss / len(train_loader)
    
    def q_sample(self, x_0, t, noise):
        # Sample from q(x_t | x_0)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    @torch.no_grad()
    def p_sample(self, x_t, t):
        # Sample from p(x_{t-1} | x_t)
        beta_t = self.beta[t].reshape(-1, 1, 1, 1)
        alpha_t = self.alpha[t].reshape(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].reshape(-1, 1, 1, 1)
        
        # Predict noise
        predicted_noise = self.model(x_t, t)
        
        # Get mean
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        # Add noise if t > 0
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(beta_t) * noise
            mean = mean + variance
        
        return mean
    
    @torch.no_grad()
    def sample(self, batch_size=16, channels=3, size=256):
        # Start from pure noise
        x = torch.randn(batch_size, channels, size, size).to(self.device)
        
        # Gradually denoise
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.tensor([t], device=self.device).repeat(batch_size)
            x = self.p_sample(x, t_batch)
        
        return x

def save_samples(trainer, epoch, save_dir='samples/ddpm'):
    """Save sample generations from the model"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate samples
    samples = trainer.sample(batch_size=4)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
        ax.axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch}.png'))
    plt.close()

def train(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_diffusion_model().to(device)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Setup loss
    criterion = nn.MSELoss()
    
    # Create trainer
    trainer = DDPMTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        grad_clip_value=1.0
    )
    
    # Load dataset
    dataset = get_dataset('celeba-hq')
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        avg_loss = trainer.train_epoch(train_loader)
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Save samples every 5 epochs
        if epoch % 5 == 0:
            save_samples(trainer, epoch)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            trainer.save_model(f'trained_models/ddpm/model_epoch_{epoch+1}.pt')

def main():
    parser = argparse.ArgumentParser(description='Train DDPM model')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs('trained_models/ddpm', exist_ok=True)
    os.makedirs('samples/ddpm', exist_ok=True)
    
    train(args)

if __name__ == '__main__':
    main() 