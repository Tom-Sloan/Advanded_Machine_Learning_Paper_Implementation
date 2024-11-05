import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_acc = 0.0
        
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/20:.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
                
        return correct / total
    
    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = correct / total
        
        # Update best accuracy
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            return accuracy, True, (all_predictions, all_labels)
        return accuracy, False, (all_predictions, all_labels)
    
    def save_model(self, save_path):
        """Save best model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
        }
        
        torch.save(checkpoint, save_path)
    
    @staticmethod
    def load_model(model, load_path, device):
        """Load a saved model"""
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['best_acc']