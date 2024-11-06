import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(self, model, criterion, optimizer, device, 
                 warmup_scheduler=None, main_scheduler=None, grad_clip_value=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.grad_clip_value = grad_clip_value
        self.best_acc = 0.0
        
    def train_epoch(self, train_loader, is_warmup=False):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            inputs = inputs.to(self.device)
            
            if isinstance(labels, tuple):
                label1, label2, lam = labels
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)
                lam = lam.to(self.device)
                
                outputs = self.model(inputs)
                loss = lam.mean() * self.criterion(outputs, label1) + (1 - lam.mean()) * self.criterion(outputs, label2)
                _, predicted = outputs.max(1)
                total += label1.size(0)
                correct += (lam.mean() * predicted.eq(label1).sum().float() + 
                          (1 - lam.mean()) * predicted.eq(label2).sum().float()).item()
            else:
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.grad_clip_value is not None:
                clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                
            self.optimizer.step()
            
            if is_warmup and self.warmup_scheduler:
                self.warmup_scheduler.step()
            elif not is_warmup and self.main_scheduler:
                self.main_scheduler.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/20:.4f}, '
                      f'Acc: {100.*correct/total:.2f}%, '
                      f'LR: {current_lr:.6f}')
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
        
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            return accuracy, True, (all_predictions, all_labels)
        return accuracy, False, (all_predictions, all_labels)
    
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_acc': self.best_acc,
        }
        
        torch.save(checkpoint, save_path)
    
    @staticmethod
    def load_model(model, load_path, device):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint['best_acc']