import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor
import math
from typing import Dict, List, Optional, Union

class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""
    def __init__(
        self, 
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        
        # Initialize LoRA matrices
        self.lora_a = nn.Linear(linear_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear_layer.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.lora_a.weight, mean=0, std=1/rank)
        nn.init.zeros_(self.lora_b.weight)
        
        # Freeze original layer
        for param in self.linear.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Original frozen path
        orig_out = self.linear(x)
        
        # LoRA path
        lora_out = self.lora_b(self.dropout(self.lora_a(x)))
        
        # Combine with scaling
        return orig_out + (self.alpha / self.rank) * lora_out

def create_lora_model(
    model_name: str = "google/vit-base-patch16-224-in21k",
    num_classes: int = None,
    rank: int = 8,
    alpha: int = 16,
    target_modules: List[str] = ["query", "value"],
    dropout: float = 0.1,
    cache_dir: Optional[str] = None
) -> nn.Module:
    """Creates a ViT model with LoRA layers"""
    
    # Load base model
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes if num_classes else 1000,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir
    )
    
    # Add LoRA layers
    model = add_lora_layers(
        model,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
        dropout=dropout
    )
    
    return model

def add_lora_layers(
    model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    target_modules: List[str] = ["query", "value"],
    dropout: float = 0.1
) -> nn.Module:
    """Adds LoRA layers to target modules in the model"""
    
    for name, module in model.named_modules():
        # Check if current module name contains any target module name
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Replace with LoRA version
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, child_name, LoRALinear(
                    module,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                ))
    
    return model

def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Gets all trainable LoRA parameters"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_a.weight, module.lora_b.weight])
    return params

def evaluate_and_compare(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    processor: AutoImageProcessor,
    id2label: Dict[int, str]
) -> None:
    """Evaluates and compares original vs LoRA model predictions"""
    model.eval()
    all_preds = []
    all_orig_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Get predictions
            outputs = model(inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            # Get original model predictions (without LoRA influence)
            # Temporarily zero out LoRA contributions
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.alpha = 0
            orig_outputs = model(inputs)
            orig_logits = orig_outputs.logits
            orig_preds = torch.argmax(orig_logits, dim=1)
            # Restore LoRA contributions
            for module in model.modules():
                if isinstance(module, LoRALinear):
                    module.alpha = alpha
            
            # Store predictions
            all_preds.extend(preds.cpu().numpy())
            all_orig_preds.extend(orig_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print comparison
    print("\nPrediction Comparison (Original/LoRA/True):")
    print("-" * 50)
    for orig_pred, lora_pred, true_label in zip(all_orig_preds, all_preds, all_labels):
        print(f"{id2label[orig_pred]}/{id2label[lora_pred]}/{id2label[true_label]}")
    
    # Calculate accuracies
    orig_correct = sum(p == l for p, l in zip(all_orig_preds, all_labels))
    lora_correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels)
    
    print("\nAccuracy Comparison:")
    print(f"Original Model: {100 * orig_correct / total:.2f}%")
    print(f"LoRA Model: {100 * lora_correct / total:.2f}%") 