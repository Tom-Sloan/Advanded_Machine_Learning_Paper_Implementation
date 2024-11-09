import torch
import torch.nn as nn
import math
from transformers import AutoModelForImageClassification
import os

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    def __init__(self, in_features, out_features, rank=4, alpha=1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Store original layer
        self.original_layer = None
    
    def forward(self, x):
        # Low-rank adaptation
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        if self.original_layer is not None:
            return self.original_layer(x) + lora_output
        return lora_output

def create_lora_model(model_name="google/vit-large-patch16-224", num_classes=None, cache_dir="./models/pretrained"):
    """Create a pretrained model with LoRA layers"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Using cache directory: {cache_dir}")
    
    # Load pretrained model
    print(f"Loading pretrained model: {model_name}")
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir
    )
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded and cached in: {cache_dir}")
    return model

def add_lora_layers(model, rank=4, alpha=1, target_modules=None):
    """Add LoRA layers to a model's target modules"""
    # Default target modules for transformer models
    if target_modules is None:
        target_modules = [
            "query",
            "key",
            "value",
            "output.dense",
            "intermediate.dense"
        ]
    
    # Keep track of added LoRA layers
    lora_layers = []
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Create LoRA layer
                lora = LoRALayer(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha
                )
                
                # Store original layer
                lora.original_layer = module
                
                # Replace the original layer with LoRA
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = model
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)
                setattr(parent_module, child_name, lora)
                
                # Track the LoRA layer
                lora_layers.append(lora)
                print(f"Added LoRA to {name}")
    
    if not lora_layers:
        print("Warning: No LoRA layers were added. Check target_modules parameter.")
        print("Available modules:")
        for name, _ in model.named_modules():
            print(f"  {name}")
    else:
        print(f"Added {len(lora_layers)} LoRA layers")
    
    # Store LoRA layers in model for easy access
    model.lora_layers = lora_layers
    return model

def get_lora_params(model):
    """Get only LoRA parameters for training"""
    if not hasattr(model, 'lora_layers'):
        raise ValueError("No LoRA layers found in model. Run add_lora_layers first.")
    
    params = []
    for layer in model.lora_layers:
        params.extend([layer.lora_A, layer.lora_B])
    
    if not params:
        raise ValueError("No LoRA parameters found!")
    
    return params