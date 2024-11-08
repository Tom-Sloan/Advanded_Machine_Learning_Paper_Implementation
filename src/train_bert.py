import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
import os
from collections import Counter
import numpy as np
import tiktoken

from data_management.dataset import get_dataset
from models.bert import create_bert_base
from training.trainer import Trainer

def ensure_model_dir(model_path):
    """Ensure the model directory exists"""
    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

def get_model_path():
    """Get the standard model path"""
    return os.path.join('./trained_models', 'bert_shakespeare/best_model.pt')

def get_tokenizer():
    """Get the GPT-2 tokenizer"""
    return tiktoken.get_encoding("gpt2")

class BERTDataset(Dataset):
    def __init__(self, sequences, max_len=64):
        self.sequences = sequences
        self.max_len = max_len
        self.tokenizer = get_tokenizer()
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Encode text using tiktoken
        token_ids = self.tokenizer.encode(self.sequences[idx])[:self.max_len]
        
        # Pad sequence
        padding_length = self.max_len - len(token_ids)
        token_ids = token_ids + [0] * padding_length  # Use 0 as padding token
        attention_mask = [1] * len(token_ids) + [0] * padding_length
        
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long)
        )

class BERTLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    def forward(self, outputs, targets, attention_mask=None):
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = outputs.view(-1, outputs.size(-1))[active_loss]
            active_labels = targets.view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
        else:
            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        return loss

def create_mlm_data(tokens, attention_mask, vocab_size, tokenizer, mask_prob=0.15):
    """Create masked language modeling data with GPT-2 tokenizer"""
    masked_tokens = tokens.clone()
    targets = tokens.clone()
    
    # Only consider tokens that are not padding
    active_tokens = attention_mask == 1
    
    # Create probability matrix for masking
    prob_matrix = torch.rand(tokens.shape, device=tokens.device)
    prob_matrix = prob_matrix * active_tokens
    
    # Create mask
    mask = (prob_matrix < mask_prob) * active_tokens
    
    # For GPT-2 tokenizer, use specific tokens for masking
    # Using common words from GPT-2 vocabulary for masking
    mask_token_options = [
        tokens.new_tensor([tokenizer.encode(" mask")[0]]),  # space + "mask"
        tokens.new_tensor([tokenizer.encode(" [MASK]")[0]]),  # space + "[MASK]"
        tokens.new_tensor([tokenizer.encode(" ___")[0]])  # space + "___"
    ]
    
    # Randomly choose mask tokens
    for i in range(mask.size(0)):
        for j in range(mask.size(1)):
            if mask[i, j]:
                masked_tokens[i, j] = mask_token_options[torch.randint(0, len(mask_token_options), (1,))].item()
    
    # Set targets for non-masked tokens to -100 (ignored by loss)
    targets[~mask] = -100
    
    return masked_tokens, targets

def build_vocab(text, min_freq=2):
    """Build vocabulary from text with better handling of rare words"""
    words = text.split()
    word_counts = Counter(words)
    
    # Create vocabulary with special tokens
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<MASK>': 2,
        '<CLS>': 3,
        '<SEP>': 4,
    }
    
    # Add all words that appear more than min_freq times
    # Sort by frequency for consistent indexing
    idx = len(vocab)
    for word, count in sorted(word_counts.items(), key=lambda x: (-x[1], x[0])):
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
            
    print(f"Vocabulary size: {len(vocab)} words")
    print(f"Most common words: {list(vocab.keys())[5:15]}")  # Print some common words
    return vocab

def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Use standard model path
    args.model_path = get_model_path()
    ensure_model_dir(args.model_path)
    
    # Load dataset
    dataset = get_dataset('shakespeare_word', args.data_dir)
    text = ' '.join(dataset.data)
    
    # Get tokenizer and vocabulary size
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab + 1  # Add 1 for padding token
    
    # Create sequences with overlap for better context
    seq_length = 64
    stride = 32
    sequences = []
    
    # Split text into overlapping sequences
    words = text.split()
    for i in range(0, len(words) - seq_length, stride):
        sequence = ' '.join(words[i:i+seq_length])
        sequences.append(sequence)
    
    # Split into train/val
    train_size = int(0.9 * len(sequences))
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]
    
    # Create datasets
    train_dataset = BERTDataset(train_sequences)
    val_dataset = BERTDataset(val_sequences)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model and optimizer with weight decay
    model = create_bert_base(vocab_size).to(device)
    
    # Use different learning rates for different parts of the model
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    # Use cosine learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = total_steps // 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=warmup_steps/total_steps,
        anneal_strategy='cos'
    )
    
    criterion = BERTLoss()
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        model.train()
        train_loss = 0
        for batch_tokens, attention_mask in tqdm(train_loader, desc="Training"):
            batch_tokens = batch_tokens.to(device)
            attention_mask = attention_mask.to(device)
            
            masked_tokens, targets = create_mlm_data(batch_tokens, attention_mask, vocab_size, tokenizer)
            
            optimizer.zero_grad()
            output = model(masked_tokens, attention_mask)
            loss = criterion(output, targets, attention_mask)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_tokens, attention_mask in tqdm(val_loader, desc="Validating"):
                batch_tokens = batch_tokens.to(device)
                attention_mask = attention_mask.to(device)
                
                masked_tokens, targets = create_mlm_data(batch_tokens, attention_mask, vocab_size, tokenizer)
                output = model(masked_tokens, attention_mask)
                loss = criterion(output, targets, attention_mask)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_val_loss,
                'epoch': epoch,
                'vocab_size': vocab_size
            }, args.model_path)
            print("New best model saved!")

def evaluate(args):
    # Use standard model path
    args.model_path = get_model_path()
    
    if not os.path.exists(args.model_path):
        print(f"No saved model found at {args.model_path}")
        return
    
    # Load dataset and model
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.n_vocab + 1
    
    # Create model and load weights
    model = create_bert_base(vocab_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load some test sequences
    dataset = get_dataset('shakespeare_word', args.data_dir)
    text = ' '.join(dataset.data)
    words = text.split()
    seq_length = 64
    sequences = [' '.join(words[i:i+seq_length]) for i in range(0, len(words)-seq_length, seq_length)]
    test_dataset = BERTDataset(sequences[-100:])  # Use last 100 sequences for testing
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    print("\nExample predictions:")
    with torch.no_grad():
        for i, (tokens, attention_mask) in enumerate(test_loader):
            if i >= 5:  # Show 5 examples
                break
                
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            
            # Get original text
            original = tokenizer.decode(tokens[0].cpu().tolist())
            
            # Create masked input
            masked_tokens, _ = create_mlm_data(tokens, attention_mask, vocab_size, tokenizer, mask_prob=0.2)
            masked_text = tokenizer.decode(masked_tokens[0].cpu().tolist())
            
            # Get model predictions
            output = model(masked_tokens, attention_mask)
            predictions = torch.argmax(output, dim=-1)
            
            # For each masked position, use the model's prediction
            final_tokens = tokens.clone()
            mask = (masked_tokens != tokens) & (attention_mask == 1)
            final_tokens[mask] = predictions[mask]
            
            predicted_text = tokenizer.decode(final_tokens[0].cpu().tolist())
            
            print(f'\nExample {i+1}:')
            print(f'Original:  {original}')
            print(f'Masked:    {masked_text}')
            print(f'Predicted: {predicted_text}')

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate BERT model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                      help='train or eval mode')
    parser.add_argument('--data_dir', type=str, default='./data/shakespeare_word',
                      help='path to dataset')
    parser.add_argument('--epochs', type=int, default=20,
                      help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='learning rate')
    parser.add_argument('--min_freq', type=int, default=5,
                      help='minimum word frequency for vocabulary')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == "__main__":
    main() 