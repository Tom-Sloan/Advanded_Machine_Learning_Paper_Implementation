import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
import os
import inquirer
from pathlib import Path
import time

from models.rag import RAG, CacheConfig, StreamingConfig
from data_management.code_document_store import CodeDocumentStore
from training.trainer import Trainer

class RAGTrainer(Trainer):
    """Custom trainer for RAG model"""
    def __init__(self, model, criterion, optimizer, device, doc_store,
                 warmup_scheduler=None, main_scheduler=None, grad_clip_value=None):
        super().__init__(model, criterion, optimizer, device,
                        warmup_scheduler, main_scheduler, grad_clip_value)
        self.doc_store = doc_store
        
    def train_epoch(self, train_loader, is_warmup=False):
        """Interactive training epoch for RAG"""
        self.model.train()
        total_loss = 0
        num_interactions = 0
        
        print("\nStarting interactive training session...")
        print("Ask questions about the codebase (type 'exit' to end epoch)")
        
        while True:
            # Get question from user
            question = input("\nQuestion: ").strip()
            if question.lower() == 'exit':
                break
                
            # Process question
            results = self.doc_store.search_code(question)
            
            # Generate answer
            context_embeddings = torch.tensor(self.doc_store.index.reconstruct_n(0, 
                self.doc_store.index.ntotal)).to(self.device)
            
            outputs = self.model([question], context_embeddings, 
                               [doc['content'] for doc in results])
            
            # Show answer
            print("\nGenerated Answer:")
            print("-" * 80)
            print(outputs['answers'][0])
            
            # Get feedback
            feedback = inquirer.prompt([
                inquirer.List('quality',
                            message="How was the response quality?",
                            choices=['Good', 'Fair', 'Poor'])
            ])
            
            # Convert feedback to loss
            quality_scores = {'Good': 0.1, 'Fair': 0.5, 'Poor': 1.0}
            loss = torch.tensor(quality_scores[feedback['quality']], 
                              device=self.device)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                             self.grad_clip_value)
            
            self.optimizer.step()
            
            # Update schedulers
            if is_warmup and self.warmup_scheduler:
                self.warmup_scheduler.step()
            elif not is_warmup and self.main_scheduler:
                self.main_scheduler.step()
            
            total_loss += loss.item()
            num_interactions += 1
        
        return total_loss / max(1, num_interactions)
    
    def evaluate(self, val_loader):
        """Interactive evaluation for RAG"""
        self.model.eval()
        total_score = 0
        num_interactions = 0
        
        print("\nStarting evaluation session...")
        print("Ask questions to evaluate the model (type 'exit' to end)")
        
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() == 'exit':
                break
                
            # Process question
            results = self.doc_store.search_code(question)
            
            # Generate answer
            with torch.no_grad():
                context_embeddings = torch.tensor(self.doc_store.index.reconstruct_n(0, 
                    self.doc_store.index.ntotal)).to(self.device)
                
                outputs = self.model([question], context_embeddings, 
                                   [doc['content'] for doc in results])
                
                print("\nGenerated Answer:")
                print("-" * 80)
                print(outputs['answers'][0])
                
                # Get feedback
                feedback = inquirer.prompt([
                    inquirer.List('quality',
                                message="How was the response quality?",
                                choices=['Good', 'Fair', 'Poor'])
                ])
                
                # Convert feedback to score
                quality_scores = {'Good': 1.0, 'Fair': 0.5, 'Poor': 0.0}
                score = quality_scores[feedback['quality']]
                
                total_score += score
                num_interactions += 1
        
        final_score = total_score / max(1, num_interactions)
        is_best = final_score > self.best_acc
        if is_best:
            self.best_acc = final_score
            
        return final_score, is_best, None

def train(args):
    """Train the RAG model"""
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Initialize document store
    doc_store = CodeDocumentStore()
    doc_store.load_codebase(args.root_dir)
    
    # Initialize model
    model = RAG(
        cache_config=CacheConfig(),
        streaming_config=StreamingConfig()
    ).to(device)
    
    # Setup optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()  # Not directly used but needed for trainer interface
    
    # Create trainer
    trainer = RAGTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        doc_store=doc_store,
        grad_clip_value=1.0
    )
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        avg_loss = trainer.train_epoch(None)  # No dataloader needed
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            trainer.save_model(f'{args.output_dir}/model_epoch_{epoch+1}.pt')

def main():
    parser = argparse.ArgumentParser(description='Train or evaluate RAG model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    parser.add_argument('--root_dir', type=str, default='./src')
    parser.add_argument('--output_dir', type=str, default='./trained_models/rag')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--save_interval', type=int, default=1)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)

if __name__ == '__main__':
    main() 