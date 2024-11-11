import torch
from typing import List, Dict, Optional
import numpy as np
from transformers import DPRContextEncoder
import faiss
import pickle
import os
from pathlib import Path
from .document_store import DocumentStore

class CodeDocumentStore(DocumentStore):
    """Document store specialized for code files and documentation"""
    
    def load_codebase(self, root_dir: str, file_extensions: List[str] = ['.py', '.md']):
        """Load code files from the codebase"""
        documents = []
        doc_ids = []
        
        # Walk through directory
        for root, _, files in os.walk(root_dir):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, root_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # Split content into chunks with overlap
                        chunks = self.chunk_code_content(content, file_path)
                        
                        # Add each chunk as a document
                        for i, chunk in enumerate(chunks):
                            documents.append(chunk)
                            doc_ids.append(f"{rel_path}#chunk{i}")
                            
                    except Exception as e:
                        print(f"Error loading {file_path}: {str(e)}")
        
        # Add documents to store
        self.add_documents(documents, doc_ids)
        print(f"Loaded {len(documents)} chunks from {len(set(doc_id.split('#')[0] for doc_id in doc_ids))} files")
        
    def chunk_code_content(self, content: str, file_path: str, chunk_size: int = 1500, overlap: int = 500) -> List[str]:
        """Split code/text content into overlapping chunks"""
        chunks = []
        
        # Handle markdown files differently
        if file_path.endswith('.md'):
            # Split on headers and keep context
            sections = content.split('\n## ')
            for section in sections:
                if section.strip():
                    chunks.append(section)
                    
        else:  # Python files
            # Split on class/function definitions while maintaining context
            lines = content.split('\n')
            current_chunk = []
            current_class = None
            current_func = None
            
            for line in lines:
                # Track class and function context
                if line.startswith('class '):
                    current_class = line
                    if current_chunk:  # Save previous chunk
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [f"File: {file_path}", line]
                elif line.startswith('def '):
                    current_func = line
                    if current_class:  # Keep class context for methods
                        if current_chunk and current_chunk[0] != current_class:
                            chunks.append('\n'.join(current_chunk))
                        current_chunk = [current_class, line]
                    else:
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                        current_chunk = [f"File: {file_path}", line]
                else:
                    current_chunk.append(line)
                    
                # Split if chunk gets too large
                if len('\n'.join(current_chunk)) >= chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    # Keep context for next chunk
                    current_chunk = []
                    if current_class:
                        current_chunk.append(current_class)
                    if current_func:
                        current_chunk.append(current_func)
            
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def search_code(self, query: str, k: int = 5) -> Dict:
        """Search codebase with a natural language query"""
        # Tokenize and encode query
        inputs = self.tokenizer(
            query,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = self.encoder(**inputs).pooler_output.numpy()
        
        # Search
        results = self.search(query_embedding, k)
        
        # Format results
        formatted_results = []
        for doc, doc_id, score in zip(results['documents'], results['doc_ids'], results['scores']):
            file_path, chunk_id = doc_id.split('#')
            formatted_results.append({
                'file': file_path,
                'chunk_id': chunk_id,
                'content': doc,
                'score': float(score)
            })
        
        return formatted_results