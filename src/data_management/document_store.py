import torch
from typing import List, Dict, Optional
import numpy as np
from transformers import DPRContextEncoder, AutoTokenizer
import faiss
import pickle
import os

class DocumentStore:
    """Store and index documents for efficient retrieval"""
    def __init__(self, 
                 encoder_name: str = "facebook/dpr-ctx_encoder-single-nq-base",
                 dimension: int = 768,
                 index_type: str = "Flat"):
        self.encoder = DPRContextEncoder.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.dimension = dimension
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == "Flat":
            self.index = faiss.IndexFlatIP(dimension)  # Inner product similarity
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        self.documents = []
        self.doc_ids = []
        
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None):
        """Add documents to the store and index their embeddings"""
        # Generate IDs if not provided
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(self.documents), 
                                           len(self.documents) + len(documents))]
        
        # Encode documents
        embeddings = []
        for doc in documents:
            # Tokenize and encode
            inputs = self.tokenizer(
                doc,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                embedding = self.encoder(**inputs).pooler_output
                embeddings.append(embedding.numpy().squeeze())
        
        embeddings = np.vstack(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and IDs
        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)
        
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Dict:
        """Search for similar documents using query embedding"""
        # Perform search
        scores, indices = self.index.search(query_embedding, k)
        
        # Get retrieved documents
        retrieved_docs = [self.documents[idx] for idx in indices[0]]
        retrieved_ids = [self.doc_ids[idx] for idx in indices[0]]
        
        return {
            'documents': retrieved_docs,
            'doc_ids': retrieved_ids,
            'scores': scores[0]
        }
    
    def save(self, path: str):
        """Save the document store to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        
        # Save documents and metadata
        metadata = {
            'documents': self.documents,
            'doc_ids': self.doc_ids,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        
        with open(f"{path}.meta", 'wb') as f:
            pickle.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str):
        """Load a document store from disk"""
        # Load metadata
        with open(f"{path}.meta", 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(dimension=metadata['dimension'],
                      index_type=metadata['index_type'])
        
        # Load FAISS index
        instance.index = faiss.read_index(f"{path}.faiss")
        
        # Restore documents and IDs
        instance.documents = metadata['documents']
        instance.doc_ids = metadata['doc_ids']
        
        return instance 