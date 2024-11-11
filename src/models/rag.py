import torch
import torch.nn as nn
from transformers import DPRQuestionEncoder, DPRContextEncoder, LlamaForCausalLM, LlamaTokenizer
from typing import List, Dict, Generator, Optional
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
from dataclasses import dataclass
from collections import OrderedDict
import bitsandbytes

@dataclass
class StreamingConfig:
    """Configuration for streaming parameters"""
    chunk_size: int = 1
    min_delay: float = 0.01
    max_delay: float = 0.1
    adaptive_speed: bool = True
    
@dataclass
class CacheConfig:
    """Configuration for cache management"""
    max_size: int = 1000
    ttl: int = 7 * 24 * 60 * 60  # 7 days in seconds
    cleanup_interval: int = 60 * 60  # Cleanup every hour
    
class LRUCache(OrderedDict):
    """Least Recently Used (LRU) cache implementation"""
    def __init__(self, max_size=1000):
        super().__init__()
        self.max_size = max_size
        
    def get(self, key):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return None
        
    def put(self, key, value):
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.max_size:
            self.popitem(last=False)

class RAG(nn.Module):
    def __init__(self, 
                 question_encoder_name="facebook/dpr-question_encoder-single-nq-base",
                 context_encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
                 llama_model_name="meta-llama/Llama-3.2-3B-Instruct",
                 n_docs=5,
                 max_length=512,
                 cache_dir="./cache/rag",
                 cache_config: Optional[CacheConfig] = None,
                 streaming_config: Optional[StreamingConfig] = None,
                 num_workers: int = 4):
        super().__init__()
        
        # Initialize components
        self.question_encoder = DPRQuestionEncoder.from_pretrained(question_encoder_name)
        self.context_encoder = DPRContextEncoder.from_pretrained(context_encoder_name)
        
        # Try to use 4-bit quantization if available, otherwise fall back to 16-bit
        try:
            print("Using 4-bit quantization with bitsandbytes")
            self.generator = LlamaForCausalLM.from_pretrained(
                llama_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except ImportError:
            print("bitsandbytes not available")
        
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_name)
        
        # Configurations
        self.n_docs = n_docs
        self.max_length = max_length
        self.num_workers = num_workers
        self.cache_config = cache_config or CacheConfig()
        self.streaming_config = streaming_config or StreamingConfig()
        
        # Setup caching
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.response_cache_file = self.cache_dir / "response_cache.json"
        self.response_cache = LRUCache(max_size=self.cache_config.max_size)
        self.last_cleanup = time.time()
        self.load_cache()
        
    def cleanup_cache(self):
        """Remove expired cache entries"""
        current_time = time.time()
        
        # Check if cleanup is needed
        if current_time - self.last_cleanup < self.cache_config.cleanup_interval:
            return
            
        # Remove expired entries
        expired_keys = [
            key for key, value in self.response_cache.items()
            if current_time - value['timestamp'] > self.cache_config.ttl
        ]
        
        for key in expired_keys:
            del self.response_cache[key]
            
        self.last_cleanup = current_time
        self.save_cache()
        
        logging.info(f"Cache cleanup: removed {len(expired_keys)} expired entries")
        
    async def parallel_retrieve(self, questions, context_embeddings, context_texts):
        """Retrieve documents in parallel using async/await"""
        async def process_batch(batch_questions):
            # Encode questions
            with torch.no_grad():
                question_embeddings = self.question_encoder(batch_questions).pooler_output
                
            # Calculate similarity scores
            scores = torch.matmul(question_embeddings, context_embeddings.transpose(0, 1))
            
            # Get top-k documents
            top_k_scores, top_k_indices = scores.topk(self.n_docs, dim=1)
            
            # Gather documents
            batch_docs = []
            for indices in top_k_indices:
                docs = [context_texts[idx] for idx in indices]
                batch_docs.append(docs)
                
            return batch_docs, top_k_scores
        
        # Split questions into batches
        batch_size = 4
        batches = [questions[i:i + batch_size] for i in range(0, len(questions), batch_size)]
        
        # Process batches in parallel
        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        all_docs = []
        all_scores = []
        for docs, scores in results:
            all_docs.extend(docs)
            all_scores.append(scores)
            
        return all_docs, torch.cat(all_scores)
        
    def get_adaptive_delay(self, chunk: str) -> float:
        """Calculate adaptive delay based on content"""
        config = self.streaming_config
        
        # Slow down for code blocks and punctuation
        if any(char in chunk for char in '{}[]().,:;'):
            return config.max_delay
        # Slow down for newlines
        elif '\n' in chunk:
            return config.max_delay * 0.75
        # Normal speed for regular text
        else:
            return config.min_delay
            
    def generate_stream(self, 
                       questions: List[str], 
                       retrieved_docs: List[List[str]], 
                       use_cache: bool = True) -> Generator[Dict, None, None]:
        """Stream generated answers with enhanced control"""
        # Cleanup cache if needed
        if use_cache:
            self.cleanup_cache()
            
        for question, docs in zip(questions, retrieved_docs):
            # Format context
            context = "\n\n".join(docs)
            
            # Check cache
            cache_key = self.get_cache_key(question, context)
            if use_cache:
                cached = self.response_cache.get(cache_key)
                if cached:
                    cached['cached'] = True
                    cached['timestamp'] = time.time()
                    yield cached
                    continue
            
            # Create prompt
            prompt = f"""[INST] You are a helpful AI assistant that answers questions about code. 
Use the following code context to answer the question. Keep your answer focused and technical.

Context:
{context}

Question: {question}

Answer: [/INST]"""
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.generator.device)
            
            # Stream generation
            streamed_output = ""
            token_buffer = []
            
            # Generate tokens
            for output in self.generator.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                streaming=True
            ):
                # Decode new token
                new_text = self.tokenizer.decode(output[-1], skip_special_tokens=True)
                token_buffer.append(new_text)
                streamed_output += new_text
                
                # Yield when buffer reaches chunk size
                if len(token_buffer) >= self.streaming_config.chunk_size:
                    chunk_text = ''.join(token_buffer)
                    token_buffer = []
                    
                    # Calculate adaptive delay
                    if self.streaming_config.adaptive_speed:
                        delay = self.get_adaptive_delay(chunk_text)
                    else:
                        delay = self.streaming_config.min_delay
                        
                    time.sleep(delay)
                    
                    response = {
                        'question': question,
                        'answer': streamed_output.split("[/INST]")[-1].strip(),
                        'cached': False,
                        'timestamp': time.time(),
                        'complete': False,
                        'chunk_size': len(chunk_text)
                    }
                    yield response
            
            # Final response
            final_answer = streamed_output.split("[/INST]")[-1].strip()
            final_response = {
                'question': question,
                'answer': final_answer,
                'cached': False,
                'timestamp': time.time(),
                'complete': True
            }
            
            # Cache the final response
            if use_cache:
                self.response_cache.put(cache_key, final_response)
                self.save_cache()
            
            yield final_response
    
    def generate(self, questions: List[str], retrieved_docs: List[List[str]], use_cache: bool = True) -> List[str]:
        """Non-streaming version that returns complete answers"""
        answers = []
        for response in self.generate_stream(questions, retrieved_docs, use_cache):
            if response['complete']:
                answers.append(response['answer'])
        return answers
    
    def forward(self, questions, context_embeddings, context_texts):
        """Full RAG pipeline: retrieve -> generate"""
        # Retrieve relevant documents
        retrieved_docs, retrieval_scores = self.retrieve(questions, 
                                                       context_embeddings, 
                                                       context_texts)
        
        # Generate answers
        answers = self.generate(questions, retrieved_docs)
        
        return {
            'answers': answers,
            'retrieved_docs': retrieved_docs,
            'retrieval_scores': retrieval_scores
        } 