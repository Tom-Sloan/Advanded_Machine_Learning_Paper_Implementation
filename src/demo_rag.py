import argparse
from data_management.code_document_store import CodeDocumentStore
from models.rag import RAG, LRUCache
import torch
import sys
import time
import os

def print_stream(text: str, delay: float = 0.02):
    """Print text with a typewriter effect"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write('\n')

def main():
    parser = argparse.ArgumentParser(description='RAG Demo for Codebase Q&A')
    parser.add_argument('--root_dir', type=str, default='./src',
                      help='Root directory of codebase')
    parser.add_argument('--no_cache', action='store_true',
                      help='Disable response caching')
    parser.add_argument('--no_stream', action='store_true',
                      help='Disable response streaming')
    parser.add_argument('--clear_cache', action='store_true',
                      help='Clear existing response cache')
    args = parser.parse_args()
    
    # Initialize document store and load codebase
    print("Initializing document store...")
    doc_store = CodeDocumentStore()
    doc_store.load_codebase(args.root_dir)
    
    # Initialize RAG model
    print("Loading RAG model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RAG().to(device)
    
    # Clear cache if requested
    if args.clear_cache:
        if os.path.exists(model.response_cache_file):
            os.remove(model.response_cache_file)
            print("Cache cleared.")
        model.response_cache = LRUCache(max_size=model.cache_config.max_size)
    
    print("\nReady for questions! (type 'exit' to quit)")
    
    while True:
        # Get question from user
        question = input("\nQuestion: ").strip()
        
        if question.lower() == 'exit':
            break
        
        # Search codebase
        results = doc_store.search_code(question)
        
        print("\nRelevant code sections:")
        print("-" * 80)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. File: {result['file']}")
            print(f"Score: {result['score']:.4f}")
            print("-" * 40)
            print(result['content'])
        
        # Generate answer using RAG
        context_embeddings = torch.tensor(doc_store.index.reconstruct_n(0, 
            doc_store.index.ntotal)).to(device)
        
        print("\nGenerating Answer:")
        print("-" * 80)
        
        if args.no_stream:
            # Non-streaming mode
            outputs = model([question], context_embeddings, doc_store.documents)
            print(outputs['answers'][0])
        else:
            # Streaming mode
            current_answer = ""
            for response in model.generate_stream(
                [question], 
                [[doc['content'] for doc in results]], 
                use_cache=not args.no_cache
            ):
                if response['cached']:
                    print("(Cached response)")
                    print(response['answer'])
                    break
                
                # Print only new content
                new_content = response['answer'][len(current_answer):]
                if new_content:
                    print_stream(new_content, delay=0.02)
                    current_answer = response['answer']
                
                if response['complete']:
                    break

if __name__ == '__main__':
    main() 