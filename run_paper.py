import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import inquirer
import os
import re
from pathlib import Path
import argparse
from src.data_management.dataset import get_dataset
import matplotlib.pyplot as plt

def get_completed_papers():
    """Read descriptions.md and extract completed papers with their sections"""
    with open('descriptions.md', 'r') as f:
        content = f.read()
    
    # Split content into sections
    sections = content.split('## ')[1:]  # Skip the first empty split
    
    completed_papers = []
    full_citations = []
    section_names = []
    
    for section in sections:
        # Get section name
        section_name = section.split('\n')[0].strip()
        # Find papers marked as completed (bolded)
        papers = re.findall(r'\*\*(.*?)\*\*', section)
        
        for paper in papers:
            # Skip papers with forward slashes
            if '/' not in paper:
                # Extract just the paper name (text between quotes)
                paper_name = paper.split('"')[1] if '"' in paper else paper
                # Prepend section name to paper name for organization
                completed_papers.append(f"{section_name}: {paper_name}")
                full_citations.append(paper)
                section_names.append(section_name)
    
    return completed_papers, full_citations, section_names

def get_paper_info(full_citation):
    """Returns formatted paper information including bullet points from descriptions.md"""
    # Read descriptions.md to get bullet points
    with open('descriptions.md', 'r') as f:
        content = f.read()
    
    # Split citation into parts
    parts = full_citation.split('"')
    authors = parts[0]
    title = parts[1] if len(parts) > 1 else ""
    conference = parts[2] if len(parts) > 2 else ""
    
    # Clean the title for searching
    clean_title = title.rstrip(',').strip()
    
    # Split content into sections
    sections = content.split('## ')
    bullet_points = []    
    for section in sections:
        # Look for the paper in this section
        papers = section.split('\n\n')
        for index, paper in enumerate(papers):
            # Check if this paper contains our title
            if clean_title in paper:
                if index + 1 < len(papers):
                    # Extract bullet points
                    lines = papers[index + 1].split('\n')
                    bullet_points = [
                        line.strip().replace('-', '•').strip() 
                        for line in lines 
                        if line.strip() and line.strip().startswith('-')
                    ]
                break
        if bullet_points:
            break
    
    # Format the information
    info = f"""
Paper Information:
{pad_string('Authors:', 20)}{authors.strip()}
{pad_string('Title:', 20)}{clean_title}
{pad_string('Conference:', 20)}{conference.strip()}

Key Points:"""
    
    # Add bullet points
    if bullet_points:
        info += '\n' + '\n'.join(bullet_points)
    else:
        info += '\nNo additional information available.'
    
    return info

def pad_string(string, length):
    """Pads a string with spaces to a given length"""
    return string.ljust(length)

def run_vit_implementation(mode='train'):
    """Run the Vision Transformer implementation"""
    print("Running Vision Transformer (ViT) implementation...")
    
    # Command to run ViT implementation
    args = [
        '--mode', mode,
        '--data_dir', './data/Stoneflies',
        '--epochs', '30',
        '--batch_size', '32',
        '--learning_rate', '0.001'
    ]
    
    if mode == 'train':
        args.append('--augment')
    
    cmd = f"python src/train_vit.py {' '.join(args)}"
    os.system(cmd)

def run_gradcam_implementation(mode='eval'):
    """Run the Grad-CAM implementation"""
    print("Running Grad-CAM implementation...")
    
    # Command to run Grad-CAM implementation
    args = [
        '--mode', mode,
        '--data_dir', './data/Stoneflies',
        '--epochs', '5',
        '--batch_size', '32',
        '--learning_rate', '0.0001',
        '--num_samples', '20'
    ]
    
    cmd = f"python src/Grad-cam.py {' '.join(args)}"
    os.system(cmd)

def sample_dataset(dataset_name):
    """Display samples from the selected dataset"""
    try:
        # Get the appropriate dataset
        data_dir = f'./data/{dataset_name}'
        dataset = get_dataset(dataset_name.lower(), data_dir)
        
        # Get samples
        if dataset_name.lower() == 'stoneflies':
            # For Stonefly dataset, get figure and samples
            fig, samples = dataset.get_sample(num_samples=5)
            plt.show()
        else:
            # For text datasets, print samples
            samples, seq_length = dataset.get_sample(num_samples=5)
            print(f"\nSamples from {dataset_name} dataset:")
            print("-" * 50)
            for i, sample in enumerate(samples, 1):
                print(f"\nSample {i}:")
                print(sample)
                print("-" * 50)
            print(f"\nSequence length: {seq_length}")
    except Exception as e:
        print(f"Error sampling dataset: {str(e)}")

def main():
    while True:
        # Get list of papers with their sections
        paper_names, full_citations, section_names = get_completed_papers()
        
        if not paper_names:
            print("No paper implementations found!")
            return
        
        # Add dataset sampling option to main menu
        main_choices = ['View Papers', 'Sample Dataset', 'Exit']
        
        questions = [
            inquirer.List('action',
                         message="What would you like to do?",
                         choices=main_choices)
        ]
        
        main_answer = inquirer.prompt(questions)
        
        if main_answer['action'] == 'Exit':
            break
            
        elif main_answer['action'] == 'Sample Dataset':
            dataset_choices = ['Stoneflies', 'Shakespeare_char', 'Shakespeare_word', 'Back']
            dataset_question = [
                inquirer.List('dataset',
                             message="Select a dataset to sample:",
                             choices=dataset_choices)
            ]
            
            dataset_answer = inquirer.prompt(dataset_question)
            if dataset_answer['dataset'] != 'Back':
                sample_dataset(dataset_answer['dataset'])
            continue
            
        elif main_answer['action'] == 'View Papers':
            # Add back/exit option to paper choices
            paper_choices = paper_names + ['Back']
            
            # Ask user which paper to view
            questions = [
                inquirer.List('paper',
                             message="Select a paper:",
                             choices=paper_choices)
            ]
            
            answers = inquirer.prompt(questions)
            selected_paper = answers['paper']
            
            if selected_paper == 'Back':
                continue
                
            # Get full citation for selected paper
            paper_idx = paper_names.index(selected_paper)
            full_citation = full_citations[paper_idx]
            section_name = section_names[paper_idx]
            
            while True:
                # Ask what they want to do with selected paper
                print(f"\n{selected_paper}\n")
                action_question = [
                    inquirer.List('action',
                                 message="What would you like to do?",
                                 choices=['View Paper Information', 'Train and Evaluate', 'Evaluate', 'Back'])
                ]
                action = inquirer.prompt(action_question)
                
                if action['action'] == 'Back':
                    break
                
                if action['action'] == 'View Paper Information':
                    print(get_paper_info(full_citation))
                    continue
                
                # Run appropriate implementation based on paper selection
                if "An Image is Worth 16x16 Words" in selected_paper:
                    if action['action'] == 'Train and Evaluate':
                        run_vit_implementation('train')
                    elif action['action'] == 'Evaluate':
                        run_vit_implementation('eval')
                elif "Grad-CAM" in selected_paper:
                    if action['action'] == 'Train and Evaluate':
                        run_gradcam_implementation('train')
                    elif action['action'] == 'Evaluate':
                        run_gradcam_implementation('eval')
                else:
                    print(f"Demo for paper '{selected_paper}' is not yet available.")

if __name__ == "__main__":
    main()

"""
Manual Execution Commands:

1. Vision Transformer (ViT):
   Training:
   python src/train_vit.py --mode train --data_dir ./Data/Stoneflies --model_path ./trained_models/vit_stonefly/best_model.pt --epochs 10 --batch_size 32 --learning_rate 0.001 --augment
   
   Evaluation:
   python src/train_vit.py --mode eval --data_dir ./Data/Stoneflies --model_path ./trained_models/vit_stonefly/best_model.pt --batch_size 32

2. Grad-CAM:
   python src/Grad-cam.py
"""
