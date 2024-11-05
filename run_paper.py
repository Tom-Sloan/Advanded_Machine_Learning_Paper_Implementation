import inquirer
import os
import re
from pathlib import Path
import argparse

def get_completed_papers():
    # Read README.md
    with open('README.md', 'r') as f:
        readme_content = f.read()
    
    # Find papers marked as completed (bolded)
    completed_papers = re.findall(r'\*\*(.*?)\*\*', readme_content)
    
    # Clean up the paper titles to just show paper name
    cleaned_papers = []
    for paper in completed_papers:
        # Extract just the paper name before the citation
        paper_name = paper.split('"')[1] if '"' in paper else paper
        cleaned_papers.append(paper_name)
        
    return cleaned_papers, completed_papers

def get_paper_info(full_citation):
    """Returns formatted paper information"""
    # Split citation into parts
    parts = full_citation.split(',')
    authors = parts[0]
    title = parts[1] if len(parts) > 1 else ""
    venue = parts[2] if len(parts) > 2 else ""
    year = parts[3] if len(parts) > 3 else ""
    
    info = f"""
Paper Information:
Authors: {authors}
Title: {title}
Venue: {venue}
Year: {year}
    """
    return info

def run_vit_implementation(mode='train'):
    """Run the Vision Transformer implementation"""
    print("Running Vision Transformer (ViT) implementation...")
    
    # Command to run ViT implementation
    # python Implementation/train_vit.py --mode train --data_dir ./Data/Stoneflies --model_path @/trained_models/vit_stonefly/best_model.pt --epochs 10 --batch_size 32 --learning_rate 0.001 --augment
    args = [
        '--mode', mode,
        '--data_dir', './data/Stoneflies',
        '--model_path', '@/trained_models/vit_stonefly/best_model.pt',
        '--epochs', '10',
        '--batch_size', '32',
        '--learning_rate', '0.001'
    ]
    
    if mode == 'train':
        args.append('--augment')
    
    cmd = f"python Implementation/train_vit.py {' '.join(args)}"
    os.system(cmd)

def run_gradcam_implementation():
    """Run the Grad-CAM implementation"""
    print("Running Grad-CAM implementation...")
    
    # Command to run Grad-CAM implementation
    # python Implementation/Grad-cam.py
    cmd = "python Implementation/Grad-cam.py"
    os.system(cmd)

def main():
    while True:
        # Get list of papers
        paper_names, full_citations = get_completed_papers()
        
        if not paper_names:
            print("No paper implementations found!")
            return
            
        # Add back/exit option to main menu
        paper_choices = paper_names + ['Exit']
        
        # Ask user which paper to view
        questions = [
            inquirer.List('paper',
                         message="Select a paper:",
                         choices=paper_choices)
        ]
        
        answers = inquirer.prompt(questions)
        selected_paper = answers['paper']
        
        if selected_paper == 'Exit':
            break
            
        # Get full citation for selected paper
        paper_idx = paper_names.index(selected_paper)
        full_citation = full_citations[paper_idx]
        
        while True:
            # Ask what they want to do with selected paper
            action_question = [
                inquirer.List('action',
                             message="What would you like to do?",
                             choices=['View Paper Information', 'Run Demo', 'Back'])
            ]
            action = inquirer.prompt(action_question)
            
            if action['action'] == 'Back':
                break
                
            if action['action'] == 'View Paper Information':
                print(get_paper_info(full_citation))
                continue
                
            # Run appropriate implementation based on paper selection
            if "An Image is Worth 16x16 Words" in selected_paper:
                mode_question = [
                    inquirer.List('mode',
                                message="Would you like to train or evaluate the model?",
                                choices=['train', 'eval'])
                ]
                mode = inquirer.prompt(mode_question)
                run_vit_implementation(mode['mode'])
            elif "Grad-cam" in selected_paper:
                run_gradcam_implementation()
            else:
                print(f"Demo for paper '{selected_paper}' is not yet available.")

if __name__ == "__main__":
    main()

"""
Manual Execution Commands:

1. Vision Transformer (ViT):
   Training:
   python Implementation/train_vit.py --mode train --data_dir ./Data/Stoneflies --model_path @/trained_models/vit_stonefly/best_model.pt --epochs 10 --batch_size 32 --learning_rate 0.001 --augment
   
   Evaluation:
   python Implementation/train_vit.py --mode eval --data_dir ./Data/Stoneflies --model_path @/trained_models/vit_stonefly/best_model.pt --batch_size 32

2. Grad-CAM:
   python Implementation/Grad-cam.py
"""
