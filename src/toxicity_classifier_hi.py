"""
toxicity_classifier_hi.py
=========================

This script loads a classifier fine-tuned for HINDI abusive text
and computes toxicity scores for the stereotypical and anti-stereotypical
completions.
"""

import argparse
import csv
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

# This model is fine-tuned for Devanagari Hindi abusive speech.
# Labels: LABEL_0 (Normal), LABEL_1 (Abusive)

MODEL_ID = "Hate-speech-CNERG/hindi-abusive-MuRIL"

def load_model():
    """Loads the Hindi toxicity model and tokenizer."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model {MODEL_ID} on {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).to(device)
    model.eval()
    
    return model, tokenizer, device

def process_file(input_path, output_path):
    """
    Reads the input CSV, computes Hindi toxicity scores, and writes to output.
    """
    model, tokenizer, device = load_model()
    
    # Use os.path.normpath to clean up paths
    input_path = os.path.normpath(input_path)
    output_path = os.path.normpath(output_path)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Processing {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ['toxicity_score_stereo', 'toxicity_score_anti']
        
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in tqdm(reader, desc="Classifying toxicity"):
            stereo_text = row['stereotypical_completion']
            anti_text = row['anti_completion']
            
            # 1. Score stereotypical text
            inputs_stereo = tokenizer(stereo_text, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                logits_stereo = model(**inputs_stereo).logits
            
            # Apply softmax to get probabilities
            probs_stereo = torch.softmax(logits_stereo, dim=1)
            # The "abusive" score is the probability of LABEL_1
            row['toxicity_score_stereo'] = probs_stereo[0, 1].cpu().item()

            # 2. Score anti-stereotypical text
            inputs_anti = tokenizer(anti_text, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                logits_anti = model(**inputs_anti).logits

            probs_anti = torch.softmax(logits_anti, dim=1)
            row['toxicity_score_anti'] = probs_anti[0, 1].cpu().item()
            
            writer.writerow(row)
            
    print(f"Toxicity analysis complete. Output saved to {output_path}")

def main():
    """Parses arguments and starts the classification."""
    parser = argparse.ArgumentParser(description="Run Hindi toxicity classification")

    parser.add_argument('--input', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'results', 'baseline_with_completions_hi.csv'),
                        help='Input CSV file with completions')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'results', 'baseline_with_toxicity_hi.csv'),
                        help='Output CSV file with toxicity scores')
    
    args = parser.parse_args()
    process_file(args.input, args.output)

if __name__ == '__main__':
    main()