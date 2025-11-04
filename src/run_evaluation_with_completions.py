"""
run_evaluation_with_completions.py
==================================

This script runs the bias evaluation and stores the full completions
needed for toxicity analysis.  It computes the log probability of the
stereotypical and anti‑stereotypical completions for each prompt and
writes template_id, demographics, prompt, completions, log probabilities
and the stereotype preference score to a CSV.
"""

import argparse
import csv
import json
import os
import random
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_prompts import generate_prompts

def calculate_completion_log_prob(prompt: str, completion: str,
                                  model: AutoModelForCausalLM,
                                  tokenizer: AutoTokenizer,
                                  device: str) -> float:
    """
    Calculates the conditional log probability of a completion given a prompt.

    This function tokenizes the prompt and completion, feeds them to the model,
    and computes the log probability of the completion sequence. It's a core
    component for scoring how likely a completion is according to the model.

    Args:
        prompt: The context or prompt text.
        completion: The completion text to be scored.
        model: The pretrained causal language model.
        tokenizer: The tokenizer for the model.
        device: The device to run the computation on ('cuda' or 'cpu').

    Returns:
        The total log probability of the completion.
    """
    # Tokenize the prompt and completion separately.
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False, return_tensors='pt').to(device)
    
    # Combine prompt and completion IDs to form the full input sequence.
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    
    # Get model logits without calculating gradients to save resources.
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    
    # Isolate the logits corresponding to the completion part of the input.
    completion_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
    
    # Convert logits to log probabilities.
    log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
    
    # Select the log probabilities of the actual tokens in the completion.
    token_log_probs = log_probs.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    
    # Sum the log probabilities and return as a standard float.
    return float(token_log_probs.sum().cpu().item())

def run_evaluation(model_id: str, probes_path: str, output_path: str,
                   subset: int = None, device: str = None) -> None:
    """
    Runs a bias evaluation and stores results including the full completions.

    This function loads a model, generates prompts, and calculates the stereotype
    preference score for each. Unlike other evaluation scripts, it specifically
    saves the stereotypical and anti-stereotypical completion strings, which are
    needed for downstream tasks like toxicity analysis.

    Args:
        model_id: The Hugging Face model identifier.
        probes_path: Path to the JSON file with probe templates.
        output_path: Path to save the output CSV file.
        subset: If specified, the number of random prompts to evaluate.
        device: The device to run on ('cuda' or 'cpu').
    """
    # Determine the computation device, defaulting to GPU if available.
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the specified model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=(torch.float16 if device == 'cuda' else None)
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()  # Set model to evaluation mode.
    
    # Generate all test cases from the probes file.
    test_cases = generate_prompts(probes_path)
    
    # If a subset size is given, shuffle and select a random subset.
    if subset:
        random.shuffle(test_cases)
        test_cases = test_cases[:subset]
        
    # Create the output directory if it doesn't exist.
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the output CSV file for writing.
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the CSV header, including columns for the completions.
        writer.writerow([
            'template_id', 'demographics', 'prompt',
            'stereotypical_completion', 'anti_completion',
            'log_prob_stereotypical', 'log_prob_anti',
            'stereotype_preference_score'
        ])
        
        # Process each test case.
        for idx, case in enumerate(test_cases, 1):
            prompt = case['prompt']
            # Extract the completion part from the full sentences.
            stereo_comp_full = case['sentence_stereotypical']
            anti_comp_full = case['sentence_anti_stereotypical']
            stereo_comp = stereo_comp_full[len(prompt):].strip()
            anti_comp = anti_comp_full[len(prompt):].strip()
            
            # Calculate log probabilities for both completions.
            lp_stereo = calculate_completion_log_prob(prompt, stereo_comp, model, tokenizer, device)
            lp_anti = calculate_completion_log_prob(prompt, anti_comp, model, tokenizer, device)
            
            # The preference score is the difference in log probabilities.
            score = lp_stereo - lp_anti
            
            # Write the complete record to the CSV file.
            writer.writerow([
                case['template_id'], json.dumps(case['demographics']), prompt,
                stereo_comp, anti_comp,
                lp_stereo, lp_anti, score
            ])
            
            # Print progress periodically.
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(test_cases)} prompts")

def main() -> None:
    """
    Main function to parse command-line arguments and start the evaluation.
    """
    # Set up the argument parser.
    parser = argparse.ArgumentParser(description="Evaluate LLM completions and store full completions")
    parser.add_argument('--model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help='Model ID')
    parser.add_argument('--probes', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'probes.json'),
                        help='Path to probes JSON file')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'results', 'baseline_with_completions.csv'),
                        help='CSV file to write results to')
    parser.add_argument('--subset', type=int, default=None, help='Number of random prompts to evaluate')
    
    # Parse arguments and run the evaluation.
    args = parser.parse_args()
    run_evaluation(args.model, args.probes, args.output, subset=args.subset)

if __name__ == '__main__':
    main()
