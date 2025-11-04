# src/run_evaluation_mini.py
"""Evaluate stereotype preference for a subset of prompts using TinyLlama.

This script loads a pretrained causal language model, generates prompts
from the probes JSON file, computes the log probability of stereotypical
and anti‑stereotypical completions given each prompt, and writes the
results to a CSV file.  It can run on CPU or GPU.

Usage:  python run_evaluation_mini.py
"""
import csv
import json
import os
import random
import sys
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_prompts import generate_prompts

# Model configuration
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES_FILE = os.path.join(PROJECT_ROOT, 'data', 'probes.json')
RESULTS_FILE = os.path.join(PROJECT_ROOT, 'results', 'baseline_results_english.csv')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_completion_log_prob(prompt: str, completion: str,
                                  model: AutoModelForCausalLM,
                                  tokenizer: AutoTokenizer) -> float:
    """
    Computes the conditional log probability of a completion given a prompt.

    This function tokenizes the prompt and the completion, concatenates them,
    and then feeds them to the model. It calculates the log probability of
    each token in the completion sequence based on the preceding tokens
    (including the prompt). The sum of these log probabilities is returned.

    Args:
        prompt: The context text.
        completion: The text to calculate the log probability for.
        model: A Hugging Face causal language model (e.g., GPT-2, Llama).
        tokenizer: The tokenizer associated with the model.

    Returns:
        The total log probability of the completion sequence.
    """
    # Tokenize the prompt and completion. `add_special_tokens=False` for the
    # completion prevents adding tokens like <bos> or <eos> which are not part of the content.
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False, return_tensors='pt').to(DEVICE)
    
    # Concatenate the token IDs to form a single input sequence.
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    
    # Perform a forward pass through the model to get logits. `torch.no_grad()`
    # is used to disable gradient calculations, which saves memory and computation.
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    
    # The logits for the completion tokens are sliced from the model's output.
    # Logits for the token at position `i` in the input are at `logits[:, i-1, :]`.
    # We want the logits for the completion part, which starts after the prompt.
    completion_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
    
    # Apply log_softmax to the logits to get log probabilities.
    log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
    
    # Use `gather` to select the log probability of the actual completion tokens.
    # `comp_ids` contains the ground-truth tokens for the completion.
    token_log_probs = log_probs.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    
    # Sum the log probabilities of all tokens in the completion.
    return float(token_log_probs.sum().cpu().item())


def run_evaluation(test_cases: List[Dict[str, str]],
                   result_path: str,
                   subset_size: int = 200) -> None:
    """
    Runs the stereotype preference evaluation on a subset of test cases.

    This function loads the specified language model and tokenizer, then iterates
    through a random subset of the provided test cases. For each case, it
    calculates the log probabilities of both the stereotypical and
    anti-stereotypical completions and computes a preference score. The results
    are written to a CSV file.

    Args:
        test_cases: A list of dictionaries, each representing a test case.
        result_path: The path to the output CSV file.
        subset_size: The number of random test cases to evaluate.
    """
    # Load the pretrained model and tokenizer from Hugging Face.
    # The model is moved to the appropriate device (GPU or CPU).
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=(torch.float16 if DEVICE == 'cuda' else None)
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model.eval()  # Set the model to evaluation mode.
    except Exception as e:
        print(f"Error loading model {MODEL_ID}: {e}")
        sys.exit(1)

    # Shuffle the test cases and select a random subset for evaluation.
    random.shuffle(test_cases)
    subset = test_cases[:subset_size]
    
    # Ensure the directory for the results file exists.
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    
    # Open the output file and write the results row by row.
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the header row for the CSV file.
        writer.writerow([
            'template_id', 'demographics', 'prompt',
            'log_prob_stereotypical', 'log_prob_anti_stereotypical', 'stereotype_preference_score'
        ])
        
        # Process each test case in the subset.
        for idx, case in enumerate(subset, 1):
            prompt = case['prompt']
            # The completion text is extracted by removing the prompt from the full sentence.
            stereo_completion = case['sentence_stereotypical'][len(prompt):].strip()
            anti_completion = case['sentence_anti_stereotypical'][len(prompt):].strip()
            
            # Calculate the log probabilities for both completions.
            lp_stereo = calculate_completion_log_prob(prompt, stereo_completion, model, tokenizer)
            lp_anti = calculate_completion_log_prob(prompt, anti_completion, model, tokenizer)
            
            # The stereotype preference score is the difference between the log probabilities.
            # A positive score indicates a preference for the stereotypical completion.
            score = lp_stereo - lp_anti
            
            # Write the results for the current test case to the CSV.
            writer.writerow([
                case['template_id'], json.dumps(case['demographics']), prompt,
                lp_stereo, lp_anti, score
            ])
            
            # Print progress every 50 cases.
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(subset)} cases")


if __name__ == '__main__':
    """
    Main execution block.
    Generates prompts from the probes file and then runs the evaluation.
    """
    # Generate all possible test cases from the probes file.
    cases = generate_prompts(PROBES_FILE)
    # Run the evaluation on the generated cases and save the results.
    run_evaluation(cases, RESULTS_FILE)
