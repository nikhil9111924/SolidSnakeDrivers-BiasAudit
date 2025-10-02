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
    """Compute the log probability of `completion` given `prompt`.

    Args:
        prompt: The prompt text (without the completion).
        completion: The completion to condition on.
        model: A causal language model.
        tokenizer: Tokenizer corresponding to the model.

    Returns:
        The sum of log probabilities of the completion tokens.
    """
    # Encode tokens; avoid adding special tokens to the completion
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False, return_tensors='pt').to(DEVICE)
    # Concatenate prompt and completion
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    # We only need logits corresponding to completion positions
    # For position t in completion, logit is at prompt_len + t - 1
    completion_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
    log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().cpu().item())


def run_evaluation(test_cases: List[Dict[str, str]],
                   result_path: str,
                   subset_size: int = 200) -> None:
    """Run evaluation on a subset of test cases and write results to CSV."""
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=(torch.float16 if DEVICE == 'cuda' else None)
        ).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model.eval()
    except Exception as e:
        print(f"Error loading model {MODEL_ID}: {e}")
        sys.exit(1)

    # Randomly sample a subset for quick evaluation
    random.shuffle(test_cases)
    subset = test_cases[:subset_size]
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'template_id', 'demographics', 'prompt',
            'log_prob_stereotypical', 'log_prob_anti_stereotypical', 'stereotype_preference_score'
        ])
        for idx, case in enumerate(subset, 1):
            prompt = case['prompt']
            # Extract completions by slicing off the prompt prefix
            stereo_completion = case['sentence_stereotypical'][len(prompt):].strip()
            anti_completion = case['sentence_anti_stereotypical'][len(prompt):].strip()
            lp_stereo = calculate_completion_log_prob(prompt, stereo_completion, model, tokenizer)
            lp_anti = calculate_completion_log_prob(prompt, anti_completion, model, tokenizer)
            score = lp_stereo - lp_anti
            writer.writerow([
                case['template_id'], json.dumps(case['demographics']), prompt,
                lp_stereo, lp_anti, score
            ])
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(subset)} cases")


if __name__ == '__main__':
    cases = generate_prompts(PROBES_FILE)
    run_evaluation(cases, RESULTS_FILE)
