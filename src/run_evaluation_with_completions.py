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
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False, return_tensors='pt').to(device)
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    completion_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
    log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
    token_log_probs = log_probs.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().cpu().item())

def run_evaluation(model_id: str, probes_path: str, output_path: str,
                   subset: int = None, device: str = None) -> None:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=(torch.float16 if device == 'cuda' else None)
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()
    test_cases = generate_prompts(probes_path)
    if subset:
        random.shuffle(test_cases)
        test_cases = test_cases[:subset]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'template_id', 'demographics', 'prompt',
            'stereotypical_completion', 'anti_completion',
            'log_prob_stereotypical', 'log_prob_anti',
            'stereotype_preference_score'
        ])
        for idx, case in enumerate(test_cases, 1):
            prompt = case['prompt']
            stereo_comp_full = case['sentence_stereotypical']
            anti_comp_full = case['sentence_anti_stereotypical']
            stereo_comp = stereo_comp_full[len(prompt):].strip()
            anti_comp = anti_comp_full[len(prompt):].strip()
            lp_stereo = calculate_completion_log_prob(prompt, stereo_comp, model, tokenizer, device)
            lp_anti = calculate_completion_log_prob(prompt, anti_comp, model, tokenizer, device)
            score = lp_stereo - lp_anti
            writer.writerow([
                case['template_id'], json.dumps(case['demographics']), prompt,
                stereo_comp, anti_comp,
                lp_stereo, lp_anti, score
            ])
            if idx % 50 == 0:
                print(f"Processed {idx}/{len(test_cases)} prompts")

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM completions and store full completions")
    parser.add_argument('--model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help='Model ID')
    parser.add_argument('--probes', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'probes.json'),
                        help='Path to probes JSON file')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'results', 'baseline_with_completions.csv'),
                        help='CSV file to write results to')
    parser.add_argument('--subset', type=int, default=None, help='Number of random prompts to evaluate')
    args = parser.parse_args()
    run_evaluation(args.model, args.probes, args.output, subset=args.subset)

if __name__ == '__main__':
    main()
