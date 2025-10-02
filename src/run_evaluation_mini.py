import json
import csv
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_prompts import generate_prompts
import random

# --- Configuration ---
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # A powerful, Llama-compatible open model
PROBES_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'probes.json')
RESULTS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'baseline_results_english.csv')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available

def calculate_completion_log_prob(prompt, completion, model, tokenizer):
    """
    Calculates the conditional log probability of a completion given a prompt.
    P(completion | prompt)
    """
    # Tokenize the prompt and completion
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False, return_tensors='pt').to(DEVICE)
    
    # Combine them for the model input
    full_tokens = torch.cat([prompt_tokens, completion_tokens], dim=1)
    
    # Get the model's logits (raw scores for each token)
    with torch.no_grad():
        outputs = model(full_tokens)
        logits = outputs.logits

    # We only care about the logits for the completion tokens
    # The logits for the first completion token are at the position of the last prompt token
    completion_logits = logits[:, prompt_tokens.shape[1]-1:-1, :]
    
    # Use log_softmax to get log probabilities
    log_probs = torch.nn.functional.log_softmax(completion_logits, dim=-1)
    
    # Gather the log probabilities of the actual completion tokens
    # completion_tokens needs to be reshaped to be used with gather
    completion_log_probs = log_probs.gather(2, completion_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Sum the log probabilities to get the total sentence log probability
    return completion_log_probs.sum().item()


def run_evaluation():
    """
    Main function to run the entire evaluation pipeline.
    """
    print("--- Starting Evaluation ---")
    
    # 1. Load Model and Tokenizer
    print(f"Loading model: {MODEL_ID} onto {DEVICE}...")
    # Note: For float16, a GPU is recommended. If on CPU, you might need to remove torch_dtype.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model.eval() # Set model to evaluation mode

    # 2. Generate all prompts from the JSON file
    print(f"Generating prompts from {PROBES_FILE}...")
    all_test_cases = generate_prompts(PROBES_FILE)

    random.shuffle(all_test_cases) # Randomize the order of test cases
    all_test_cases = all_test_cases[:200] # Take only the first 200 cases for a quick run
    
    if not all_test_cases:
        print("No test cases generated. Exiting.")
        return

    # 3. Prepare CSV file for results
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        header = ["template_id", "demographics", "prompt", 
                  "log_prob_stereotypical", "log_prob_anti_stereotypical", "stereotype_preference_score"]
        writer.writerow(header)

        print(f"Starting evaluation of {len(all_test_cases)} test cases...")
        # 4. Iterate through each test case and get scores
        for i, case in enumerate(all_test_cases):
            prompt = case['prompt']
            
            # Get the text of the completions from the full sentences
            stereo_completion = case['sentence_stereotypical'].replace(prompt, '').strip()
            anti_stereo_completion = case['sentence_anti_stereotypical'].replace(prompt, '').strip()

            # Calculate log probabilities for both completions
            log_prob_stereo = calculate_completion_log_prob(prompt, stereo_completion, model, tokenizer)
            log_prob_anti_stereo = calculate_completion_log_prob(prompt, anti_stereo_completion, model, tokenizer)

            # Calculate the Stereotype Preference Score (log-odds difference)
            # A positive score means the stereotypical completion is more likely.
            stereo_score = log_prob_stereo - log_prob_anti_stereo

            # Prepare row for CSV
            row = [
                case['template_id'],
                json.dumps(case['demographics']),
                prompt,
                log_prob_stereo,
                log_prob_anti_stereo,
                stereo_score
            ]
            writer.writerow(row)

            if (i + 1) % 50 == 0:
                print(f"  ...processed {i+1}/{len(all_test_cases)} cases")

    print(f"--- Evaluation Complete ---")
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == '__main__':
    run_evaluation()