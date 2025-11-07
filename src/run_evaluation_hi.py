#!/usr/bin/env python3
"""
Run Hindi evaluation and store full completions.
"""
import argparse, csv, json, os, random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generate_prompts_hi import generate_prompts

@torch.no_grad()
def calculate_completion_log_prob(prompt: str, completion: str, model, tokenizer, device: str) -> float:
    prompt_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    comp_ids = tokenizer.encode(completion, add_special_tokens=False, return_tensors='pt').to(device)
    input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
    logits = model(input_ids).logits
    comp_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
    log_probs = torch.nn.functional.log_softmax(comp_logits, dim=-1)
    token_log_probs = log_probs.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().cpu().item())

def run_evaluation(model_id: str, probes_path: str, output_path: str,
                   subset: int | None = None, seed: int | None = 42,
                   dtype: str = "float16"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype,
                                                 device_map="auto" if device == "cuda" else None)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    cases = generate_prompts(probes_path)
    if subset:
        if seed is not None:
            random.seed(seed)
        random.shuffle(cases)
        cases = cases[:subset]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'template_id', 'demographics', 'prompt',
            'stereotypical_completion', 'anti_completion',
            'log_prob_stereotypical', 'log_prob_anti',
            'stereotype_preference_score'
        ])
        for case in cases:
            prompt = case['prompt']
            stereo_full = case['sentence_stereotypical']
            anti_full = case['sentence_anti_stereotypical']
            stereo_comp = stereo_full[len(prompt):].strip()
            anti_comp = anti_full[len(prompt):].strip()
            lp_st = calculate_completion_log_prob(prompt, stereo_comp, model, tokenizer, device)
            lp_at = calculate_completion_log_prob(prompt, anti_comp, model, tokenizer, device)
            writer.writerow([
                case['template_id'], json.dumps(case['demographics'], sort_keys=True, ensure_ascii=False), prompt,
                stereo_comp, anti_comp, lp_st, lp_at, lp_st - lp_at
            ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--probes', default=os.path.join('data', 'probes_hi.json'))
    ap.add_argument('--output', default=os.path.join('results', 'baseline_with_completions_hi.csv'))
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dtype', choices=["float16","bfloat16","float32"], default="float16")
    args = ap.parse_args()
    run_evaluation(args.model, args.probes, args.output, args.subset, args.seed, args.dtype)

if __name__ == "__main__":
    main()
