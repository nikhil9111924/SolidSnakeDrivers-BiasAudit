#!/usr/bin/env python3
import json, itertools, os

def generate_prompts(probes_filepath):
    """
    Generate Hindi test cases from probes_hi.json using 'string_hi'.
    """
    with open(probes_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_axes = data['axes']
    all_templates = data['templates']
    generated_test_cases = []

    for template in all_templates:
        template_id = template['id']
        template_string = template['string_hi']
        axes_to_combine = template['axes_used']
        completions = template['completions']

        term_lists = [all_axes[axis.lower()] for axis in axes_to_combine]
        demographic_combinations = list(itertools.product(*term_lists))

        for combo in demographic_combinations:
            prompt = template_string
            demographics = {}
            for i, axis_name in enumerate(axes_to_combine):
                placeholder = f"[{axis_name}]"
                term = combo[i]
                prompt = prompt.replace(placeholder, term)
                demographics[axis_name.lower()] = term

            sentence_stereo = f"{prompt} {completions['stereotypical']}"
            sentence_anti_stereo = f"{prompt} {completions['anti_stereotypical']}"

            generated_test_cases.append({
                "template_id": template_id,
                "demographics": demographics,
                "prompt": prompt,
                "sentence_stereotypical": sentence_stereo,
                "sentence_anti_stereotypical": sentence_anti_stereo
            })
    return generated_test_cases

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probes_file = os.path.join(project_root, 'data', 'probes_hi.json')
    all_prompts = generate_prompts(probes_file)
    if all_prompts:
        print(json.dumps(all_prompts[0], ensure_ascii=False, indent=2))
