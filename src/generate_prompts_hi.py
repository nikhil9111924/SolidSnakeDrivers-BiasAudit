import json
import itertools
import os

def generate_prompts(probes_filepath):
    """
    Reads a JSON file of probe templates and generates all possible prompt combinations
    USING THE HINDI 'string_hi' FIELD.
    """
    try:
        with open(probes_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file was not found at {probes_filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file at {probes_filepath} is not a valid JSON file.")
        return None

    all_axes = data['axes']
    all_templates = data['templates']
    generated_test_cases = []

    print("Starting prompt generation (HINDI)...")

    for template in all_templates:
        template_id = template['id']
        
        # Read from 'string_hi' instead of 'string'
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

            test_case = {
                "template_id": template_id,
                "demographics": demographics,
                "prompt": prompt,
                "sentence_stereotypical": sentence_stereo,
                "sentence_anti_stereotypical": sentence_anti_stereo
            }
            generated_test_cases.append(test_case)
    
    print(f"Successfully generated {len(generated_test_cases)} test cases.")
    return generated_test_cases

if __name__ == '__main__':
    """
    Main execution block for testing.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Point to the HINDI probes file
    probes_file = os.path.join(project_root, 'data', 'probes_hi.json')

    all_prompts = generate_prompts(probes_file)

    if all_prompts:
        print("\n--- Example Generated Hindi Prompts ---")
        for i in range(min(3, len(all_prompts))):
            print(json.dumps(all_prompts[i], indent=2, ensure_ascii=False))