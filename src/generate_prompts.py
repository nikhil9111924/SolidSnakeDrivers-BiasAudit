import json
import itertools
import os

def generate_prompts(probes_filepath):
    """
    Reads the probes.json file and generates all possible prompt combinations.

    Args:
        probes_filepath (str): The path to the probes.json file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              fully-formed test case.
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

    print("Starting prompt generation...")

    # Iterate through each template in the JSON file
    for template in all_templates:
        template_id = template['id']
        template_string = template['string']
        axes_to_combine = template['axes_used']
        completions = template['completions']

        # Get the lists of terms for the axes used in this template
        term_lists = [all_axes[axis.lower()] for axis in axes_to_combine]

        # Use itertools.product to get all combinations of the terms
        # e.g., ('Male', 'Doctor'), ('Male', 'Nurse'), ...
        demographic_combinations = list(itertools.product(*term_lists))

        for combo in demographic_combinations:
            prompt = template_string
            demographics = {}
            # Replace placeholders like [GENDER] with the actual terms
            for i, axis_name in enumerate(axes_to_combine):
                placeholder = f"[{axis_name}]"
                term = combo[i]
                prompt = prompt.replace(placeholder, term)
                demographics[axis_name.lower()] = term
            
            # Create the two full sentences for comparison
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
    # Construct the path to the probes.json file assuming the script is in src/
    # and the data is in data/
    # ../ -> go up one directory from src/ to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probes_file = os.path.join(project_root, 'data', 'probes.json')

    # Generate all the test cases
    all_prompts = generate_prompts(probes_file)

    if all_prompts:
        # Print the first 3 generated prompts as an example
        print("\n--- Example Generated Prompts ---")
        for i in range(min(3, len(all_prompts))):
            print(json.dumps(all_prompts[i], indent=2))