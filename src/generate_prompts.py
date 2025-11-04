import json
import itertools
import os

def generate_prompts(probes_filepath):
    """
    Reads a JSON file of probe templates and generates all possible prompt combinations.

    This function opens a JSON file containing axes of demographic terms (like gender, profession)
    and templates with placeholders. It then iterates through each template, combining it with
    all possible permutations of the demographic terms it uses. Each combination forms a
    fully-realized test case.

    Args:
        probes_filepath (str): The path to the probes.json file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              fully-formed test case with a unique prompt and associated metadata.
              Returns None if the file cannot be found or is not valid JSON.
    """
    # Attempt to open and parse the JSON file, with error handling for common issues.
    try:
        with open(probes_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        # Handle cases where the file does not exist.
        print(f"Error: The file was not found at {probes_filepath}")
        return None
    except json.JSONDecodeError:
        # Handle cases where the file is not correctly formatted as JSON.
        print(f"Error: The file at {probes_filepath} is not a valid JSON file.")
        return None

    # Extract the top-level keys from the loaded data.
    all_axes = data['axes']
    all_templates = data['templates']
    generated_test_cases = []

    print("Starting prompt generation...")

    # Iterate through each template defined in the JSON file.
    for template in all_templates:
        template_id = template['id']
        template_string = template['string']
        axes_to_combine = template['axes_used']
        completions = template['completions']

        # Retrieve the lists of demographic terms for the axes specified in the current template.
        # e.g., if axes_used is ["GENDER", "PROFESSION"], this gets the lists of genders and professions.
        term_lists = [all_axes[axis.lower()] for axis in axes_to_combine]

        # Use itertools.product to create all unique combinations of the demographic terms.
        # For example, ('Male', 'Doctor'), ('Male', 'Nurse'), ('Female', 'Doctor'), etc.
        demographic_combinations = list(itertools.product(*term_lists))

        # For each unique combination of demographics, create a specific test case.
        for combo in demographic_combinations:
            prompt = template_string
            demographics = {}
            # Replace placeholders like [GENDER] and [PROFESSION] with the actual terms from the combo.
            for i, axis_name in enumerate(axes_to_combine):
                placeholder = f"[{axis_name}]"
                term = combo[i]
                prompt = prompt.replace(placeholder, term)
                # Store the specific demographics used in this instance.
                demographics[axis_name.lower()] = term
            
            # Construct the two full sentences for stereotype comparison.
            sentence_stereo = f"{prompt} {completions['stereotypical']}"
            sentence_anti_stereo = f"{prompt} {completions['anti_stereotypical']}"

            # Assemble the final test case dictionary.
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
    Main execution block.
    This part of the script runs only when the file is executed directly. It locates
    the probes.json file, calls the generation function, and prints a few examples
    of the generated test cases.
    """
    # Construct the absolute path to the probes.json file.
    # This assumes the script is in the 'src' directory and the data is in a parallel 'data' directory.
    # os.path.abspath(__file__) gets the full path of the current script.
    # os.path.dirname(...) gets the directory containing it.
    # The outer os.path.dirname(...) goes up one level to the project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probes_file = os.path.join(project_root, 'data', 'probes.json')

    # Call the main function to generate all test cases.
    all_prompts = generate_prompts(probes_file)

    # If prompts were generated successfully, print the first few as a sample.
    if all_prompts:
        print("\n--- Example Generated Prompts ---")
        # Use min(3, len(all_prompts)) to avoid errors if fewer than 3 prompts are generated.
        for i in range(min(3, len(all_prompts))):
            # Print each example in a nicely formatted JSON structure.
            print(json.dumps(all_prompts[i], indent=2))