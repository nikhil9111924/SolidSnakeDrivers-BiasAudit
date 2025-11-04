"""
translate_templates.py
=======================

This module handles translation of the English probe templates into Hindi
using a pretrained machine translation model.  It performs a simple
round‑trip translation (English→Hindi→English) so that annotators can
compare the back‑translation against the original for quality assurance.

The translation logic preserves demographic placeholders (e.g., `[GENDER]`)
by splitting the template string into literal segments and placeholder
segments.  Only literal segments are passed through the translator.

Usage:
    python translate_templates.py \
        --input ../data/probes.json \
        --output ../data/probes_hi.json
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch  # needed for translation generation
from transformers import MarianMTModel, MarianTokenizer

def load_translation_model(src_lang: str = "en", tgt_lang: str = "hi"
                           ) -> Tuple[MarianMTModel, MarianTokenizer]:
    """
    Loads a pretrained MarianMT model and tokenizer for translation.

    This function constructs the model name based on the source and target
    languages and downloads the corresponding model and tokenizer from the
    Hugging Face Hub.

    Args:
        src_lang: The source language code (e.g., "en" for English).
        tgt_lang: The target language code (e.g., "hi" for Hindi).

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    # The model name follows a standard format for Helsinki-NLP's Opus-MT models.
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    # Load the tokenizer and the model from the pretrained checkpoint.
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_sentence(sentence: str, model: MarianMTModel,
                       tokenizer: MarianTokenizer) -> str:
    """
    Translates a single sentence using the provided model and tokenizer.

    Args:
        sentence: The text to translate.
        model: The pretrained translation model.
        tokenizer: The tokenizer for the model.

    Returns:
        The translated sentence as a string.
    """
    # If the sentence is empty, return an empty string to avoid processing.
    if not sentence:
        return ""
    # Tokenize the input sentence and create tensors for the model.
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True)
    # Generate the translation without calculating gradients.
    with torch.no_grad():
        outputs = model.generate(**inputs)
    # Decode the generated token IDs back into a string, skipping special tokens.
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def split_template(template: str) -> List[str]:
    """
    Splits a template string into a list of literal segments and placeholders.

    This allows for translating only the parts of the string that are natural
    language, while keeping placeholders like `[GENDER]` intact.

    Example:
        "The [GENDER] [PROFESSION]" -> ['The ', '[GENDER]', ' ', '[PROFESSION]', '']

    Args:
        template: The string to split.

    Returns:
        A list of strings, alternating between literals and placeholders.
    """
    segments: List[str] = []
    current = ""
    i = 0
    # Iterate through the template string character by character.
    while i < len(template):
        if template[i] == '[':
            # When a '[' is found, it marks the start of a placeholder.
            # First, append any preceding literal text.
            if current:
                segments.append(current)
                current = ""
            # Find the closing ']' and extract the full placeholder.
            end = template.find(']', i) + 1
            segments.append(template[i:end])
            i = end
        else:
            # Otherwise, it's part of a literal segment.
            current += template[i]
            i += 1
    # Append any remaining literal text at the end of the string.
    if current:
        segments.append(current)
    return segments

def translate_template_string(template: str, model: MarianMTModel,
                              tokenizer: MarianTokenizer) -> Tuple[str, str]:
    """
    Translates a template string while preserving its placeholders.

    This function splits the template, translates only the literal segments to
    Hindi, reassembles the template, and then performs a back-translation to
    English for quality checking.

    Args:
        template: The template string with placeholders.
        model: The English-to-Hindi translation model.
        tokenizer: The tokenizer for the En-Hi model.

    Returns:
        A tuple containing:
        - The translated Hindi template string.
        - The back-translated English string.
    """
    # Split the template into literals and placeholders.
    segments = split_template(template)
    translated_segments: List[str] = []
    # Translate only the literal segments.
    for seg in segments:
        if seg.startswith('[') and seg.endswith(']'):
            translated_segments.append(seg)  # Keep placeholders as is.
        else:
            # Translate the stripped literal segment to avoid translating whitespace.
            translated_segments.append(translate_sentence(seg.strip(), model, tokenizer))
            
    # Reassemble the translated segments into a single Hindi template string.
    hindi_template = "".join(seg if seg.startswith('[') else seg + " "
                             for seg in translated_segments).strip()
                             
    # Load a reverse model for back-translation (Hindi to English).
    back_model, back_tok = load_translation_model(src_lang="hi", tgt_lang="en")
    back_segments: List[str] = []
    # Translate the Hindi segments back to English.
    for seg in translated_segments:
        if seg.startswith('[') and seg.endswith(']'):
            back_segments.append(seg)  # Keep placeholders.
        else:
            back_segments.append(translate_sentence(seg.strip(), back_model, back_tok))
            
    # Reassemble the back-translated segments.
    back_translation = "".join(seg if seg.startswith('[') else seg + " "
                               for seg in back_segments).strip()
                               
    return hindi_template, back_translation

def translate_probes(probes_data: Dict[str, any]) -> Dict[str, any]:
    """
    Translates all templates within a probes data structure.

    Args:
        probes_data: A dictionary loaded from a probes JSON file.

    Returns:
        A new dictionary with translated templates including the Hindi version
        and the English back-translation.
    """
    # Load the primary (En-Hi) translation model.
    model, tokenizer = load_translation_model()
    translated_templates = []
    # Iterate through each template in the input data.
    for tmpl in probes_data['templates']:
        # Translate the template string.
        hindi_str, back_str = translate_template_string(tmpl['string'], model, tokenizer)
        # Create a new template dictionary with the added translations.
        translated_templates.append({
            **tmpl,
            'string_hi': hindi_str,
            'string_back_translation': back_str
        })
    # Return the full data structure with the updated templates list.
    return {'axes': probes_data['axes'], 'templates': translated_templates}

def main() -> None:
    """
    Main function to handle command-line arguments and run the translation process.
    """
    # Set up argument parser for input and output file paths.
    parser = argparse.ArgumentParser(description="Translate English probe templates to Hindi")
    parser.add_argument('--input', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'probes.json'),
                        help='Path to the input probes.json file (English)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'probes_hi.json'),
                        help='Path to write the translated probes JSON file')
    args = parser.parse_args()
    
    # Load the original English probes data from the input file.
    with open(args.input, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
        
    # Perform the translation on the entire dataset.
    translated_data = translate_probes(probes_data)
    
    # Write the translated data to the output file.
    # `ensure_ascii=False` is important for correctly saving non-ASCII characters (like Hindi).
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
    print(f"Translated probes saved to {args.output}")

if __name__ == '__main__':
    main()
