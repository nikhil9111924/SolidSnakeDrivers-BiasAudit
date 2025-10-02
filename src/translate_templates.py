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
    """Load a MarianMT model for English→Hindi translation."""
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_sentence(sentence: str, model: MarianMTModel,
                       tokenizer: MarianTokenizer) -> str:
    """Translate a single sentence."""
    if not sentence:
        return ""
    inputs = tokenizer([sentence], return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def split_template(template: str) -> List[str]:
    """
    Split a template into literal segments and placeholders.
    Example: "The [GENDER] [PROFESSION]" → ['The ', '[GENDER]', ' ', '[PROFESSION]', '']
    """
    segments: List[str] = []
    current = ""
    i = 0
    while i < len(template):
        if template[i] == '[':
            if current:
                segments.append(current)
                current = ""
            end = template.find(']', i) + 1
            segments.append(template[i:end])
            i = end
        else:
            current += template[i]
            i += 1
    if current:
        segments.append(current)
    return segments

def translate_template_string(template: str, model: MarianMTModel,
                              tokenizer: MarianTokenizer) -> Tuple[str, str]:
    """
    Translate a template while preserving placeholders.
    Returns (hindi_template, back_translation).
    """
    segments = split_template(template)
    translated_segments: List[str] = []
    for seg in segments:
        if seg.startswith('[') and seg.endswith(']'):
            translated_segments.append(seg)
        else:
            translated_segments.append(translate_sentence(seg.strip(), model, tokenizer))
    # Assemble Hindi template
    hindi_template = "".join(seg if seg.startswith('[') else seg + " "
                             for seg in translated_segments).strip()
    # Back‑translate to English using a reverse model
    back_model, back_tok = load_translation_model(src_lang="hi", tgt_lang="en")
    back_segments: List[str] = []
    for seg in translated_segments:
        if seg.startswith('[') and seg.endswith(']'):
            back_segments.append(seg)
        else:
            back_segments.append(translate_sentence(seg.strip(), back_model, back_tok))
    back_translation = "".join(seg if seg.startswith('[') else seg + " "
                               for seg in back_segments).strip()
    return hindi_template, back_translation

def translate_probes(probes_data: Dict[str, any]) -> Dict[str, any]:
    """Translate all templates in the probes JSON data."""
    model, tokenizer = load_translation_model()
    translated_templates = []
    for tmpl in probes_data['templates']:
        hindi_str, back_str = translate_template_string(tmpl['string'], model, tokenizer)
        translated_templates.append({
            **tmpl,
            'string_hi': hindi_str,
            'string_back_translation': back_str
        })
    return {'axes': probes_data['axes'], 'templates': translated_templates}

def main() -> None:
    parser = argparse.ArgumentParser(description="Translate English probe templates to Hindi")
    parser.add_argument('--input', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'probes.json'),
                        help='Path to the input probes.json file (English)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'data', 'probes_hi.json'),
                        help='Path to write the translated probes JSON file')
    args = parser.parse_args()
    with open(args.input, 'r', encoding='utf-8') as f:
        probes_data = json.load(f)
    translated_data = translate_probes(probes_data)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)
    print(f"Translated probes saved to {args.output}")

if __name__ == '__main__':
    main()
