#!/usr/bin/env python3
"""
toxicity_classifier_hi.py
Hindi toxicity scoring (binary abusive detection).

Now accepts column names OR indices (0- or 1-based).
"""
import argparse, csv, os, sys
from typing import List, Sequence
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

def _is_int_like(s: str) -> bool:
    try:
        int(s.strip()); return True
    except Exception:
        return False

def _resolve_col(arg: str, fieldnames: Sequence[str]) -> str:
    arg = arg.strip().replace("\u2013", "-").replace("\u2014", "-")
    if _is_int_like(arg):
        idx = int(arg)
        if 0 <= idx < len(fieldnames):
            return fieldnames[idx]
        if 1 <= idx <= len(fieldnames):
            return fieldnames[idx - 1]
        raise SystemExit(f"[ERR] Column index {idx} out of range. CSV has {len(fieldnames)} columns.")
    if arg in fieldnames:
        return arg
    raise SystemExit(f"[ERR] Column '{arg}' not found. Available: {list(fieldnames)}")

def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    model.eval()
    return tok, model, device

@torch.no_grad()
def score_texts(texts: List[str], tok, model, device: str, batch_size: int = 32, max_length: int = 256) -> List[float]:
    scores = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Toxicity scoring (HI)"):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]  # abusive probability
        scores.extend(probs.detach().cpu().tolist())
    return scores

def process_file(input_path: str, output_path: str, stereo_col_arg: str, anti_col_arg: str,
                 model_id: str = "Hate-speech-CNERG/hindi-abusive-MuRIL",
                 batch_size: int = 32, threshold: float = 0.5):
    tok, model, device = load_model(model_id)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        stereo_col = _resolve_col(stereo_col_arg, reader.fieldnames) if stereo_col_arg else "stereotypical_completion"
        anti_col   = _resolve_col(anti_col_arg, reader.fieldnames) if anti_col_arg else "anti_completion"

        fieldnames = list(reader.fieldnames) + [
            "toxicity_score_stereo_hi", "toxicity_score_anti_hi",
            "toxic_flag_stereo_hi", "toxic_flag_anti_hi"
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        rows, s_texts, a_texts = [], [], []
        for row in reader:
            s_texts.append(row.get(stereo_col, "") or "")
            a_texts.append(row.get(anti_col, "") or "")
            rows.append(row)

        s_scores = score_texts(s_texts, tok, model, device, batch_size=batch_size)
        a_scores = score_texts(a_texts, tok, model, device, batch_size=batch_size)

        for row, s, a in zip(rows, s_scores, a_scores):
            row["toxicity_score_stereo_hi"] = f"{s:.6f}"
            row["toxicity_score_anti_hi"] = f"{a:.6f}"
            row["toxic_flag_stereo_hi"] = int(s >= threshold)
            row["toxic_flag_anti_hi"] = int(a >= threshold)
            writer.writerow(row)

    print(f"[OK] wrote {output_path} (model={model_id})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stereo-col", dest="stereo_col", default=None,
                    help="Column name or index (0-based or 1-based). Default 'stereotypical_completion'")
    ap.add_argument("--anti-col", dest="anti_col", default=None,
                    help="Column name or index (0-based or 1-based). Default 'anti_completion'")
    ap.add_argument("--model", default="Hate-speech-CNERG/hindi-abusive-MuRIL")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    process_file(args.input, args.output, args.stereo_col, args.anti_col,
                 args.model, args.batch_size, args.threshold)

if __name__ == "__main__":
    main()
