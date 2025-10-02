"""
analyze_results.py
==================

Analyze bias audit results.  It loads a CSV containing template IDs,
demographics, log probabilities, completions and stereotype preference scores,
and computes summary statistics grouped by template or by a demographic axis.
It can also generate a histogram of the scores.

Example:
    python analyze_results.py \
        --input ../results/baseline_with_completions.csv \
        --histogram score_histogram.png \
        --summary template_summary.csv \
        --axis gender \
        --axis_summary gender_summary.csv
"""

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

def parse_demographics(demo_str: str) -> Dict[str, str]:
    return json.loads(demo_str)

def load_results(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['demographics_dict'] = df['demographics'].apply(parse_demographics)
    return df

def compute_summary_by_template(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby('template_id')['stereotype_preference_score'].agg(['mean', 'count']).reset_index()

def compute_summary_by_axis(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    df_axis = df.copy()
    df_axis[axis] = df_axis['demographics_dict'].apply(lambda d: d.get(axis, 'UNKNOWN'))
    return df_axis.groupby(axis)['stereotype_preference_score'].agg(['mean', 'count']).reset_index()

def plot_histogram(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    df['stereotype_preference_score'].hist(bins=20)
    plt.title('Distribution of Stereotype Preference Scores')
    plt.xlabel('Score (positive = stereotypical preference)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Histogram saved to {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bias audit results")
    parser.add_argument('--input', type=str, required=True, help='Path to results CSV')
    parser.add_argument('--histogram', type=str, help='File to save histogram plot (PNG)')
    parser.add_argument('--summary', type=str, help='CSV file to save summary statistics by template')
    parser.add_argument('--axis', type=str, help='Demographic axis to aggregate by (e.g., gender)')
    parser.add_argument('--axis_summary', type=str, help='CSV file to save axis-level summary')
    args = parser.parse_args()
    df = load_results(args.input)
    if args.histogram:
        plot_histogram(df, args.histogram)
    if args.summary:
        summary = compute_summary_by_template(df)
        summary.to_csv(args.summary, index=False)
        print(f"Template summary saved to {args.summary}")
    if args.axis and args.axis_summary:
        axis_summary = compute_summary_by_axis(df, args.axis)
        axis_summary.to_csv(args.axis_summary, index=False)
        print(f"Axis summary saved to {args.axis_summary}")

if __name__ == '__main__':
    main()
