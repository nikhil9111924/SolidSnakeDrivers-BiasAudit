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
    """
    Parses a JSON string representing demographics into a dictionary.

    Args:
        demo_str: A string in JSON format (e.g., '{"gender": "Male", "profession": "Doctor"}').

    Returns:
        A dictionary containing the parsed demographic key-value pairs.
    """
    # The input string is loaded as a JSON object, converting it into a Python dictionary.
    return json.loads(demo_str)

def load_results(csv_path: str) -> pd.DataFrame:
    """
    Loads bias audit results from a CSV file into a pandas DataFrame.

    Args:
        csv_path: The file path to the results CSV.

    Returns:
        A pandas DataFrame with an added column 'demographics_dict' containing parsed demographics.
    """
    # Read the CSV file into a DataFrame.
    df = pd.read_csv(csv_path)
    # Apply the 'parse_demographics' function to the 'demographics' column
    # to create a new column with dictionary objects.
    df['demographics_dict'] = df['demographics'].apply(parse_demographics)
    return df

def compute_summary_by_template(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes summary statistics (mean and count) of stereotype preference scores, grouped by template ID.

    Args:
        df: The DataFrame containing the results.

    Returns:
        A DataFrame with summary statistics for each template.
    """
    # Group the DataFrame by 'template_id' and calculate the mean and count
    # of the 'stereotype_preference_score' for each group.
    return df.groupby('template_id')['stereotype_preference_score'].agg(['mean', 'count']).reset_index()

def compute_summary_by_axis(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """
    Computes summary statistics for stereotype preference scores, grouped by a specified demographic axis.

    Args:
        df: The DataFrame containing the results.
        axis: The demographic axis to group by (e.g., 'gender', 'age').

    Returns:
        A DataFrame with summary statistics for each value within the specified axis.
    """
    # Create a copy of the DataFrame to avoid modifying the original.
    df_axis = df.copy()
    # Extract the value for the specified 'axis' from the 'demographics_dict' column.
    # If the axis is not present, it defaults to 'UNKNOWN'.
    df_axis[axis] = df_axis['demographics_dict'].apply(lambda d: d.get(axis, 'UNKNOWN'))
    # Group by the specified axis and compute the mean and count of scores.
    return df_axis.groupby(axis)['stereotype_preference_score'].agg(['mean', 'count']).reset_index()

def plot_histogram(df: pd.DataFrame, output_path: str) -> None:
    """
    Generates and saves a histogram of the stereotype preference scores.

    Args:
        df: The DataFrame containing the scores.
        output_path: The file path to save the generated histogram image.
    """
    # Create a new plot figure with a specified size.
    plt.figure(figsize=(8, 5))
    # Generate a histogram from the 'stereotype_preference_score' column with 20 bins.
    df['stereotype_preference_score'].hist(bins=20)
    # Set the title and labels for the plot.
    plt.title('Distribution of Stereotype Preference Scores')
    plt.xlabel('Score (positive = stereotypical preference)')
    plt.ylabel('Frequency')
    # Display a grid and ensure the layout is tight.
    plt.grid(True)
    plt.tight_layout()
    # Save the plot to the specified output path.
    plt.savefig(output_path)
    print(f"Histogram saved to {output_path}")

def main() -> None:
    """
    Main function to parse arguments and run the analysis.
    It loads data, generates a histogram, and computes summaries as specified by command-line arguments.
    """
    # Set up an argument parser to handle command-line inputs.
    parser = argparse.ArgumentParser(description="Analyze bias audit results")
    parser.add_argument('--input', type=str, required=True, help='Path to results CSV')
    parser.add_argument('--histogram', type=str, help='File to save histogram plot (PNG)')
    parser.add_argument('--summary', type=str, help='CSV file to save summary statistics by template')
    parser.add_argument('--axis', type=str, help='Demographic axis to aggregate by (e.g., gender)')
    parser.add_argument('--axis_summary', type=str, help='CSV file to save axis-level summary')
    
    # Parse the provided arguments.
    args = parser.parse_args()
    
    # Load the results from the input CSV file.
    df = load_results(args.input)
    
    # If a path for the histogram is provided, generate and save it.
    if args.histogram:
        plot_histogram(df, args.histogram)
        
    # If a path for the template summary is provided, compute and save it.
    if args.summary:
        summary = compute_summary_by_template(df)
        summary.to_csv(args.summary, index=False)
        print(f"Template summary saved to {args.summary}")
        
    # If an axis and a path for the axis summary are provided, compute and save it.
    if args.axis and args.axis_summary:
        axis_summary = compute_summary_by_axis(df, args.axis)
        axis_summary.to_csv(args.axis_summary, index=False)
        print(f"Axis summary saved to {args.axis_summary}")

if __name__ == '__main__':
    main()
