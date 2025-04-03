import os
import json
import pandas as pd
import re
import argparse

"""
This script generates a CSV table containing the following columns:
    'Model',
    'Benchmark',
    'Total Input Tokens',
    'Total Output Tokens',
    'Cost per Input Token', # Placeholder for cost calculation
    'Cost per Output Token', # Placeholder for cost calculation
    'Cost Input Tokens', # Placeholder for cost calculation
    'Cost Output Tokens', # Placeholder for cost calculation
    'Total Cost' # Placeholder for cost calculation
"""


def parse_directory_name(directory):
    """
    Extracts the Benchmark and Model from the directory name using regex.
    - Benchmark: everything before 'model='
    - Model: extracted using 'model=' followed by the model name
    """
    benchmark = re.split(r',?model=', directory)[0]
    model_match = re.search(r'model=([^,]+)', directory)
    model = model_match.group(1).replace('-', '_') if model_match else 'unknown_model'
    return benchmark, model

def extract_tokens(stats_file):
    """Extracts Total Input and Output Tokens from stats.json."""
    try:
        with open(stats_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        print(f"Error reading {stats_file}: {e}")
        return 0, 0

    input_tokens = 0
    output_tokens = 0

    for item in data:
        if isinstance(item, dict) and 'name' in item and 'sum' in item:
            name = item['name']['name']
            if name == 'num_prompt_tokens':
                input_tokens = item['sum']
            elif name == 'max_num_completion_tokens':
                output_tokens = item['sum']

    return input_tokens, output_tokens

def generate_table(base_dir):
    """Generates a table for all directories in the base directory."""
    rows = []
    
    if not os.path.isdir(base_dir):
        print(f"Error: The base directory '{base_dir}' does not exist.")
        return pd.DataFrame()

    directories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for directory in directories:
        stats_file = os.path.join(directory, 'stats.json')

        if not os.path.isfile(stats_file):
            print(f"Skipping {directory}: stats.json not found.")
            continue

        benchmark, model = parse_directory_name(os.path.basename(directory))
        input_tokens, output_tokens = extract_tokens(stats_file)

        row = {
            'Model': model,
            'Benchmark': benchmark,
            'Total Input Tokens': input_tokens,
            'Total Output Tokens': output_tokens,
            'Cost per Input Token': '',
            'Cost per Output Token': '',
            'Cost Input Tokens': '',
            'Cost Output Tokens': '',
            'Total Cost': ''
        }
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Extract benchmark statistics and save to CSV.")
    parser.add_argument("base_dir", type=str, help="Path to the suite directory containing HELM benchmark runs.")
    parser.add_argument("--output_file", type=str, default="cost_table.csv", help="Output CSV file path.")
    args = parser.parse_args()

    # Generate the table
    table = generate_table(args.base_dir)

    if not table.empty:
        table.to_csv(args.output_file, index=False)
        print(f"Table saved to {args.output_file}")
        print(table)
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()

