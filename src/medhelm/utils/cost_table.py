import os
import json
import pandas as pd
import re
import argparse

from medhelm.utils.constants import BENCHMARK_NAME_MAPPING, BENCHMARK_QUESTION_TYPE


def parse_directory_name(directory):
    benchmark, sep, rest = directory.partition(':')
    subset = None
    if "med_dialog" in benchmark:
        subset = benchmark.split(",")[-1]
        benchmark = "med_dialog"
    subset = None
    if sep:
        subset_part = rest.split('model=')[0].rstrip(',')
        subset = subset_part if subset_part else subset

    model_match = re.search(r'model=([^,]+)', directory)
    model_raw = model_match.group(1) if model_match else 'unknown_model'
    model_name = re.sub(r'^[^_]+_', '', model_raw).replace('-', '_')

    return benchmark, subset, model_name

def extract_tokens(stats_file):
    try:
        with open(stats_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        print(f"Error reading {stats_file}: {e}")
        return 0, 0, 0

    input_tokens = 0
    output_tokens = 0
    test_instances = 0

    for item in data:
        if isinstance(item, dict) and 'name' in item and 'sum' in item:
            name = item['name']['name']
            if name == 'num_prompt_tokens':
                input_tokens = item['sum']
            elif name == 'max_num_completion_tokens':
                output_tokens = item['sum']
            elif name == 'num_completions':
                test_instances = item['sum']

    return input_tokens, output_tokens, test_instances

def generate_table(base_dir):
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

        benchmark, _, model = parse_directory_name(os.path.basename(directory))
        input_tokens, output_tokens, test_instances = extract_tokens(stats_file)

        row = {
            'Model': model,
            'Benchmark': benchmark,
            'Question Type': BENCHMARK_QUESTION_TYPE.get(benchmark, 'Unknown'),
            'Total Input Tokens': input_tokens,
            'Total Output Tokens': output_tokens,
            'Test Instances': test_instances,
        }
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Generate and aggregate benchmark stats.")
    parser.add_argument("base_dir", type=str, help="Path to directory containing benchmark runs.")
    parser.add_argument("--output_file", type=str, default="aggregated_results.csv", help="Path to save aggregated CSV.")
    args = parser.parse_args()

    # Step 1: Generate raw stats
    table = generate_table(args.base_dir)

    if table.empty:
        print("No valid data found.")
        return

    # Step 2: Apply benchmark name mapping
    table['Benchmark'] = table['Benchmark'].map(BENCHMARK_NAME_MAPPING).fillna(table['Benchmark'])

    # Step 3: Aggregate
    aggregated = table.groupby(['Model', 'Benchmark', 'Question Type'], dropna=False).agg({
        'Test Instances': 'sum',
        'Total Input Tokens': 'sum',
        'Total Output Tokens': 'sum',
    }).reset_index()

    # Step 4: Save aggregated result
    aggregated.to_csv(args.output_file, index=False)
    print(f"Aggregated results saved to {args.output_file}")

if __name__ == "__main__":
    main()
