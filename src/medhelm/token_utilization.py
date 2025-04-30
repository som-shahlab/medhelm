import pandas as pd
import argparse

def calculate_total_tokens_and_cost(input_file, output_file):
    # Read the input CSV with thousands separator
    df = pd.read_csv(input_file, thousands=',')

    # Group by 'Model' and sum the relevant columns
    grouped = df.groupby('Model', as_index=False)[[
        'Total Input Tokens', 'Total Output Tokens',
        'Total Judge Input Tokens', 'Total Judge Output Tokens',
        'Cost Input Tokens', 'Cost Output Tokens',
        'Cost Jury Input Tokens', 'Cost Jury Output Tokens'
    ]].sum()

    # Add derived columns
    grouped['Benchmark Tokens'] = grouped['Total Input Tokens'] + grouped['Total Output Tokens']
    grouped['Benchmark Cost'] = grouped['Cost Input Tokens'] + grouped['Cost Output Tokens']

    # Add judge/jury total tokens and cost
    grouped['Total Judge Tokens'] = grouped['Total Judge Input Tokens'] + grouped['Total Judge Output Tokens']
    grouped['Total Jury Cost'] = grouped['Cost Jury Input Tokens'] + grouped['Cost Jury Output Tokens']

    # Add total cost based on the sum of benchmark cost and jury cost
    grouped['Total Cost'] = grouped['Benchmark Cost'] + grouped['Total Jury Cost']

    # Reorder or select columns to include in output
    grouped = grouped[[
        'Model',
        'Total Input Tokens', 'Total Output Tokens', 'Benchmark Tokens',
        'Total Judge Input Tokens', 'Total Judge Output Tokens', 'Total Judge Tokens',
        'Cost Input Tokens', 'Cost Output Tokens', 'Benchmark Cost',
        'Cost Jury Input Tokens', 'Cost Jury Output Tokens', 'Total Jury Cost', 'Total Cost'
    ]]

    # Write the result to the output CSV
    grouped.to_csv(output_file, index=False)

    print(f"Total tokens, judge tokens, costs, and total cost per model have been written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate total tokens and cost per model from a CSV file.")
    parser.add_argument('--input_file', '-i', type=str, required=True, help="Path to the input CSV file (cost CSV)")
    parser.add_argument('--output_file', '-o', type=str, default="../data/token_cost_summary.csv", help="Path to the output CSV file")
    args = parser.parse_args()

    calculate_total_tokens_and_cost(args.input_file, args.output_file)
