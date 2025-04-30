import argparse
import pandas as pd
import matplotlib.pyplot as plt

from medhelm.utils.constants import BENCHMARKS, BENCHMARK_NAME_MAPPING,  MODEL_NAME_MAPPING


def main(
    costs_file: str,
    leaderboard_file: str,
    output_file: str,
    category: str
):
    data = pd.read_csv(costs_file)

    # Filter benchmarks by category if specified
    if category:
        # Step 1: Collect all benchmark IDs under the given category
        category_benchmarks = set()
        if category in BENCHMARKS:
            for subcat in BENCHMARKS[category].values():
                category_benchmarks.update(subcat)
        else:
            raise ValueError(f"Category '{category}' not found in BENCHMARKS")

        # Step 2: Map to pretty names using BENCHMARK_NAME_MAPPING
        pretty_benchmarks = {
            BENCHMARK_NAME_MAPPING[bench]
            for bench in category_benchmarks
            if bench in BENCHMARK_NAME_MAPPING
        }
        # Step 3: Filter the data
        data = data[data['Benchmark'].isin(pretty_benchmarks)]
        assert len(pretty_benchmarks) == len(data["Benchmark"].unique()), f"Benchmarks for {category} don't match."

    data["Model"] = data["Model"].map(MODEL_NAME_MAPPING)
    aggregated_data = data.groupby('Model')['Cost Input Output Tokens'].sum()

    leaderboard_data = pd.read_csv(leaderboard_file)
    mean_win_rate = leaderboard_data.groupby('Model')['Mean win rate'].mean()

    merged_data = pd.DataFrame({
        'Aggregated Cost': aggregated_data,
        'Mean Win Rate': mean_win_rate
    }).dropna()

    print(merged_data)

    colors = plt.cm.tab10(range(len(merged_data)))
    merged_data['Color'] = [colors[i] for i in range(len(merged_data))]

    plt.figure(figsize=(12, 8))
    for i, (model, row) in enumerate(merged_data.iterrows()):
        plt.scatter(
            row['Aggregated Cost'], 
            row['Mean Win Rate'], 
            color=row['Color'], 
            edgecolors='black', 
            s=200,
            label=model  # Label for traditional legend
        )

    title = 'Cost vs Mean Win Rate'
    if category:
        title = f'{title} ({category})'
    plt.title(title, fontsize=16)
    plt.xlabel('Total Cost (USD)', fontsize=14)
    plt.ylabel('Mean Win Rate', fontsize=14)
    plt.grid(True)
    plt.legend(title="Model", fontsize=10, title_fontsize=12, loc='best')  # Traditional legend
    plt.tight_layout()

    plt.xlim(merged_data['Aggregated Cost'].min() * 0.9, merged_data['Aggregated Cost'].max() * 1.1)
    plt.ylim(merged_data['Mean Win Rate'].min() * 0.9, merged_data['Mean Win Rate'].max() * 1.1)

    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates plots for cost vs performance of models.')
    parser.add_argument('--costs_file', '-c', type=str, required=True, help='Path to the costs CSV file')
    parser.add_argument('--leaderboard_file', '-l', type=str, required=True, help='Path to the leaderboard CSV file')
    parser.add_argument('--output_file', '-o', type=str, default='./plots/cost_vs_winrate.png', help='Path for the output image file')
    parser.add_argument('--category', type=str, required=False, help='Benchmark category to filter by')
    args = parser.parse_args()
    main(
        args.costs_file,
        args.leaderboard_file,
        args.output_file,
        args.category
    )
