import argparse
import pandas as pd
import matplotlib.pyplot as plt

from medhelm.utils.constants import BENCHMARKS, BENCHMARK_NAME_MAPPING, MODEL_NAME_MAPPING


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Set the font to a safe, standard sans-serif font (Arial or Helvetica)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


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

    # Define provider for each model
    provider_mapping = {
        "Claude 3.5 Sonnet (20241022)": "Anthropic",
        "Claude 3.7 Sonnet (20250219)": "Anthropic",
        "Gemini 1.5 Pro (001)": "Google",
        "Gemini 2.0 Flash": "Google",
        "GPT-4o (2024-05-13)": "OpenAI",
        "GPT-4o mini (2024-07-18)": "OpenAI",
        "o3-mini (2025-01-31)": "OpenAI",
        "Llama 3.3 Instruct (70B)": "Meta",
        "DeepSeek R1": "DeepSeek"
    }
    
    # Add provider information to the dataframe
    merged_data['Provider'] = merged_data.index.map(provider_mapping)
    
    # Get unique providers
    providers = merged_data['Provider'].unique()
    
    # Assign base colors from tab10 to each provider
    provider_colors = {}
    tab10_colors = plt.cm.tab10.colors
    for i, provider in enumerate(providers):
        provider_colors[provider] = tab10_colors[i % len(tab10_colors)]
    
    # Assign colors to models based on their provider
    merged_data['Color'] = merged_data['Provider'].map(provider_colors)
    
    # Define marker shapes to distinguish models from the same provider
    # Available markers: 'o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd', '|', '_'
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Assign marker shapes to models
    model_markers = {}
    for provider in providers:
        provider_models = merged_data[merged_data['Provider'] == provider].index.tolist()
        
        # Assign different marker shapes to models from the same provider
        for j, model in enumerate(provider_models):
            model_markers[model] = marker_styles[j % len(marker_styles)]
    
    # Add marker information to the dataframe
    merged_data['Marker'] = merged_data.index.map(model_markers)

    plt.figure(figsize=(12, 8))
    
    # Create custom handles for the legend, sorted by provider
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # First, sort the dataframe by provider to group models from the same provider
    sorted_data = merged_data.sort_values(by=['Provider', 'Mean Win Rate'], ascending=[True, False])
    
    # Plot each point with its assigned color and marker
    for model, row in sorted_data.iterrows():
        plt.scatter(
            row['Aggregated Cost'], 
            row['Mean Win Rate'], 
            color=row['Color'],
            marker=row['Marker'],
            edgecolors='black', 
            s=200,
            label=model
        )
        legend_elements.append(Line2D([0], [0], marker=row['Marker'], color='w', 
                                      markerfacecolor=row['Color'],
                                      markeredgecolor='black', markersize=12, 
                                      label=f"{model} [{row['Provider']}]"))

    title = 'Cost vs Mean Win Rate'
    if category:
        title = f'{title} ({category})'
    plt.title(title, fontsize=16)
    plt.xlabel('Cost (USD)', fontsize=14)
    plt.ylabel('Mean Win Rate', fontsize=14)
    plt.grid(True)
    
    # Use custom legend elements
    plt.legend(handles=legend_elements, title="Model [Provider]", fontsize=10, title_fontsize=12, loc='best')
    
    plt.tight_layout()

    plt.xlim(merged_data['Aggregated Cost'].min() * 0.9, merged_data['Aggregated Cost'].max() * 1.1)
    plt.ylim(merged_data['Mean Win Rate'].min() * 0.9, merged_data['Mean Win Rate'].max() * 1.1)

    plt.savefig(output_file, dpi=500, bbox_inches='tight')
    print(f"Plot saved as '{output_file}'")
    
    output_file_pdf = output_file.replace(".png", ".pdf")
    plt.savefig(output_file_pdf, dpi=500, bbox_inches='tight')
    print(f"Plot saved as '{output_file_pdf}'")


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