import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from medhelm.utils.constants import METRIC_RANGES


"""
This script generates a heatmap of language model performance on various medical benchmarks.

Usage:
    python plot_medical_benchmarks_heatmap.py --input <input_csv_file> --output <output_image_file>

The input CSV comes from copying the leaderboard into a csv file.
"""


# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Generate a heatmap of language model performance on medical benchmarks.')
parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output', '-o', type=str, default='medhelm_heatmap.png', help='Path for the output image file')
args = parser.parse_args()

df = pd.read_csv(args.input)
df = df.set_index('Model')

benchmark_columns = [col for col in df.columns if col != 'Mean win rate']

short_names = {}
for col in benchmark_columns:
    match = re.match(r'^(.*?)( -|$)', col)
    if match:
        short_name = match.group(1).strip()
        short_names[col] = short_name
    else:
        short_names[col] = col

df[benchmark_columns] = df[benchmark_columns].replace("-", 0)

df[benchmark_columns] = df[benchmark_columns].apply(pd.to_numeric)

plt.figure(figsize=(20, 8))

df_norm = df[benchmark_columns].copy()
for col in df_norm.columns:
    min_val = METRIC_RANGES[col][0]
    max_val = METRIC_RANGES[col][1]
    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

ax = sns.heatmap(
    df_norm,
    annot=df_norm.values,
    cmap="RdYlGn", 
    fmt='.2f',
    linewidths=.5,
    vmin=0,
    vmax=1,
    cbar_kws={'label': 'Normalized Score'},
    annot_kws={"size": 7, "color": "black"}
)

ax.set_xticklabels([short_names[col] for col in benchmark_columns], rotation=45, ha='right', fontsize=9)

ax.set_title('Language Model Performance on Medical Benchmarks', fontsize=16)
ax.set_xlabel('Benchmark', fontsize=14)
ax.set_ylabel('Language Model', fontsize=14)

plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=10)

plt.tight_layout()
output_file = args.output
plt.savefig(output_file, bbox_inches='tight', dpi=300)

print(f"Heatmap saved to {output_file}")