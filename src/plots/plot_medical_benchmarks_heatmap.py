import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import argparse

"""
This script generates a heatmap of language model performance on various medical benchmarks.

Usage:
    python plot_medical_benchmarks_heatmap.py --input <input_csv_file> --output <output_image_file>

The input CSV comes from copying the leaderboard into a csv file.
"""


# Add command-line argument parsing
parser = argparse.ArgumentParser(description='Generate a heatmap of language model performance on medical benchmarks.')
parser.add_argument('--input', '-i', type=str, required=True, help='Path to the input CSV file')
parser.add_argument('--output', '-o', type=str, default='medical_benchmarks_heatmap.png', help='Path for the output image file')
args = parser.parse_args()

# Define metric ranges
metric_ranges = {
    "MedCalc-Bench - MedCalc Acc...": [0, 1],
    "CLEAR - EM": [0, 1],
    "MTSamples - Accuracy": [1, 5],
    "Medec - MedecFlagAcc": [0, 1],
    "EHRSHOT - EM": [0, 1],
    "HeadQA - EM": [0, 1],
    "Medbullets - EM": [0, 1],
    "MedAlign - Accuracy": [1, 5],
    "ADHD-Behavior - EM": [0, 1],
    "ADHD-MedEffects - EM": [0, 1],
    "DischargeMe - Accuracy": [1, 5],
    "ACI-Bench - Accuracy": [1, 5],
    "MTSamples Procedures - Accu...": [1, 5],
    "MIMIC-RRS - Accuracy": [1, 5],
    "MIMIC-BHC - Accuracy": [1, 5],
    "NoteExtract - Accuracy": [1, 5],
    "MedicationQA - Accuracy": [1, 5],
    "PatientInstruct - Accuracy": [1, 5],
    "MedDialog - Accuracy": [1, 5],
    "MedConfInfo - EM": [0, 1],
    "MEDIQA - Accuracy": [1, 5],
    "MentalHealth - Accuracy": [1, 5],
    "PubMedQA - EM": [0, 1],
    "EHRSQL - EHRSQLExeAcc": [0, 1],
    "BMT-Status - EM": [0, 1],
    "RaceBias - EM": [0, 1],
    "N2C2-CT - EM": [0, 1],
    "HospiceReferral - EM": [0, 1],
    "MIMIC-IV Billing Code - MIM...": [0, 1],
    "ClinicReferral - EM": [0, 1],
    "CDI-QA - EM": [0, 1],
    "ENT-Referral - EM": [0, 1]
}

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

plt.figure(figsize=(16, 10))

df_norm = df[benchmark_columns].copy()
for col in df_norm.columns:
    min_val = metric_ranges[col][0]
    max_val = metric_ranges[col][1]
    df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

ax = sns.heatmap(df_norm, annot=df[benchmark_columns].values, cmap="RdYlGn", 
                 fmt='.2f', linewidths=.5, vmin=0, vmax=1,
                 cbar_kws={'label': 'Normalized Score'})

ax.set_xticklabels([short_names[col] for col in benchmark_columns], rotation=45, ha='right', fontsize=9)

ax.set_title('Language Model Performance on Medical Benchmarks', fontsize=16)
ax.set_xlabel('Benchmark', fontsize=14)
ax.set_ylabel('Language Models', fontsize=14)

plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=10)

plt.tight_layout()
output_file = args.output
plt.savefig(output_file, bbox_inches='tight', dpi=300)

print(f"Heatmap saved to {output_file}")