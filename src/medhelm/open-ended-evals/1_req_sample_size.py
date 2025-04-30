import numpy as np
from scipy import stats
import pandas as pd

def zou_icc_sample_size(expected_icc, width, k=2, assurance_prob=0.8, alpha=0.05):
    """
    Calculate sample size for ICC estimation based on Zou (2012).
    
    Parameters:
    - expected_icc: Expected ICC value (e.g., 0.65 for substantial reliability)
    - width: Desired width of the confidence interval
    - k: Number of raters per subject (default 2 for human vs LLM)
    - assurance_prob: Desired probability of achieving the precision (default 0.8)
    - alpha: Significance level (default 0.05)
    
    Returns:
    - Required sample size (number of subjects/texts)
    """
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(assurance_prob)
    
    # Formula based on desired width (Equation 5 in Zou's paper)
    A = (1 - expected_icc) * (1 + (k - 1) * expected_icc)
    B = k - 2 + 2 * expected_icc - 2 * k * expected_icc
    
    # Calculate N using the formula from Equation 5
    term = (A * z_alpha + np.sqrt(A**2 * z_alpha**2 + 4 * (width/2) * z_alpha * z_beta * A * abs(B))) / (width/2 * np.sqrt(2 * k * (k - 1)))
    n = 1 + term**2
    
    return np.ceil(n)

def apply_fpc(n, population_size):
    """
    Apply finite population correction to the sample size.
    
    Parameters:
    - n: Calculated sample size
    - population_size: Total population size (dataset size)
    
    Returns:
    - Adjusted sample size
    """
    return (population_size * n) / (population_size + n - 1)

# Define parameters
expected_icc = 0.75  # Substantial reliability
assurance = 0.8      # 80% assurance/power
alpha = 0.05         # Significance level

# Define datasets with their population sizes
datasets = {
    "MTSamples Replicate": 78,
    "MTSamples Procedures": 62,
    "MedAlign": 43,
    "MedicationQA": 689,
    "PatientInstruct": 361,
    "MedDialog": 1000,
    "MEDIQA": 150,
    "MentalHealth": 67,
    "DischargeMe": 1000,
    "ACI-Bench": 120,
    "MIMIC-RRS": 1000,
    "NoteExtract": 487
}

# Original required samples from your table (for comparison)
original_samples = {
    "MTSamples Replicate": 29,
    "MTSamples Procedures": 27,
    "MedAlign": 22,
    "MedicationQA": 44,
    "PatientInstruct": 42,
    "MedDialog": 45,
    "MEDIQA": 36,
    "MentalHealth": 28,
    "DischargeMe": 45,
    "ACI-Bench": 34,
    "MIMIC-RRS": 45,
    "NoteExtract": 43
}

# Test different width parameters
widths = [0.2, 0.25, 0.3, 0.35, 0.4]
width_results = {}

print("Sample size requirements for different confidence interval widths:")
print("=" * 80)

for width in widths:
    base_n = zou_icc_sample_size(expected_icc, width, assurance_prob=assurance, alpha=alpha)
    print(f"\nWidth = {width}:")
    print(f"Base sample size (before FPC): {int(base_n)}")
    
    # Calculate for each dataset
    adjusted_samples = {}
    for dataset, population in datasets.items():
        if base_n > population:
            adjusted_samples[dataset] = population
        else:
            adjusted_n = apply_fpc(base_n, population)
            adjusted_samples[dataset] = int(np.ceil(adjusted_n))
    
    width_results[width] = adjusted_samples
    
    # Print sample sizes for a few example datasets
    print(f"Example adjusted sample sizes (with FPC):")
    for dataset in ["MTSamples Replicate", "MEDIQA", "NoteExtract"]:
        print(f"  {dataset}: {adjusted_samples[dataset]}")

# Create and display a comprehensive comparison table
comparison_df = pd.DataFrame({'Total Samples': pd.Series(datasets),
                             'Original Required': pd.Series(original_samples)})

# Add columns for each width
for width in widths:
    comparison_df[f'Width {width}'] = pd.Series(width_results[width])

print("\n\nFull comparison across all datasets and width parameters:")
print(comparison_df)

# Find the width that gives sample sizes closest to the original
differences = {}
for width in widths:
    total_diff = sum(abs(width_results[width][dataset] - original_samples[dataset]) 
                    for dataset in datasets)
    differences[width] = total_diff

closest_width = min(differences, key=differences.get)
print(f"\nWidth parameter closest to your original sample sizes: {closest_width}")

# Calculate percentage differences for the closest width
closest_samples = width_results[closest_width]
comparison_df['Closest ICC Required'] = pd.Series(closest_samples)
comparison_df['Percentage Difference'] = ((comparison_df['Closest ICC Required'] - 
                                        comparison_df['Original Required']) / 
                                        comparison_df['Original Required'] * 100).round(1)

print("\nComparison with closest width parameter:")
print(comparison_df[['Total Samples', 'Original Required', 'Closest ICC Required', 'Percentage Difference']])

print("\nAssumptions:")
print(f"- Expected ICC: {expected_icc} (substantial reliability)")
print(f"- Most feasible confidence interval width: {closest_width}")
print(f"- Assurance probability: {assurance*100}%")
print(f"- Significance level: {alpha*100}%")
print(f"- Number of raters: 2 (human vs LLM)")
