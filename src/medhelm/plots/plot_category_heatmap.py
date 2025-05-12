import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import argparse


def plot_category_heatmap(
    df_path: str,
    output_path: str,
    aggregated=False
):
    df = pd.read_csv(df_path)
    
    # Define category mappings
    category_mappings = {
        'Clinical Decision Support': [
            'MedCalc-Bench - MedCalc Acc...',
            'CLEAR - EM',
            'HeadQA - EM',
            'Medbullets - EM',
            'CDI-QA - EM',
            'Medec - MedecFlagAcc',
            'MTSamples - Jury Score',
            'EHRSHOT - EM',
            'MedAlign - Jury Score',
            'ADHD-Behavior - EM',
            'ADHD-MedEffects - EM',
        ],
        'Clinical Note Generation': [
            'ACI-Bench - Jury Score',
            'MTSamples Procedures - Accu...',
            'NoteExtract - Jury Score',
            'MIMIC-RRS - Jury Score',
            'DischargeMe - Jury Score',
            'MIMIC-BHC - Jury Score'
        ],
        'Patient Communication and Education': [
            'MedicationQA - Jury Score',
            'MEDIQA - Jury Score',
            'MentalHealth - Jury Score',
            'MedDialog - Jury Score',
            'MedConfInfo - EM',
            'PatientInstruct - Jury Score',
            'ProxySender - EM',
            'PrivacyDetection - EM',
        ],
        'Medical Research Assistance': [
            'EHRSQL - EHRSQLExeAcc',
            'BMT-Status - EM',
            'N2C2-CT - EM',
            'RaceBias - EM',
            'PubMedQA - EM',
            'MedHallu - EM'
        ],
        'Administration and Workflow': [
            'HospiceReferral - EM',
            'MIMIC-IV Billing Code - MIM...',
            'ClinicReferral - EM',
            'CDI-QA - EM',
            'ENT-Referral - EM',
        ]
    }

    if aggregated:
        # Aggregated view (categories)
        results = {}
        for model in df['Model']:
            results[model] = {}
            for category, metrics in category_mappings.items():
                scores = []
                for metric in metrics:
                    if metric in df.columns:
                        score = df.loc[df['Model'] == model, metric].iloc[0]
                        # Convert percentage scores to 0-1 scale
                        if score > 1:
                            score = score / 5.0  # Assuming 5-point scale
                        scores.append(score)
                results[model][category] = np.mean(scores)
        
        heatmap_df = pd.DataFrame(results).T
        plt.figure(figsize=(14, 6))
        title = "Mean Normalized Scores by Model and Category"
    else:
        # Non-aggregated view (all datasets)
        # Get all columns except 'Model' and 'Mean win rate'
        # Get metric columns and rename any with "Accuracy" to "Jury Score"
        # Get metric columns and standardize naming
        metric_columns = [col for col in df.columns if col not in ['Model', 'Mean win rate']]
        replacements = {
            'Accuracy': 'Jury Score',
            'MIM...': 'Jury Score', 
            'Accu...': 'Jury Score',
            'MedCalc Acc...': 'MedCalc Accuracy'
        }
        for old, new in replacements.items():
            metric_columns = [col.replace(old, new) for col in metric_columns]
        print(metric_columns)
        
        # Rename columns in dataframe
        rename_dict = {}
        for col in df.columns:
            if 'Accuracy' in col:
                rename_dict[col] = col.replace('Accuracy', 'Jury Score')
            elif 'MIM...' in col:
                rename_dict[col] = col.replace('MIM...', 'Jury Score')
            elif 'Accu...' in col:
                rename_dict[col] = col.replace('Accu...', 'Jury Score')
            elif 'MedCalc Acc...' in col:
                rename_dict[col] = col.replace('MedCalc Acc...', 'MedCalc Accuracy')
        df = df.rename(columns=rename_dict)
        
        # Normalize scores
        print(df.columns)
        for col in metric_columns:
            print(col)
            if df[col].max() > 1:
                df[col] = df[col] / 5.0
        
        # Prepare data for heatmap
        heatmap_df = df.set_index('Model')[metric_columns]
        print(heatmap_df)
        plt.figure(figsize=(20, 8))  # Larger figure for more columns
        title = "Individual Benchmark Scores by Model"

    # Color scheme setup
    colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
    bounds = [0, .5, .6, .7, .8, .9, 1]
    if not aggregated:
        colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"]
        bounds = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

    # Plot
    ax = sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn", #use RdYlGn for continuous color scheme, cmap for discrete color scheme
        # norm=norm,
        cbar_kws={"ticks": bounds},
        linewidths=0.5,
        linecolor='gray',
        square=False
    )

    # Labeling
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    
    return plt.gcf()

if __name__ == "__main__":
    # Example usage:
    # Aggregated view (default)
    args = argparse.ArgumentParser()
    args.add_argument("--leaderboard_path", type=str, default="/share/pi/nigam/users/aunell/medhelm/data/leaderboard.csv")
    args.add_argument("--output_path", type=str, default="../medhelm/plots/category_heatmap_AGG.png")
    args.add_argument("--aggregated", type=bool, default=True)
    args = args.parse_args()
    plot_category_heatmap(df_path=args.leaderboard_path, output_path=args.output_path, aggregated=args.aggregated)
    plt.close()
    