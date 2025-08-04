import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse

METRIC_INFO = [
    # Clinical Decision Support
    ("MedCalc-Bench - MedCalc Accuracy", "MedCalc-Bench - MedCalc Accuracy", "Supporting Diagnostic Decisions"),
    ("CLEAR - EM", "CLEAR - EM", "Supporting Diagnostic Decisions"),
    ("MTSamples - Accuracy", "MTSamples - Jury Score", "Planning Treatments"),
    ("Medec - MedecFlagAcc", "Medec - MedecFlagAcc", "Planning Treatments"),
    ("EHRSHOT - EM", "EHRSHOT - EM", "Predicting Patient Risks and Outcomes"),
    ("HeadQA - EM", "HeadQA - EM", "Providing Clinical Knowledge Support"),
    ("Medbullets - EM", "Medbullets - EM", "Providing Clinical Knowledge Support"),
    ("MedQA - EM", "MedQA - EM", "Providing Clinical Knowledge Support"),
    ("MedMCQA - EM", "MedMCQA - EM", "Providing Clinical Knowledge Support"),
    ("MedAlign - Accuracy", "MedAlign - Jury Score", "Providing Clinical Knowledge Support"),
    ("ADHD-Behavior - EM", "ADHD-Behavior - EM", "Providing Clinical Knowledge Support"),
    ("ADHD-MedEffects - EM", "ADHD-MedEffects - EM", "Providing Clinical Knowledge Support"),

    # Clinical Note Generation
    ("DischargeMe - Accuracy", "DischargeMe - Jury Score", "Documenting Patient Visits"),
    ("ACI-Bench - Accuracy", "ACI-Bench - Jury Score", "Documenting Patient Visits"),
    ("MTSamples Procedures - Jury Score", "MTSamples Procedures - Jury Score", "Recording Procedures"),
    ("MIMIC-RRS - Accuracy", "MIMIC-RRS - Jury Score", "Documenting Diagnostic Reports"),
    ("MIMIC-BHC - Accuracy", "MIMIC-BHC - Jury Score", "Documenting Patient Visits"),
    ("NoteExtract - Accuracy", "NoteExtract - Jury Score", "Documenting Care Plans"),

    # Patient Communication and Education
    ("MedicationQA - Accuracy", "MedicationQA - Jury Score", "Providing Patient Education Resources"),
    ("PatientInstruct - Accuracy", "PatientInstruct - Jury Score", "Delivering Personalized Care Instructions"),
    ("MedDialog - Accuracy", "MedDialog - Jury Score", "Patient-Provider Messaging"),
    ("MedConfInfo - EM", "MedConfInfo - EM", "Patient-Provider Messaging"),
    ("MEDIQA - Accuracy", "MEDIQA - Jury Score", "Enhancing Patient Understanding and Accessibility in Health Communication"),
    ("MentalHealth - Accuracy", "MentalHealth - Jury Score", "Facilitating Patient Engagement and Support"),
    ("ProxySender - EM", "ProxySender - EM", "Patient-Provider Messaging"),
    ("PrivacyDetection - EM", "PrivacyDetection - EM", "Patient-Provider Messaging"),

    # Medical Research Assistance
    ("PubMedQA - EM", "PubMedQA - EM", "Conducting Literature Research"),
    ("EHRSQL - EHRSQLExeAcc", "EHRSQL - EHRSQLExeAcc", "Analyzing Clinical Research Data"),
    ("BMT-Status - EM", "BMT-Status - EM", "Recording Research Processes"),
    ("RaceBias - EM", "RaceBias - EM", "Ensuring Clinical Research Quality"),
    ("N2C2-CT - EM", "N2C2-CT - EM", "Managing Research Enrollment"),
    ("MedHallu - EM", "MedHallu - EM", "Ensuring Clinical Research Quality"),

    # Administration and Workflow
    ("HospiceReferral - EM", "HospiceReferral - EM", "Scheduling Resources and Staff"),
    ("MIMIC-IV Billing Code - MIMICBillingF1", "MIMIC-IV Billing Code - MIMICBillingF1", "Overseeing Financial Activities"),
    ("ClinicReferral - EM", "ClinicReferral - EM", "Organizing Workflow Processes"),
    ("CDI-QA - EM", "CDI-QA - EM", "Care Coordination and Planning"),
    ("ENT-Referral - EM", "ENT-Referral - EM", "Care Coordination and Planning"),
]


# Generate mappings
RENAME_MAP = {orig: display for orig, display, _ in METRIC_INFO}
CATEGORY_MAP = {}
for _, display, cat in METRIC_INFO:
    CATEGORY_MAP.setdefault(cat, []).append(display)


def normalize_scores(df, metric_cols):
    for col in metric_cols:
        if df[col].max() > 1:
            df[col] = (df[col] - 1.0) / (5.0 - 1.0)
    return df

def get_top_bottom_per_model(df, metric_cols, top_k=2, bottom_k=2):
    """Return a dictionary of models to the metrics where they are top_k and/or bottom_k."""
    top_dict = {model: {"top": [], "bottom": []} for model in df["Model"]}
    
    for metric in metric_cols:
        sorted_df = df.sort_values(metric, ascending=False).reset_index(drop=True)
        top_models = sorted_df.head(top_k)["Model"].tolist()
        bottom_models = sorted_df.tail(bottom_k)["Model"].tolist()

        for model in top_models:
            top_dict[model]["top"].append(metric)
        for model in bottom_models:
            top_dict[model]["bottom"].append(metric)

    return top_dict

# Wrap long labels for better display
def wrap_labels(labels, max_width=25):
    return ['\n'.join(label[i:i+max_width] for i in range(0, len(label), max_width)) for label in labels]


def plot_category_heatmap(df_path, output_path, aggregated=False, transpose=False):
    df = pd.read_csv(df_path)
    metric_cols = [v for v in RENAME_MAP.values() if v in df.columns]
    if len(metric_cols) != len(RENAME_MAP):
        print(f"Warning: {len(metric_cols)} metrics found in the dataframe, expected {len(RENAME_MAP)}")
    df = normalize_scores(df, metric_cols)
    top_bottom_dict = get_top_bottom_per_model(df, metric_cols)

    if aggregated:
        df_agg = df.set_index("Model")[metric_cols].copy()
        col_to_cat = {}
        for cat, cols in CATEGORY_MAP.items():
            for col in cols:
                col_to_cat[col] = cat
        df_agg = df_agg.rename(columns=col_to_cat)
        df_agg = df_agg.groupby(by=df_agg.columns, axis=1).mean()
        heatmap_df = df_agg
        title = "Mean Normalized Scores by Model and Subcategory"
        plt.figure(figsize=(14, 6))
    else:
        heatmap_df = df.set_index("Model")[metric_cols]
        if transpose:
            heatmap_df = heatmap_df.T
            plt.figure(figsize=(10, 14))
        else:
            # Wrap x-axis labels to improve readability
            heatmap_df.columns = wrap_labels(heatmap_df.columns, max_width=25)
            plt.figure(figsize=(24, 8))
        title = "Individual Benchmark Scores by Model"

    bounds = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1] if not aggregated else [0, .5, .6, .7, .8, .9, 1]

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        cbar_kws={"ticks": bounds},
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"fontsize": 7}
    )

    plt.title(title, fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    return plt.gcf(), top_bottom_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard_path", type=str, default="../medhelm/data/leaderboard.csv")
    parser.add_argument("--output_path", type=str, default="../medhelm/plots/category_heatmap_subcategory.png")
    parser.add_argument("--aggregated", action="store_true")
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    fig, top_bottom_dict = plot_category_heatmap(
        df_path=args.leaderboard_path,
        output_path=args.output_path,
        aggregated=args.aggregated,
        transpose=args.transpose
    )
    plt.close()
    for model, info in top_bottom_dict.items():
        print(f"\nModel: {model}")
        if info["top"]:
            print("  Top 2 in:", ', '.join(info["top"]))
        if info["bottom"]:
            print("  Bottom 2 in:", ', '.join(info["bottom"]))
