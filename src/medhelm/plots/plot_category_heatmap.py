import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse


# Define all tasks as a compact list of tuples: (original column, display name, category)
METRIC_INFO = [
    # Clinical Decision Support
    ("MedCalc-Bench - MedCalc Acc...", "MedCalc-Bench - MedCalcAcc", "Clinical Decision Support"),
    ("CLEAR - EM", "CLEAR - EM", "Clinical Decision Support"),
    ("MTSamples - Accuracy", "MTSamples - Jury Score", "Clinical Decision Support"),
    ("Medec - MedecFlagAcc", "Medec - MedecFlagAcc", "Clinical Decision Support"),
    ("EHRSHOT - EM", "EHRSHOT - EM", "Clinical Decision Support"),
    ("HeadQA - EM", "HeadQA - EM", "Clinical Decision Support"),
    ("Medbullets - EM", "Medbullets - EM", "Clinical Decision Support"),
    ("MedAlign - Accuracy", "MedAlign - Jury Score", "Clinical Decision Support"),
    ("ADHD-Behavior - EM", "ADHD-Behavior - EM", "Clinical Decision Support"),
    ("ADHD-MedEffects - EM", "ADHD-MedEffects - EM", "Clinical Decision Support"),
    ("CDI-QA - EM", "CDI-QA - EM", "Clinical Decision Support"),

    # Clinical Note Generation
    ("ACI-Bench - Accuracy", "ACI-Bench - Jury Score", "Clinical Note Generation"),
    ("MTSamples Procedures - Accu...", "MTSamples Procedures - Jury Score", "Clinical Note Generation"),
    ("NoteExtract - Accuracy", "NoteExtract - Jury Score", "Clinical Note Generation"),
    ("MIMIC-RRS - Accuracy", "MIMIC-RRS - Jury Score", "Clinical Note Generation"),
    ("DischargeMe - Accuracy", "DischargeMe - Jury Score", "Clinical Note Generation"),
    ("MIMIC-BHC - Accuracy", "MIMIC-BHC - Jury Score", "Clinical Note Generation"),

    # Patient Communication and Education
    ("MedicationQA - Accuracy", "MedicationQA - Jury Score", "Patient Communication and Education"),
    ("PatientInstruct - Accuracy", "PatientInstruct - Jury Score", "Patient Communication and Education"),
    ("MedDialog - Accuracy", "MedDialog - Jury Score", "Patient Communication and Education"),
    ("MedConfInfo - EM", "MedConfInfo - EM", "Patient Communication and Education"),
    ("MEDIQA - Accuracy", "MEDIQA - Jury Score", "Patient Communication and Education"),
    ("MentalHealth - Accuracy", "MentalHealth - Jury Score", "Patient Communication and Education"),
    ("ProxySender - EM", "ProxySender - EM", "Patient Communication and Education"),
    ("PrivacyDetection - EM", "PrivacyDetection - EM", "Patient Communication and Education"),

    # Medical Research Assistance
    ("EHRSQL - EHRSQLExeAcc", "EHRSQL - EHRSQLExecAcc ", "Medical Research Assistance"),
    ("BMT-Status - EM", "BMT-Status - EM", "Medical Research Assistance"),
    ("RaceBias - EM", "RaceBias - EM", "Medical Research Assistance"),
    ("N2C2-CT - EM", "N2C2-CT - EM", "Medical Research Assistance"),
    ("MedHallu - EM", "MedHallu - EM", "Medical Research Assistance"),
    ("PubMedQA - EM", "PubMedQA - EM", "Medical Research Assistance"),

    # Administration and Workflow
    ("HospiceReferral - EM", "HospiceReferral - EM", "Administration and Workflow"),
    ("MIMIC-IV Billing Code - MIM...", "MIMIC-IV Billing Code - EM", "Administration and Workflow"),
    ("ClinicReferral - EM", "ClinicReferral - EM", "Administration and Workflow"),
    ("ENT-Referral - EM", "ENT-Referral - EM", "Administration and Workflow"),
]

# Generate mappings
RENAME_MAP = {orig: display for orig, display, _ in METRIC_INFO}
CATEGORY_MAP = {}
for _, display, cat in METRIC_INFO:
    CATEGORY_MAP.setdefault(cat, []).append(display)


def normalize_scores(df, metric_cols):
    for col in metric_cols:
        if df[col].max() > 1:
            df[col] = df[col] / 5.0
    return df


def plot_category_heatmap(df_path, output_path, aggregated=False, transpose=False):
    df = pd.read_csv(df_path)
    df = df.rename(columns=RENAME_MAP)

    metric_cols = [v for v in RENAME_MAP.values() if v in df.columns]
    if len(metric_cols) != len(RENAME_MAP):
        print(f"Warning: {len(metric_cols)} metrics found in the dataframe, expected {len(RENAME_MAP)}")
    df = normalize_scores(df, metric_cols)

    if aggregated:
        df_agg = df.set_index("Model")[metric_cols].copy()
        col_to_cat = {}
        for cat, cols in CATEGORY_MAP.items():
            for col in cols:
                col_to_cat[col] = cat
        df_agg = df_agg.rename(columns=col_to_cat)
        df_agg = df_agg.groupby(by=df_agg.columns, axis=1).mean()
        heatmap_df = df_agg
        title = "Mean Normalized Scores by Model and Category"
        plt.figure(figsize=(14, 6))
    else:
        heatmap_df = df.set_index("Model")[metric_cols]
        if transpose:
            heatmap_df = heatmap_df.T
            plt.figure(figsize=(10, 14))
        else:
            plt.figure(figsize=(20, 8))
        title = "Individual Benchmark Scores by Model"

    bounds = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1] if not aggregated else [0, .5, .6, .7, .8, .9, 1]

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        cbar_kws={"ticks": bounds},
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    return plt.gcf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard_path", type=str, default="../medhelm/data/leaderboard.csv")
    parser.add_argument("--output_path", type=str, default="../medhelm/plots/category_heatmap_new.png")
    parser.add_argument("--aggregated", action="store_true")
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()

    plot_category_heatmap(
        df_path=args.leaderboard_path,
        output_path=args.output_path,
        aggregated=args.aggregated,
        transpose=args.transpose
    )
    plt.close()
