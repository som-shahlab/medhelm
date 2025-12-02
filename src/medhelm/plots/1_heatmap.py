import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Set the font to a safe, standard sans-serif font (Arial or Helvetica)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


# Define all tasks as a compact list of tuples: (original column, display name, category)
METRIC_INFO = [
    ("MedCalc-Bench - MedCalc Accuracy", "MedCalc-Bench - MedCalc Accuracy", "Clinical Decision Support", "Supporting Diagnostic Decisions"),
    ("CLEAR - EM", "CLEAR - EM", "Clinical Decision Support", "Supporting Diagnostic Decisions"),
    ("MTSamples - Accuracy", "MTSamples - Jury Score", "Clinical Decision Support", "Planning Treatments"),
    ("Medec - MedecFlagAcc", "Medec - MedecFlagAcc", "Clinical Decision Support", "Planning Treatments"),
    ("EHRSHOT - EM", "EHRSHOT - EM", "Clinical Decision Support", "Predicting Patient Risks and Outcomes"),
    ("HeadQA - EM", "HeadQA - EM", "Clinical Decision Support", "Providing Clinical Knowledge Support"),
    ("Medbullets - EM", "Medbullets - EM", "Clinical Decision Support", "Providing Clinical Knowledge Support"),
    ("MedQA - EM", "MedQA - EM", "Clinical Decision Support", "Providing Clinical Knowledge Support"),
    ("MedMCQA - EM", "MedMCQA - EM", "Clinical Decision Support", "Providing Clinical Knowledge Support"),
    ("MedAlign - Accuracy", "MedAlign - Jury Score", "Clinical Decision Support", "Providing Clinical Knowledge Support"),
    ("ADHD-Behavior - EM", "ADHD-Behavior - EM", "Clinical Decision Support", "Providing Clinical Knowledge Support"),
    ("ADHD-MedEffects - EM", "ADHD-MedEffects - EM", "Clinical Decision Support", "Providing Clinical Knowledge Support"),

    # Clinical Note Generation
    ("DischargeMe - Accuracy", "DischargeMe - Jury Score", "Clinical Note Generation", "Documenting Patient Visits"),
    ("ACI-Bench - Accuracy", "ACI-Bench - Jury Score", "Clinical Note Generation", "Documenting Patient Visits"),
    ("MTSamples Procedures - Jury Score", "MTSamples Procedures - Jury Score", "Clinical Note Generation", "Recording Procedures"),
    ("MIMIC-RRS - Accuracy", "MIMIC-RRS - Jury Score", "Clinical Note Generation", "Documenting Diagnostic Reports"),
    ("MIMIC-BHC - Accuracy", "MIMIC-BHC - Jury Score", "Clinical Note Generation", "Documenting Patient Visits"),
    ("NoteExtract - Accuracy", "NoteExtract - Jury Score", "Clinical Note Generation", "Documenting Care Plans"),

    # Patient Communication and Education
    ("MedicationQA - Accuracy", "MedicationQA - Jury Score", "Patient Communication and Education", "Providing Patient Education Resources"),
    ("PatientInstruct - Accuracy", "PatientInstruct - Jury Score", "Patient Communication and Education", "Delivering Personalized Care Instructions"),
    ("MedDialog - Accuracy", "MedDialog - Jury Score", "Patient Communication and Education", "Patient-Provider Messaging"),
    ("MedConfInfo - EM", "MedConfInfo - EM", "Patient Communication and Education", "Patient-Provider Messaging"),
    ("MEDIQA - Accuracy", "MEDIQA - Jury Score", "Patient Communication and Education", "Enhancing Patient Understanding and Accessibility in Health Communication"),
    ("MentalHealth - Accuracy", "MentalHealth - Jury Score", "Patient Communication and Education", "Facilitating Patient Engagement and Support"),
    ("ProxySender - EM", "ProxySender - EM", "Patient Communication and Education", "Patient-Provider Messaging"),
    ("PrivacyDetection - EM", "PrivacyDetection - EM", "Patient Communication and Education", "Patient-Provider Messaging"),

    # Medical Research Assistance
    ("PubMedQA - EM", "PubMedQA - EM", "Medical Research Assistance", "Conducting Literature Research"),
    ("EHRSQL - EHRSQLExeAcc", "EHRSQL - EHRSQLExeAcc", "Medical Research Assistance", "Analyzing Clinical Research Data"),
    ("BMT-Status - EM", "BMT-Status - EM", "Medical Research Assistance", "Recording Research Processes"),
    ("RaceBias - EM", "RaceBias - EM", "Medical Research Assistance", "Ensuring Clinical Research Quality"),
    ("N2C2-CT - EM", "N2C2-CT - EM", "Medical Research Assistance", "Managing Research Enrollment"),
    ("MedHallu - EM", "MedHallu - EM", "Medical Research Assistance", "Ensuring Clinical Research Quality"),

    # Administration and Workflow
    ("HospiceReferral - EM", "HospiceReferral - EM", "Administration and Workflow", "Scheduling Resources and Staff"),
    ("MIMIC-IV Billing Code - MIMICBillingF1", "MIMIC-IV Billing Code - MIMICBillingF1", "Administration and Workflow", "Overseeing Financial Activities"),
    ("ClinicReferral - EM", "ClinicReferral - EM", "Administration and Workflow", "Organizing Workflow Processes"),
    ("CDI-QA - EM", "CDI-QA - EM", "Administration and Workflow", "Care Coordination and Planning"),
    ("ENT-Referral - EM", "ENT-Referral - EM", "Administration and Workflow", "Care Coordination and Planning"),
]


# Generate mappings
RENAME_MAP = {orig: display for orig, display, _, _ in METRIC_INFO}
CATEGORY_MAP = {}
SUBCATEGORY_MAP = {}
for _, display, cat, subcat in METRIC_INFO:
    CATEGORY_MAP.setdefault(cat, []).append(display)
    SUBCATEGORY_MAP.setdefault(subcat, []).append(display)


def normalize_scores(df, metric_cols):
    for col in metric_cols:
        if df[col].max() > 1:
            df[col] = (df[col] -1.0) / (5.0 - 1.0)
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

    if aggregated == "Category":
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
    elif aggregated == "Subcategory":
        df_agg = df.set_index("Model")[metric_cols].copy()
        
        # Map metrics to subcategories
        col_to_subcat = {}
        for subcat, cols in SUBCATEGORY_MAP.items():
            for col in cols:
                col_to_subcat[col] = subcat
        df_agg = df_agg.rename(columns=col_to_subcat)
        
        # Average by subcategory
        df_agg = df_agg.groupby(by=df_agg.columns, axis=1).mean()
        
        # Order subcategories by their parent category order
        # First get category order (list of categories)
        category_order = list(CATEGORY_MAP.keys())
        
        # Map subcategory to its parent category for sorting
        subcat_to_cat = {}
        for cat, cols in CATEGORY_MAP.items():
            for subcat in SUBCATEGORY_MAP.keys():
                # If any subcat columns are in this category, assign
                if any(col in cols for col in SUBCATEGORY_MAP[subcat]):
                    subcat_to_cat[subcat] = cat
        
        # Now create sorted list of subcategories by category order
        sorted_subcats = []
        for cat in category_order:
            # Subcats in this category
            cat_subcats = [sc for sc, c in subcat_to_cat.items() if c == cat]
            # Preserve original subcategory order (optional: sort alphabetically)
            cat_subcats_sorted = sorted(cat_subcats)
            sorted_subcats.extend(cat_subcats_sorted)

        # Filter to keep only subcategories actually in df_agg.columns and preserve order
        sorted_subcats = [sc for sc in sorted_subcats if sc in df_agg.columns]

        # Reorder columns
        df_agg = df_agg[sorted_subcats]

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
        cmap="viridis",
        vmin=0,
        vmax=1,
        cbar_kws={"ticks": np.linspace(0, 1, 11)},
        linewidths=0.5,
        linecolor='gray',
        annot_kws={"fontsize": 7}
    )

    plt.title(title, fontsize=14)
    plt.xticks(rotation=30, ha='right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    
    output_path_pdf = output_path.replace(".png", ".pdf")
    plt.savefig(output_path_pdf, dpi=500, bbox_inches='tight')
    print(f"Plot saved as '{output_path_pdf}'")
    
    return plt.gcf(), top_bottom_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard_path", type=str, default="../medhelm/data/leaderboard.csv")
    parser.add_argument("--output_path_image", type=str, default="../medhelm/plots/category_heatmap.png")
    parser.add_argument("--output_path_text", type=str, default="../medhelm/data/stats/subcategory_heatmap_test_analysis.txt")
    parser.add_argument("--aggregated", type=str, default="False")
    parser.add_argument("--transpose", action="store_true")
    args = parser.parse_args()
    if args.aggregated == "False":
        args.output_path_image = args.output_path_image.replace(".png", "_benchmark.png")
    elif args.aggregated == "Subcategory":
        args.output_path_image = args.output_path_image.replace(".png", "_subcategory.png")
    elif args.aggregated == "Category":
        args.output_path_image = args.output_path_image.replace(".png", "_category.png")
        
    fig, top_bottom_dict = plot_category_heatmap(
        df_path=args.leaderboard_path,
        output_path=args.output_path_image,
        aggregated=args.aggregated,
        transpose=args.transpose
    )
    plt.close()
    with open(args.output_path_text, 'w') as f:
        for model, info in top_bottom_dict.items():
            f.write(f"\nModel: {model}\n")
            if info["top"]:
                f.write(f"  Top 2 in: {', '.join(info['top'])}\n")
            if info["bottom"]:
                f.write(f"  Bottom 2 in: {', '.join(info['bottom'])}\n")
