"""
Modular multi-visualization script for benchmark data.

Generates 5 plot types from the same hardcoded dataset:
  1. Spider/Radar plot
  2. Grouped bar chart
  3. Bubble chart
  4. Treemap
  5. Heatmap
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import squarify

# ── Data ──────────────────────────────────────────────────────────────────────

DATA = [
    ["MEDHELM","MedCalc-Bench [ME20]","supporting diagnostic decisions",1000,1000],
    ["MEDHELM","CLEAR [ME21]","planning treatments",1022,1022],
    ["MEDHELM","MTSamples [ME22]","planning treatments",427,0],
    ["MEDHELM","Medec [ME23]","planning treatments",597,0],
    ["MEDHELM","EHRSHOT [ME24]","predicting patient risks and outcomes",1000,1000],
    ["MEDHELM","HeadQA [ME25]","providing clinical knowledge support",1000,0],
    ["MEDHELM","Medbullets [ME26]","providing clinical knowledge support",308,0],
    ["MEDHELM","MedQA [ME1]","providing clinical knowledge support",1000,0],
    ["MEDHELM","MedMCQA [MA7]","providing clinical knowledge support",1000,0],
    ["MEDHELM","MedAlign [ME27]","providing clinical knowledge support",149,149],
    ["MEDHELM","ADHD-Behavior [ME28]","providing clinical knowledge support",423,423],
    ["MEDHELM","ADHD-MedEffects [ME29]","providing clinical knowledge support",915,915],
    ["MEDHELM","DischargeMe [ME30]","documenting patient visits",1000,1000],
    ["MEDHELM","ACI-Bench [ME31]","documenting patient visits",120,120],
    ["MEDHELM","MTSamples Procedures [ME22]","recording procedures",128,0],
    ["MEDHELM","MIMIC-RRS [ME32]","documenting diagnostic reports",1000,1000],
    ["MEDHELM","MIMIC-BHC [ME33]","documenting patient visits",1000,1000],
    ["MEDHELM","NoteExtract","documenting care plans",487,487],
    ["MEDHELM","MedicationQA [ME34]","providing patient education resources",689,0],
    ["MEDHELM","PatientInstruct","delivering personalized care instructions",361,361],
    ["MEDHELM","MedDialog [ME35]","patient-provider messaging",1000,0],
    ["MEDHELM","MedConfInfo [ME36]","patient-provider messaging",1000,1000],
    ["MEDHELM","MEDIQA-QA [ME37]","enhancing patient understanding and accessibility in health",150,0],
    ["MEDHELM","MentalHealth","facilitating patient engagement and support",67,0],
    ["MEDHELM","PrivacyDetection [ME38]","patient-provider messaging",300,0],
    ["MEDHELM","ProxySender [ME38]","patient-provider messaging",300,0],
    ["MEDHELM","PubMedQA [ME39]","conducting literature research",1000,0],
    ["MEDHELM","EHRSQL [ME40]","analyzing clinical research data",1000,0],
    ["MEDHELM","BMT-Status","recording research processes",220,220],
    ["MEDHELM","RaceBias [ME41]","ensuring clinical research quality",167,0],
    ["MEDHELM","N2C2-CT [ME42]","managing research enrollment",86,86],
    ["MEDHELM","MedHallu [ME43]","ensuring clinical research quality",1000,0],
    ["MEDHELM","HospiceReferral","scheduling resources and staff",1000,1000],
    ["MEDHELM","MIMIC-IV Billing Code [ME44]","overseeing financial activities",1000,1000],
    ["MEDHELM","ClinicReferral","organizing workflow processes",326,326],
    ["MEDHELM","CDI-QA","care coordination and planning",1000,1000],
    ["MEDHELM","ENT-Referral","care coordination and planning",1000,1000],
    ["OTHER","First DoNoHarm","planning treatments",100,100],
    ["OTHER","SCT Bench","supporting diagnostic decisions",750,0],
    ["OTHER","CPC Bench","supporting diagnostic decisions",2437,2437],
    ["OTHER","CPC Bench","planning treatments",812,812],
    ["OTHER","CPC Bench","providing clinical knowledge support",1625,1625],
    ["OTHER","CPC Bench","analyzing clinical research data",812,812],
    ["OTHER","CPC Bench","conducting literature research",812,812],
    ["OTHER","CPC Bench","documenting diagnostic reports",812,812],
    ["OTHER","CPC Bench","providing patient education resources",813,813],
    ["OTHER","HealthBench","care coordination and planning",714,0],
    ["OTHER","HealthBench","organizing workflow processes",714,0],
    ["OTHER","HealthBench","providing patient education resources",714,0],
    ["OTHER","HealthBench","documenting diagnostic reports",714,0],
    ["OTHER","HealthBench","patient-provider messaging",714,0],
    ["OTHER","HealthBench","supporting diagnostic decisions",715,0],
    ["OTHER","HealthBench","providing clinical knowledge support",715,0],
    ["OTHER","MedAgentBench","supporting diagnostic decisions",85,28],
    ["OTHER","MedAgentBench","documenting patient visits",43,14],
    ["OTHER","MedAgentBench","documenting diagnostic reports",43,14],
    ["OTHER","MedAgentBench","predicting patient risks and outcomes",43,14],
    ["OTHER","MedAgentBench","care coordination and planning",43,15],
    ["OTHER","MedAgentBench","planning treatments",43,15],
]

COLUMNS = ["Dataset_Group", "Dataset", "Subcategory", "Questions", "Unique_Patients"]

DEFAULT_OUTPUT_DIR = "/share/pi/nigam/users/aunell/medhelm/plots"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_dataframe():
    df = pd.DataFrame(DATA, columns=COLUMNS)
    df["Subcategory"] = df["Subcategory"].str.title()
    df["Dataset"] = df["Dataset"].str.replace(r"\s*\[.*?\]", "", regex=True)
    return df


def ordered_subcategories(df):
    """Return subcategories in the order they first appear in the dataframe."""
    return list(df["Subcategory"].unique())


def save_plot(output_dir, name):
    """Save current figure as PNG, then close."""
    path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}.png")


def wrap_label(text, width=25):
    """Insert newlines into long label text."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if current and len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return "\n".join(lines)


# ── 1. Spider / Radar Plot ────────────────────────────────────────────────────

def _plot_spider_impl(df, output_dir, log_scale):
    agg = (
        df.groupby(["Dataset_Group", "Subcategory"], as_index=False)
          .agg({"Questions": "sum", "Unique_Patients": "sum"})
    )

    # Build a lookup of benchmark names per (group, subcategory)
    benchmark_names = (
        df.groupby(["Dataset_Group", "Subcategory"])["Dataset"]
          .apply(lambda x: ", ".join(sorted(x.unique())))
          .to_dict()
    )

    subcategories = ordered_subcategories(df)
    N = len(subcategories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Normalize dot size
    min_size, max_size = 80, 800
    max_patients = agg["Unique_Patients"].max()
    agg["Norm_Size"] = (
        (agg["Unique_Patients"] / max_patients) * (max_size - min_size) + min_size
    )

    # Normalize radial axis
    if log_scale:
        agg["Radial"] = np.log1p(agg["Questions"])
    else:
        agg["Radial"] = agg["Questions"].astype(float)
    max_radial = agg["Radial"].max()
    agg["Norm_Questions"] = agg["Radial"] / max_radial

    plt.figure(figsize=(16, 16))
    ax = plt.subplot(111, polar=True)

    for group in agg["Dataset_Group"].unique():
        gdf = agg[agg["Dataset_Group"] == group]
        values, sizes, labels = [], [], []
        for sub in subcategories:
            row = gdf[gdf["Subcategory"] == sub]
            if not row.empty:
                values.append(row["Norm_Questions"].values[0])
                sizes.append(row["Norm_Size"].values[0])
                labels.append(benchmark_names.get((group, sub), ""))
            else:
                values.append(0)
                sizes.append(min_size)
                labels.append("")
        values += values[:1]
        sizes += sizes[:1]
        labels += labels[:1]

        line, = ax.plot(angles, values, linewidth=2, label=group)
        color = line.get_color()
        ax.fill(angles, values, alpha=0.15, color=color)
        ax.scatter(angles, values, s=sizes, color=color, zorder=5)

        # Annotate each dot with contributing benchmark names
        for angle, val, lbl in zip(angles[:-1], values[:-1], labels[:-1]):
            if val > 0 and lbl:
                # Shorten long labels: show first 2 benchmarks + count
                names = lbl.split(", ")
                if len(names) > 2:
                    display = ", ".join(names[:2]) + f"\n+{len(names)-2} more"
                else:
                    display = ", ".join(names)
                ax.annotate(
                    display,
                    xy=(angle, val),
                    fontsize=5,
                    ha="center", va="bottom",
                    textcoords="offset points",
                    xytext=(0, 8),
                    alpha=0.85,
                )

    scale_label = "log scale" if log_scale else "linear scale"
    suffix = "log" if log_scale else "linear"

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([wrap_label(s) for s in subcategories], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), title="Dataset Group")
    plt.title(
        f"Dataset Coverage by Subcategory\n"
        f"(dot size = unique patients, radial = questions, {scale_label})",
        fontsize=14,
    )
    plt.tight_layout()
    save_plot(output_dir, f"3_spider_{suffix}")


def plot_spider(df, output_dir):
    _plot_spider_impl(df, output_dir, log_scale=True)
    _plot_spider_impl(df, output_dir, log_scale=False)


# ── 2. Grouped Bar Chart ─────────────────────────────────────────────────────

def plot_grouped_bar(df, output_dir):
    agg = (
        df.groupby(["Dataset_Group", "Subcategory"], as_index=False)
          .agg({"Questions": "sum", "Unique_Patients": "sum"})
    )

    # Build lookup of benchmark names per (group, subcategory)
    benchmark_names = (
        df.groupby(["Dataset_Group", "Subcategory"])["Dataset"]
          .apply(lambda x: ", ".join(sorted(x.unique())))
          .to_dict()
    )

    subcategories = ordered_subcategories(df)
    groups = ["MEDHELM", "OTHER"]
    n_subs = len(subcategories)
    x = np.arange(n_subs) * 1.4  # extra spacing between groups
    bar_width = 0.55

    fig, ax = plt.subplots(figsize=(max(24, n_subs * 1.8), 10))

    for i, group in enumerate(groups):
        gdf = agg[agg["Dataset_Group"] == group]
        questions = []
        patients = []
        names = []
        for sub in subcategories:
            row = gdf[gdf["Subcategory"] == sub]
            questions.append(row["Questions"].values[0] if not row.empty else 0)
            patients.append(row["Unique_Patients"].values[0] if not row.empty else 0)
            names.append(benchmark_names.get((group, sub), ""))

        questions = np.array(questions)
        patients = np.array(patients, dtype=float)
        patients = np.minimum(patients, questions)
        non_patient = questions - patients

        offset = (i - 0.5) * bar_width
        color = f"C{i}"

        # Bottom stack: patient notes portion (striped)
        ax.bar(x + offset, patients, bar_width,
               color=color, hatch="//", edgecolor="white", linewidth=0.5,
               zorder=3, label=group)
        # Top stack: non-patient portion (solid)
        ax.bar(x + offset, non_patient, bar_width, bottom=patients,
               color=color, zorder=3)

        # Annotate with benchmark names (wrap to two lines if long)
        for j, (xpos, nms) in enumerate(zip(x + offset, names)):
            if questions[j] > 0 and nms:
                label = wrap_label(nms, 30)
                ax.text(
                    xpos, questions[j] + 20, label,
                    ha="center", va="bottom", fontsize=9, rotation=90,
                )

    # Add legend entry for striped portion
    from matplotlib.patches import Patch
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="lightgray", edgecolor="white", hatch="//",
                         label="Patient notes portion"))
    ax.legend(handles=handles, title="Dataset Group", fontsize=12, title_fontsize=13)

    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(s, 20) for s in subcategories],
                       rotation=45, ha="right", fontsize=11)
    ax.set_ylabel("Total Questions", fontsize=14)
    ax.set_xlabel("Subcategory", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("Questions per Subcategory (MEDHELM vs OTHER)\nStriped area = patient notes",
                 fontsize=20)

    # Raise y-axis limit to fit annotation text
    ymax = ax.get_ylim()[1]
    ax.set_ylim(top=ymax * 1.45)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_plot(output_dir, "3_grouped_bar")


# ── 3. Bubble Chart ──────────────────────────────────────────────────────────

def plot_bubble(df, output_dir):
    subcats = ordered_subcategories(df)
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(subcats))
    color_map = {s: cmap(i) for i, s in enumerate(subcats)}

    fig, ax = plt.subplots(figsize=(14, 10))

    markers = {"MEDHELM": "o", "OTHER": "D"}

    for _, row in df.iterrows():
        group = row["Dataset_Group"]
        ax.scatter(
            row["Questions"],
            row["Unique_Patients"],
            s=120,
            c=[color_map[row["Subcategory"]]],
            marker=markers[group],
            edgecolors="black" if group == "OTHER" else "none",
            linewidths=0.8,
            alpha=0.8,
            zorder=3,
        )

    # Legend for subcategories (color)
    handles_sub = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[s], markersize=8, label=wrap_label(s, 30))
        for s in subcats
    ]
    leg1 = ax.legend(
        handles=handles_sub, title="Subcategory",
        bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, title_fontsize=8,
    )
    ax.add_artist(leg1)

    # Legend for meta-group (shape)
    handles_grp = [
        plt.Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                   markersize=8, label=g, markeredgecolor="black", markeredgewidth=0.5)
        for g, m in markers.items()
    ]
    ax.legend(
        handles=handles_grp, title="Meta-Group",
        bbox_to_anchor=(1.02, 0.35), loc="upper left", fontsize=8,
    )

    ax.set_xlabel("Questions Count")
    ax.set_ylabel("Unique Patient Notes")
    ax.set_title("Benchmarks: Questions vs Patients\n(shape = meta-group, color = subcategory)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_plot(output_dir, "3_bubble")


# ── 4. Treemap ────────────────────────────────────────────────────────────────

def plot_treemap(df, output_dir):
    # Each tile = one row in df; size = Questions
    plot_df = df[df["Questions"] > 0].copy()
    # Preserve dataframe order for subcategories
    sub_order = {s: i for i, s in enumerate(ordered_subcategories(df))}
    plot_df = plot_df.sort_values(
        ["Dataset_Group", "Subcategory", "Dataset"],
        key=lambda col: col.map(sub_order) if col.name == "Subcategory" else col,
    )

    sizes = plot_df["Questions"].values
    patients = plot_df["Unique_Patients"].values

    # Color by patient count (gradient)
    norm = mcolors.Normalize(vmin=0, vmax=patients.max())
    cmap = plt.cm.YlOrRd
    colors = [cmap(norm(p)) for p in patients]

    # Labels: short dataset name + group indicator
    labels = [
        f"{row['Dataset']}\n({row['Dataset_Group'][0]})"
        for _, row in plot_df.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(18, 10))
    squarify.plot(
        sizes=sizes, label=labels, color=colors, alpha=0.85,
        text_kwargs={"fontsize": 6, "wrap": True}, ax=ax,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Unique Patient Notes")

    ax.set_title(
        "Benchmark Treemap\n(tile size = questions, color = patient notes, M=MEDHELM / O=OTHER)",
        fontsize=13,
    )
    ax.axis("off")
    plt.tight_layout()
    save_plot(output_dir, "3_treemap")


# ── 5. Heatmap ────────────────────────────────────────────────────────────────

def plot_heatmap(df, output_dir):
    # Sort by subcategory then dataset for grouping
    # Preserve dataframe order for subcategories
    sub_order = {s: i for i, s in enumerate(ordered_subcategories(df))}
    plot_df = df.copy()
    plot_df["_sub_order"] = plot_df["Subcategory"].map(sub_order)
    plot_df = plot_df.sort_values(["_sub_order", "Dataset"]).drop(columns="_sub_order").reset_index(drop=True)

    # Create label combining dataset + group
    plot_df["Label"] = plot_df["Dataset"] + "  [" + plot_df["Dataset_Group"] + "]"

    # Build matrix: two columns (Questions, Patients), normalized independently
    q_vals = plot_df["Questions"].values.astype(float)
    p_vals = plot_df["Unique_Patients"].values.astype(float)
    q_norm = q_vals / q_vals.max() if q_vals.max() > 0 else q_vals
    p_norm = p_vals / p_vals.max() if p_vals.max() > 0 else p_vals
    matrix = np.column_stack([q_norm, p_norm])

    n_rows = len(plot_df)
    fig, (ax_grp, ax_heat) = plt.subplots(
        1, 2, figsize=(10, max(12, n_rows * 0.28)),
        gridspec_kw={"width_ratios": [0.08, 1]}, sharey=True,
    )

    # Side color bar for meta-group
    group_colors = {"MEDHELM": "#4C72B0", "OTHER": "#DD8452"}
    grp_matrix = np.array([
        [1 if g == "MEDHELM" else 0] for g in plot_df["Dataset_Group"]
    ])
    grp_cmap = mcolors.ListedColormap([group_colors["OTHER"], group_colors["MEDHELM"]])
    ax_grp.imshow(grp_matrix, aspect="auto", cmap=grp_cmap, interpolation="nearest")
    ax_grp.set_xticks([0])
    ax_grp.set_xticklabels(["Group"], fontsize=8)
    ax_grp.tick_params(axis="y", left=False)

    # Main heatmap
    im = ax_heat.imshow(matrix, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    ax_heat.set_xticks([0, 1])
    ax_heat.set_xticklabels(["Questions", "Patients"], fontsize=10)
    ax_heat.set_yticks(range(n_rows))
    ax_heat.set_yticklabels(plot_df["Label"].values, fontsize=7)

    # Add subcategory dividers
    prev_sub = None
    for i, sub in enumerate(plot_df["Subcategory"]):
        if prev_sub is not None and sub != prev_sub:
            ax_heat.axhline(y=i - 0.5, color="black", linewidth=0.8)
            ax_grp.axhline(y=i - 0.5, color="black", linewidth=0.8)
        prev_sub = sub

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.02)
    cbar.set_label("Normalized Value")

    # Meta-group legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=group_colors["MEDHELM"], label="MEDHELM"),
        Patch(facecolor=group_colors["OTHER"], label="OTHER"),
    ]
    ax_grp.legend(handles=legend_elements, loc="lower left", fontsize=7,
                  bbox_to_anchor=(0, -0.06))

    ax_heat.set_title(
        "Benchmark Heatmap\n(rows grouped by subcategory, lines separate groups)",
        fontsize=12,
    )
    plt.tight_layout()
    save_plot(output_dir, "3_heatmap")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = build_dataframe()
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print("Generating plots...")
    plot_spider(df, output_dir)
    plot_grouped_bar(df, output_dir)
    plot_bubble(df, output_dir)
    plot_treemap(df, output_dir)
    plot_heatmap(df, output_dir)
    print("Done – all plots saved to", output_dir)


if __name__ == "__main__":
    main()
