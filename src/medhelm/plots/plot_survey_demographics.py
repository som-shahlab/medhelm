import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


def main(input_file: str, output_dir: str):
    # Sort by count
    df = pd.read_csv(input_file)
    df = df.dropna(subset=['Attempted'])
    speciality_counts = df['Speciality'].value_counts()

    colors = plt.get_cmap('tab20').colors
    num_bars = len(speciality_counts)
    bar_colors = [colors[i % 20] for i in range(num_bars)]  # Cycle through if more than 10

    plt.figure(figsize=(6, 4))
    sns.barplot(
        x=speciality_counts.values,
        y=speciality_counts.index,
        palette=bar_colors
    )
    plt.xlabel('Number of Participants')
    plt.ylabel('Speciality')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speciality_counts.png'), dpi=300)
    print(f"Plot saved to {os.path.join(output_dir, 'speciality_counts.png')}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates plots for clinician demographics.')
    parser.add_argument('--input_file', '-i', type=str, required=True, help='Path to the survey demographics CSV file')
    parser.add_argument('--output_dir', '-o', type=str, default='./plots/cost_vs_winrate.png', help='Path for the output image file')
    args = parser.parse_args()
    main(
        args.input_file,
        args.output_dir,
    )