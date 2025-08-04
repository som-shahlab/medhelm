#!/usr/bin/env python3
"""
plot_win_rate_table.py
------------------
Benchmark-wide leaderboard on the raw 0-1 scale, but ranking by
pairwise win rates instead of "best-on-benchmark" counts.

For each model we now report
  • win_rate  - mean of pairwise win rates vs every other model
                (taken directly from the "Mean win rate" column in the CSV)
  • win_sd    - SD of those pairwise win rates
  • macro_avg - mean score across all benchmarks
  • sd        - SD of those scores
"""
import sys, re
import argparse
from pathlib import Path
import pandas as pd
import numpy as np           

def clean(col):
    """Strip metric suffix after first ' - '."""
    return re.sub(r"\s+", " ", col.split(" - ", 1)[0]).strip()

def rescale_1_to_5(series: pd.Series) -> pd.Series:
    """Detect 1-5 Likert columns and rescale to 0-1."""
    if series.dropna().between(0, 1).all():
        return series               # already 0-1
    if series.dropna().between(1, 5).all():
        return (series - 1) / 4.0   # Likert → true 0-1 range
    return series                   # leave other ranges unchanged

# ------------------------------------------------------------------ main
def main(csv_path, output_path=None):
    df = pd.read_csv(csv_path)

    # clean any benchmark column names
    df = df.rename(columns={c: clean(c)
                            for c in df.columns
                            if c not in ("Model", "Mean win rate")})

    score_cols = [c for c in df.columns if c not in ("Model", "Mean win rate")]

    # rescale 1-5 → 0-1 where needed
    for col in score_cols:
        df[col] = rescale_1_to_5(df[col])

    # use the provided mean win rate
    df["win_rate"] = df["Mean win rate"]

    # compute pairwise win-rate sd
    models = df["Model"].tolist()
    pairwise_sd = []
    for i, row in df.iterrows():
        rates = []
        for j, other in df.iterrows():
            if i == j:
                continue
            # fraction of benchmarks where model i beats model j
            wins_ij = (row[score_cols] > df.loc[j, score_cols]).mean()
            rates.append(wins_ij)
        pairwise_sd.append(np.std(rates, ddof=0))
    df["win_sd"] = pairwise_sd

    # macro-average & SD of raw scores
    df["macro_avg"] = df[score_cols].mean(axis=1)
    df["sd"]        = df[score_cols].std(axis=1)

    leaderboard = (df[["Model", "win_rate", "win_sd", "macro_avg", "sd"]]
                   .sort_values("win_rate", ascending=False)
                   .reset_index(drop=True))

    print("\n=== Benchmark-wide leaderboard ===")
    print(leaderboard.to_string(index=False,
                                float_format=lambda x: f"{x:0.2f}",
                                col_space=12))

    # output to LaTeX
    latex = (leaderboard.rename(columns={
                "Model":      "Model (snapshot)",
                "win_rate":   "Win-rate$\\uparrow$",
                "win_sd":     "Win SD$\\downarrow$",
                "macro_avg":  "Macro-avg",
                "sd":         "SD"})
            .to_latex(index=False, float_format="%.2f",
                      column_format="lccccc", escape=False))
    
    # determine output path
    if output_path:
        out_path = Path(output_path)
    else:
        out_path = Path(csv_path).with_suffix(".tex")
    
    out_path.write_text(latex)
    print(f"\nLaTeX table written to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark leaderboard from CSV data")
    parser.add_argument("--input", help="Input CSV file path")
    parser.add_argument("--output", help="Output LaTeX file path (default: input file with .tex extension)")
    
    args = parser.parse_args()
    main(args.input, args.output)