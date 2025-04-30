#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Composite (accuracy + completeness + clarity)/3 score.
Stats reported per‑dataset *and* an overall pooled row (“ALL”):

    ICC3k‑z   • Spearman ρ   • Weighted κ   • Human ICC3k‑z baseline
"""

import sys, numpy as np, pandas as pd, pingouin as pg
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from typing import Optional, Tuple

# ---------- helpers ---------------------------------------------------------
def zscore_within(df_long: pd.DataFrame) -> pd.DataFrame:
    return df_long.assign(
        score_z=lambda x: (x['score'] -
                           x.groupby('rater')['score'].transform('mean'))
                          / x.groupby('rater')['score'].transform('std'))

def to_bins(s: pd.Series) -> pd.Series:
    return np.select([s <= 2, s == 3, s >= 4], [0, 1, 2])

def most_overlap_pair(dsub: pd.DataFrame) -> Optional[Tuple[str, str]]:
    counts = (dsub.groupby(['username', 'instance_id'])
                    .size()
                    .unstack(fill_value=0))
    if counts.shape[0] < 2:
        return None
    overlap = counts.dot(counts.T)
    np.fill_diagonal(overlap.values, 0)
    if overlap.values.max() == 0:
        return None
    r1, r2 = np.unravel_index(overlap.values.argmax(), overlap.shape)
    return overlap.index[r1], overlap.index[r2]

def compute_stats(tag: str, subset: pd.DataFrame, store: list) -> None:
    """Calculate the four stats on *subset* and append to store."""
    human_overall = (subset.groupby("instance_id")["overall"].mean())
    llm_overall   = (subset.groupby("instance_id")["llm_overall"].first())
    merged = pd.concat([human_overall, llm_overall], axis=1)

    # 1. ICC3k‑z
    long = (merged.reset_index()
                    .melt(id_vars="instance_id", var_name="rater",
                          value_name="score")
                    .pipe(zscore_within)
                    .rename(columns={"score_z": "score"}))
    icc_result = pg.intraclass_corr(long, 'instance_id', 'rater', 'score').query("Type=='ICC3k'").iloc[0]
    icc = icc_result['ICC']
    
    # Add confidence intervals - fix the attribute access
    ci_lower = icc_result['CI95%'][0]  # Lower bound of 95% CI
    ci_upper = icc_result['CI95%'][1]  # Upper bound of 95% CI
    
    store.append((tag, "ICC3k_z", icc))
    store.append((tag, "ICC3k_z_ci_lower", ci_lower))
    store.append((tag, "ICC3k_z_ci_upper", ci_upper))

    # 2. Spearman ρ
    spearman_result = spearmanr(merged["overall"], merged["llm_overall"])
    rho = spearman_result.correlation
    
    # Calculate Spearman CI using Fisher's z transformation
    n = len(merged)
    z_transform = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1 / np.sqrt(n - 3)
    z_lower = z_transform - 1.96 * se
    z_upper = z_transform + 1.96 * se
    rho_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
    rho_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
    
    store.append((tag, "spearman_rho", rho))
    store.append((tag, "spearman_rho_ci_lower", rho_lower))
    store.append((tag, "spearman_rho_ci_upper", rho_upper))

    # 3. Weighted κ
    human_lbl = human_overall.round().astype(int)
    llm_lbl   = llm_overall  .round().astype(int)
    kappa = cohen_kappa_score(
        human_lbl,
        llm_lbl,
        weights='quadratic',
        labels=[1,2,3,4,5]
    )
    
    # Bootstrap confidence intervals for kappa
    # Note: This is computationally intensive and may take time
    n_bootstrap = 1000
    bootstrap_kappas = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(len(human_lbl), len(human_lbl), replace=True)
        h_sample = human_lbl.iloc[idx].values
        l_sample = llm_lbl.iloc[idx].values
        
        try:
            k = cohen_kappa_score(h_sample, l_sample, weights='quadratic', labels=[1,2,3,4,5])
            bootstrap_kappas.append(k)
        except ValueError:
            # If some labels aren't present in the bootstrap sample
            continue
    
    if len(bootstrap_kappas) > 50:  # Ensure we have enough bootstrap samples
        kappa_lower = np.percentile(bootstrap_kappas, 2.5)
        kappa_upper = np.percentile(bootstrap_kappas, 97.5)
    else:
        kappa_lower = kappa_upper = np.nan
    
    store.append((tag, "weighted_kappa", kappa))
    store.append((tag, "weighted_kappa_ci_lower", kappa_lower))
    store.append((tag, "weighted_kappa_ci_upper", kappa_upper))

    # 4. Human↔Human ICC3k‑z baseline
    pair = most_overlap_pair(subset)
    if pair:
        hlong = (subset[subset.username.isin(pair)]
                    .assign(score=subset.loc[
                            subset.username.isin(pair),
                            ["accuracy","completeness","clarity"]].mean(axis=1))
                    [["username","instance_id","score"]]
                    .rename(columns={"username":"rater"})
                    .pipe(zscore_within)
                    .rename(columns={"score_z":"score"}))
        try:
            h_icc_result = pg.intraclass_corr(hlong,'instance_id','rater','score').query("Type=='ICC3k'").iloc[0]
            h_icc = h_icc_result['ICC']
            h_icc_lower = h_icc_result['CI95%'][0]
            h_icc_upper = h_icc_result['CI95%'][1]
        except ValueError:
            h_icc = h_icc_lower = h_icc_upper = np.nan
    else:
        h_icc = h_icc_lower = h_icc_upper = np.nan
    
    store.append((tag, "human_ICC3k_z", h_icc))
    store.append((tag, "human_ICC3k_z_ci_lower", h_icc_lower))
    store.append((tag, "human_ICC3k_z_ci_upper", h_icc_upper))
    
df = pd.read_csv('output.csv')

# composite columns
df["overall"]     = df[["accuracy","completeness","clarity"]].mean(axis=1)
df["llm_overall"] = df[["llm_accuracy","llm_completeness","llm_clarity"]].mean(axis=1)

rows = []

# per‑dataset stats
for name, dsub in df.groupby("dataset_name"):
    compute_stats(name, dsub, rows)

subset_names = ["aci_bench", "medi_qa"]          # ⇦ change list as needed
subset_df    = df[df["dataset_name"].isin(subset_names)]
compute_stats("SUBSET("+",".join(subset_names)+")", subset_df, rows)

# ---------------------------------------------------
#  pooled 'ALL' row (unchanged)
# ---------------------------------------------------
compute_stats("ALL", df, rows)

# table
out = (pd.DataFrame(rows, columns=["dataset","stat","value"])
         .set_index(["dataset","stat"]).unstack("stat").round(3))

# Format with confidence intervals
formatted_out = pd.DataFrame(index=out.index)

# ICC3k_z with CI
formatted_out["ICC3k_z"] = out["value"]["ICC3k_z"].map('{:.3f}'.format) + " (" + \
                        out["value"]["ICC3k_z_ci_lower"].map('{:.3f}'.format) + ", " + \
                        out["value"]["ICC3k_z_ci_upper"].map('{:.3f}'.format) + ")"

# Spearman rho with CI                        
formatted_out["spearman_rho"] = out["value"]["spearman_rho"].map('{:.3f}'.format) + " (" + \
                             out["value"]["spearman_rho_ci_lower"].map('{:.3f}'.format) + ", " + \
                             out["value"]["spearman_rho_ci_upper"].map('{:.3f}'.format) + ")"

# Weighted kappa with CI
formatted_out["weighted_kappa"] = out["value"]["weighted_kappa"].map('{:.3f}'.format) + " (" + \
                               out["value"]["weighted_kappa_ci_lower"].map('{:.3f}'.format) + ", " + \
                               out["value"]["weighted_kappa_ci_upper"].map('{:.3f}'.format) + ")"

# Human ICC with CI
formatted_out["human_ICC3k_z"] = out["value"]["human_ICC3k_z"].map('{:.3f}'.format) + " (" + \
                              out["value"]["human_ICC3k_z_ci_lower"].map('{:.3f}'.format) + ", " + \
                              out["value"]["human_ICC3k_z_ci_upper"].map('{:.3f}'.format) + ")"

pd.set_option("display.width", 100)
print("\nComposite‑score agreement statistics (per dataset + pooled):\n")
print(formatted_out)
