import pandas as pd
import os
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import argparse

MODEL_NUM = 9
DATASET_NUM = 14
NUM_CSV_FILES = 7

def analyze_leaderboard_rankings(directory_path):
    """
    Analyze CSV leaderboard files to compare model rankings within the same dataset across different files.
    
    Args:
        directory_path (str): Path to directory containing CSV files
    """
    
    # Store rankings organized by dataset name
    dataset_rankings = defaultdict(dict)  # {dataset_name: {file_name: {model: rank}}}
    all_models = set()
    
    # Get all CSV files in directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    # Assert expected number of CSV files
    assert len(csv_files) == NUM_CSV_FILES, f"Expected {NUM_CSV_FILES} CSV files, found {len(csv_files)}"
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    print("="*80)
    
    # Process each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Clean column names (remove whitespace)
            df.columns = df.columns.str.strip()
            
            # Get model column (assumed to be first column)
            model_col = df.columns[0]
            
            # Get all numeric columns (datasets)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                print(f"Warning: No numeric columns found in {csv_file}")
                continue

            # Assert expected dimensions
            assert len(df) == MODEL_NUM, f"Expected {MODEL_NUM} models in {csv_file}, found {len(df)}"
            assert len(numeric_cols) == DATASET_NUM, f"Expected {DATASET_NUM} datasets in {csv_file}, found {len(numeric_cols)}"
            
            # Check for duplicate models
            assert len(df[model_col].unique()) == len(df), f"Duplicate models found in {csv_file}"
            
            # Check for missing values in numeric columns
            for col in numeric_cols:
                assert not df[col].isna().any(), f"Missing values found in column {col} of {csv_file}"
            
            # Process each dataset in this file
            for dataset in numeric_cols:
                # Sort by score (descending - higher is better)
                sorted_df = df.sort_values(by=dataset, ascending=False)
                
                # Create ranking dictionary for this dataset in this file
                file_rankings = {}
                for idx, row in sorted_df.iterrows():
                    model = row[model_col]
                    score = row[dataset]
                    rank = sorted_df.index.get_loc(idx) + 1
                    file_rankings[model] = {'rank': rank, 'score': score}
                    all_models.add(model)
                
                # Assert all models have unique ranks
                ranks = [data['rank'] for data in file_rankings.values()]
                assert len(set(ranks)) == len(ranks), f"Duplicate ranks found in {dataset} of {csv_file}"
                assert set(ranks) == set(range(1, len(ranks) + 1)), f"Missing ranks in {dataset} of {csv_file}"
                
                # Store in dataset_rankings
                dataset_rankings[dataset][csv_file] = file_rankings
                
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    # Assert all models appear in all files
    assert len(all_models) == MODEL_NUM, f"Expected {MODEL_NUM} unique models across all files, found {len(all_models)}"
    
    # Dataset-specific cross-file analysis
    print("\n" + "="*80)
    print("🔍 DATASET-SPECIFIC RANKING ANALYSIS")
    print("="*80)
    
    all_models = sorted(list(all_models))
    
    for dataset_name, file_data in dataset_rankings.items():
        files_with_dataset = list(file_data.keys())
        assert len(files_with_dataset) == NUM_CSV_FILES, f"Dataset {dataset_name} missing from some files"
        
        # Get models that appear in ALL files for this dataset
        common_models = set(all_models)
        for file_name, rankings in file_data.items():
            common_models = common_models.intersection(set(rankings.keys()))
        
        # Get models that appear in ANY files for this dataset
        any_models = set()
        for file_name, rankings in file_data.items():
            any_models = any_models.union(set(rankings.keys()))
        
        # Assert all models appear in all files for each dataset
        assert len(common_models) == len(any_models), f"Not all models appear in all files for dataset {dataset_name}"
        assert len(common_models) == MODEL_NUM, f"Expected {MODEL_NUM} models in dataset {dataset_name}, found {len(common_models)}"
        
        model_stats = {}
        
        for model in sorted(common_models):
            ranks = []
            scores = []
            
            for file_name in files_with_dataset:
                if model in file_data[file_name]:
                    rank = file_data[file_name][model]['rank']
                    score = file_data[file_name][model]['score']
                    ranks.append(rank)
                    scores.append(score)
            
            # Calculate consistency metrics
            if len(ranks) >= 2:
                rank_range = max(ranks) - min(ranks)
                rank_std = np.std(ranks)
                
                model_stats[model] = {
                    'ranks': ranks,
                    'scores': scores,
                    'range': rank_range,
                    'std': rank_std,
                    'avg_rank': np.mean(ranks)
                }
        
        # Pairwise file comparisons for this dataset
        correlations = []
        total_swaps = 0
        total_comparisons = 0
        
        # Track top 2 and bottom 2 place consistency
        top_2_models = []
        bottom_2_models = []
        
        for file_name in files_with_dataset:
            file_rankings = file_data[file_name]
            if file_rankings:
                # Find top 2 (ranks 1 and 2)
                top_2 = [model for model, data in file_rankings.items() if data['rank'] in [1, 2]]
                top_2_models.append(set(top_2))
                
                # Find bottom 2 (highest 2 rank numbers)
                all_ranks = [data['rank'] for data in file_rankings.values()]
                max_rank = max(all_ranks)
                second_max_rank = sorted(all_ranks, reverse=True)[1]
                bottom_2 = [model for model, data in file_rankings.items() if data['rank'] in [max_rank, second_max_rank]]
                bottom_2_models.append(set(bottom_2))
        
        for i, file1 in enumerate(files_with_dataset):
            for j, file2 in enumerate(files_with_dataset[i+1:], i+1):
                # Get models common to both files
                models_file1 = set(file_data[file1].keys())
                models_file2 = set(file_data[file2].keys())
                common_in_pair = models_file1.intersection(models_file2)
                
                # Assert all models are in both files
                assert len(common_in_pair) == MODEL_NUM, f"Not all models found in both {file1} and {file2} for dataset {dataset_name}"
                
                # Get rankings for comparison
                ranks1 = []
                ranks2 = []
                
                for model in common_in_pair:
                    ranks1.append(file_data[file1][model]['rank'])
                    ranks2.append(file_data[file2][model]['rank'])
                
                # Calculate correlation
                corr, p_value = stats.spearmanr(ranks1, ranks2)
                correlations.append(corr)
                
                # Count ranking swaps (simple count, no weighting)
                swaps = 0
                for m1 in common_in_pair:
                    for m2 in common_in_pair:
                        if m1 != m2:
                            rank1_m1 = file_data[file1][m1]['rank']
                            rank1_m2 = file_data[file1][m2]['rank']
                            rank2_m1 = file_data[file2][m1]['rank']
                            rank2_m2 = file_data[file2][m2]['rank']
                            
                            # Check if relative order switched
                            if ((rank1_m1 < rank1_m2 and rank2_m1 > rank2_m2) or
                                (rank1_m1 > rank1_m2 and rank2_m1 < rank2_m2)):
                                swaps += 1
                
                swaps = swaps // 2  # Each swap counted twice
                total_swaps += swaps
                total_comparisons += 1
        
        # Dataset summary
        if correlations:
            avg_corr = np.mean(correlations)
            avg_swaps = total_swaps / total_comparisons if total_comparisons > 0 else 0
            
            print(f"📊 SUMMARY for {dataset_name}:")
            print(f"   Average correlation: {avg_corr:.3f}")
            print(f"   Average swaps: {avg_swaps:.1f}")
            
            # Top 2 consistency analysis
            if top_2_models:
                # Calculate pairwise overlap for top 2
                top_2_overlaps = []
                for i in range(len(top_2_models)):
                    for j in range(i+1, len(top_2_models)):
                        overlap = len(top_2_models[i].intersection(top_2_models[j]))
                        top_2_overlaps.append(overlap)
                
                avg_top_2_overlap = np.mean(top_2_overlaps) if top_2_overlaps else 0
                print(f"   Top 2 consistency (avg overlap): {avg_top_2_overlap:.1f}/2 ({avg_top_2_overlap/2*100:.1f}%)")
                
                # Show most frequent top 2 models
                from collections import Counter
                all_top_2_models = []
                for models_set in top_2_models:
                    all_top_2_models.extend(models_set)
                
                top_2_counts = Counter(all_top_2_models)
                print(f"   Most frequent in top 2:")
                for model, count in top_2_counts.most_common(5):
                    percentage = (count / len(files_with_dataset)) * 100
                    print(f"      {model}: {count}/{len(files_with_dataset)} times ({percentage:.1f}%)")
            
            # Bottom 2 consistency analysis
            if bottom_2_models:
                # Calculate pairwise overlap for bottom 2
                bottom_2_overlaps = []
                for i in range(len(bottom_2_models)):
                    for j in range(i+1, len(bottom_2_models)):
                        overlap = len(bottom_2_models[i].intersection(bottom_2_models[j]))
                        bottom_2_overlaps.append(overlap)
                
                avg_bottom_2_overlap = np.mean(bottom_2_overlaps) if bottom_2_overlaps else 0
                print(f"   Bottom 2 consistency (avg overlap): {avg_bottom_2_overlap:.1f}/2 ({avg_bottom_2_overlap/2*100:.1f}%)")
                
                # Show most frequent bottom 2 models
                all_bottom_2_models = []
                for models_set in bottom_2_models:
                    all_bottom_2_models.extend(models_set)
                
                bottom_2_counts = Counter(all_bottom_2_models)
                print(f"   Most frequent in bottom 2:")
                for model, count in bottom_2_counts.most_common(5):
                    percentage = (count / len(files_with_dataset)) * 100
                    print(f"      {model}: {count}/{len(files_with_dataset)} times ({percentage:.1f}%)")
            
            if avg_corr > 0.8:
                consistency_level = "VERY HIGH"
            elif avg_corr > 0.6:
                consistency_level = "HIGH"
            elif avg_corr > 0.4:
                consistency_level = "MODERATE"
            else:
                consistency_level = "LOW"
            
            print(f"   Consistency: {consistency_level}")
            
            # Show most/least consistent models
            if model_stats:
                most_consistent = min(model_stats.items(), key=lambda x: x[1]['range'])
                least_consistent = max(model_stats.items(), key=lambda x: x[1]['range'])
        
        print()
    
    # Overall summary
    print("\n" + "="*80)
    print("📋 OVERALL SUMMARY")
    print("="*80)
    
    datasets_analyzed = [d for d in dataset_rankings.keys() if len(dataset_rankings[d]) >= 2]
    
    print(f"• Total files analyzed: {len(csv_files)}")
    print(f"• Total unique models: {len(all_models)}")
    print(f"• Datasets found: {len(dataset_rankings)}")
    print(f"• Datasets with multiple files: {len(datasets_analyzed)}")
    
    if datasets_analyzed:
        print(f"• Datasets analyzed: {', '.join(datasets_analyzed)}")
    
    print(f"\n🤖 All models found: {', '.join(sorted(all_models))}")
    
    # Aggregated statistical analysis
    print("\n" + "="*80)
    print("📊 AGGREGATED STATISTICAL ANALYSIS")
    print("="*80)
    
    if len(datasets_analyzed) > 0:
        all_correlations = []
        all_swaps = []
        dataset_summary = []
        all_top_2_overlaps = []
        all_bottom_2_overlaps = []
        
        # Collect all correlations and swaps across datasets
        for dataset_name in datasets_analyzed:
            file_data = dataset_rankings[dataset_name]
            files_with_dataset = list(file_data.keys())
            
            if len(files_with_dataset) >= 2:
                dataset_correlations = []
                dataset_swaps = []
                
                # Collect top 2 and bottom 2 sets for this dataset
                top_2_models = []
                bottom_2_models = []
                
                for file_name in files_with_dataset:
                    file_rankings = file_data[file_name]
                    if file_rankings:
                        # Find top 2
                        top_2 = [model for model, data in file_rankings.items() if data['rank'] in [1, 2]]
                        top_2_models.append(set(top_2))
                        
                        # Find bottom 2
                        all_ranks = [data['rank'] for data in file_rankings.values()]
                        max_rank = max(all_ranks)
                        second_max_rank = sorted(all_ranks, reverse=True)[1]
                        bottom_2 = [model for model, data in file_rankings.items() if data['rank'] in [max_rank, second_max_rank]]
                        bottom_2_models.append(set(bottom_2))
                
                # Calculate top 2 overlaps for this dataset
                for i in range(len(top_2_models)):
                    for j in range(i+1, len(top_2_models)):
                        overlap = len(top_2_models[i].intersection(top_2_models[j]))
                        all_top_2_overlaps.append(overlap)
                
                # Calculate bottom 2 overlaps for this dataset
                for i in range(len(bottom_2_models)):
                    for j in range(i+1, len(bottom_2_models)):
                        overlap = len(bottom_2_models[i].intersection(bottom_2_models[j]))
                        all_bottom_2_overlaps.append(overlap)
                
                for i, file1 in enumerate(files_with_dataset):
                    for j, file2 in enumerate(files_with_dataset[i+1:], i+1):
                        models_file1 = set(file_data[file1].keys())
                        models_file2 = set(file_data[file2].keys())
                        common_in_pair = models_file1.intersection(models_file2)
                        
                        assert len(common_in_pair) == MODEL_NUM, f"Model count mismatch in {file1} vs {file2} for {dataset_name}"
                        
                        # Calculate correlation
                        ranks1 = [file_data[file1][model]['rank'] for model in common_in_pair]
                        ranks2 = [file_data[file2][model]['rank'] for model in common_in_pair]
                        corr, p_value = stats.spearmanr(ranks1, ranks2)
                        dataset_correlations.append(corr)
                        all_correlations.append(corr)
                        
                        # Calculate swaps (simple count)
                        swaps = 0
                        for m1 in common_in_pair:
                            for m2 in common_in_pair:
                                if m1 != m2:
                                    rank1_m1 = file_data[file1][m1]['rank']
                                    rank1_m2 = file_data[file1][m2]['rank']
                                    rank2_m1 = file_data[file2][m1]['rank']
                                    rank2_m2 = file_data[file2][m2]['rank']
                                    
                                    if ((rank1_m1 < rank1_m2 and rank2_m1 > rank2_m2) or
                                        (rank1_m1 > rank1_m2 and rank2_m1 < rank2_m2)):
                                        swaps += 1
                        
                        swaps = swaps // 2
                        dataset_swaps.append(swaps)
                        all_swaps.append(swaps)
                
                if dataset_correlations:
                    avg_corr = np.mean(dataset_correlations)
                    avg_swaps = np.mean(dataset_swaps)
                    dataset_summary.append({
                        'dataset': dataset_name,
                        'avg_correlation': avg_corr,
                        'avg_swaps': avg_swaps,
                        'num_comparisons': len(dataset_correlations)
                    })
        
        if all_correlations:
            overall_avg_corr = np.mean(all_correlations)
            overall_std_corr = np.std(all_correlations)
            overall_avg_swaps = np.mean(all_swaps)
            overall_std_swaps = np.std(all_swaps)
            
            print(f"🔍 OVERALL STATISTICS:")
            print(f"   Total pairwise comparisons: {len(all_correlations)}")
            print(f"   Average correlation: {overall_avg_corr:.3f} ± {overall_std_corr:.3f}")
            print(f"   Average swaps: {overall_avg_swaps:.1f} ± {overall_std_swaps:.1f}")
            
            # Top 2 and Bottom 2 consistency statistics
            if all_top_2_overlaps:
                overall_avg_top_2_overlap = np.mean(all_top_2_overlaps)
                overall_std_top_2_overlap = np.std(all_top_2_overlaps)
                print(f"   Average top 2 overlap: {overall_avg_top_2_overlap:.1f}/2 ± {overall_std_top_2_overlap:.1f} ({overall_avg_top_2_overlap/2*100:.1f}%)")
            
            if all_bottom_2_overlaps:
                overall_avg_bottom_2_overlap = np.mean(all_bottom_2_overlaps)
                overall_std_bottom_2_overlap = np.std(all_bottom_2_overlaps)
                print(f"   Average bottom 2 overlap: {overall_avg_bottom_2_overlap:.1f}/2 ± {overall_std_bottom_2_overlap:.1f} ({overall_avg_bottom_2_overlap/2*100:.1f}%)")
            
            # Statistical significance testing
            print(f"\n📈 STATISTICAL SIGNIFICANCE ANALYSIS:")
            
            # Test if correlations are significantly different from 0 (no correlation)
            t_stat, p_value = stats.ttest_1samp(all_correlations, 0)
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"   H0: Correlations = 0 (no similarity)")
            print(f"   t-statistic: {t_stat:.3f}, p-value: {p_value:.6f}{significance}")
            
            # Test if correlations are significantly different from 1 (perfect correlation)
            t_stat_perfect, p_value_perfect = stats.ttest_1samp(all_correlations, 1)
            significance_perfect = "***" if p_value_perfect < 0.001 else "**" if p_value_perfect < 0.01 else "*" if p_value_perfect < 0.05 else ""
            print(f"   H0: Correlations = 1 (perfect similarity)")
            print(f"   t-statistic: {t_stat_perfect:.3f}, p-value: {p_value_perfect:.6f}{significance_perfect}")
            
            # Test if correlations are significantly different from 0.7 (high similarity threshold)
            t_stat_high, p_value_high = stats.ttest_1samp(all_correlations, 0.7)
            significance_high = "***" if p_value_high < 0.001 else "**" if p_value_high < 0.01 else "*" if p_value_high < 0.05 else ""
            print(f"   H0: Correlations = 0.7 (high similarity threshold)")
            print(f"   t-statistic: {t_stat_high:.3f}, p-value: {p_value_high:.6f}{significance_high}")
            
            # Test if swaps are significantly different from 0 (no swaps)
            t_stat_swaps, p_value_swaps = stats.ttest_1samp(all_swaps, 0)
            significance_swaps = "***" if p_value_swaps < 0.001 else "**" if p_value_swaps < 0.01 else "*" if p_value_swaps < 0.05 else ""
            print(f"   H0: Swaps = 0 (no ranking changes)")
            print(f"   t-statistic: {t_stat_swaps:.3f}, p-value: {p_value_swaps:.6f}{significance_swaps}")
            
            # Test top 2 overlap significance
            if all_top_2_overlaps:
                t_stat_top2, p_value_top2 = stats.ttest_1samp(all_top_2_overlaps, 1)  # Test against random overlap of 1
                significance_top2 = "***" if p_value_top2 < 0.001 else "**" if p_value_top2 < 0.01 else "*" if p_value_top2 < 0.05 else ""
                print(f"   H0: Top 2 overlap = 1 (random overlap)")
                print(f"   t-statistic: {t_stat_top2:.3f}, p-value: {p_value_top2:.6f}{significance_top2}")
            
            # Test bottom 2 overlap significance
            if all_bottom_2_overlaps:
                t_stat_bottom2, p_value_bottom2 = stats.ttest_1samp(all_bottom_2_overlaps, 1)  # Test against random overlap of 1
                significance_bottom2 = "***" if p_value_bottom2 < 0.001 else "**" if p_value_bottom2 < 0.01 else "*" if p_value_bottom2 < 0.05 else ""
                print(f"   H0: Bottom 2 overlap = 1 (random overlap)")
                print(f"   t-statistic: {t_stat_bottom2:.3f}, p-value: {p_value_bottom2:.6f}{significance_bottom2}")
            
            # Overall interpretation
            print(f"\n🎯 INTERPRETATION:")
            
            if overall_avg_corr > 0.8:
                consistency_verdict = "VERY HIGH CONSISTENCY"
                interpretation = "Leaderboards are highly similar - model rankings are very stable"
            elif overall_avg_corr > 0.6:
                consistency_verdict = "HIGH CONSISTENCY"
                interpretation = "Leaderboards are quite similar - model rankings are generally stable"
            elif overall_avg_corr > 0.4:
                consistency_verdict = "MODERATE CONSISTENCY"
                interpretation = "Leaderboards show moderate similarity - some ranking variation exists"
            elif overall_avg_corr > 0.2:
                consistency_verdict = "LOW CONSISTENCY"
                interpretation = "Leaderboards differ significantly - ranking variation is substantial"
            else:
                consistency_verdict = "VERY LOW CONSISTENCY"
                interpretation = "Leaderboards are quite different - rankings vary considerably"
            
            print(f"   Overall verdict: {consistency_verdict}")
            print(f"   Interpretation: {interpretation}")
            
            # Top/Bottom consistency interpretation
            if all_top_2_overlaps:
                avg_top_2_pct = overall_avg_top_2_overlap / 2 * 100
                if avg_top_2_pct > 75:
                    top_2_verdict = "VERY HIGH"
                elif avg_top_2_pct > 50:
                    top_2_verdict = "HIGH"
                elif avg_top_2_pct > 25:
                    top_2_verdict = "MODERATE"
                else:
                    top_2_verdict = "LOW"
                print(f"   Top 2 consistency: {top_2_verdict} ({avg_top_2_pct:.1f}% average overlap)")
            
            if all_bottom_2_overlaps:
                avg_bottom_2_pct = overall_avg_bottom_2_overlap / 2 * 100
                if avg_bottom_2_pct > 75:
                    bottom_2_verdict = "VERY HIGH"
                elif avg_bottom_2_pct > 50:
                    bottom_2_verdict = "HIGH"
                elif avg_bottom_2_pct > 25:
                    bottom_2_verdict = "MODERATE"
                else:
                    bottom_2_verdict = "LOW"
                print(f"   Bottom 2 consistency: {bottom_2_verdict} ({avg_bottom_2_pct:.1f}% average overlap)")
            
            # Statistical significance interpretation
            if p_value < 0.05:
                print(f"   Statistical significance: Rankings are significantly correlated (p < 0.05)")
            else:
                print(f"   Statistical significance: Rankings are NOT significantly correlated (p >= 0.05)")
            
            # Confidence interval for correlation
            sem = overall_std_corr / np.sqrt(len(all_correlations))
            ci_lower = overall_avg_corr - 1.96 * sem
            ci_upper = overall_avg_corr + 1.96 * sem
            print(f"   95% Confidence Interval for correlation: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        else:
            print("⚠️  No statistical analysis possible - insufficient data for correlations")
    
    else:
        print("⚠️  No datasets found with multiple files for analysis")

def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Analyze CSV leaderboard rankings by dataset')
    parser.add_argument('directory', help='Path to directory containing CSV files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    
    analyze_leaderboard_rankings(args.directory)

if __name__ == "__main__":
    # If no command line arguments, use current directory
    directory = "/share/pi/nigam/users/aunell/medhelm/judge_ablations"
    analyze_leaderboard_rankings(directory)