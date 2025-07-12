import pandas as pd
import os
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import argparse

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
            
            print(f"\n📊 File: {csv_file}")
            print(f"   Models: {len(df)}")
            print(f"   Datasets: {len(numeric_cols)} ({', '.join(numeric_cols)})")
            
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
                
                # Store in dataset_rankings
                dataset_rankings[dataset][csv_file] = file_rankings
                
                # Print individual dataset rankings
                print(f"\n   🏆 Rankings for {dataset}:")
                for model, data in sorted(file_rankings.items(), key=lambda x: x[1]['rank']):
                    print(f"      {data['rank']:2d}. {model:<40} ({data['score']:.3f})")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    # Dataset-specific cross-file analysis
    print("\n" + "="*80)
    print("🔍 DATASET-SPECIFIC RANKING ANALYSIS")
    print("="*80)
    
    all_models = sorted(list(all_models))
    
    for dataset_name, file_data in dataset_rankings.items():
        if len(file_data) < 2:
            print(f"\n⚠️  Dataset '{dataset_name}' only found in 1 file - skipping comparison")
            continue
            
        print(f"\n📊 DATASET: {dataset_name}")
        print("="*60)
        
        files_with_dataset = list(file_data.keys())
        print(f"Found in {len(files_with_dataset)} files: {', '.join(files_with_dataset)}")
        
        # Get models that appear in ALL files for this dataset
        common_models = set(all_models)
        for file_name, rankings in file_data.items():
            common_models = common_models.intersection(set(rankings.keys()))
        
        # Get models that appear in ANY files for this dataset
        any_models = set()
        for file_name, rankings in file_data.items():
            any_models = any_models.union(set(rankings.keys()))
        
        print(f"Models in all files: {len(common_models)}")
        print(f"Models in any file: {len(any_models)}")
        
        if len(common_models) == 0:
            print("⚠️  No common models across all files for this dataset")
            continue
        
        # Show ranking comparison table
        print(f"\n📋 RANKING COMPARISON TABLE:")
        print("-" * 100)
        
        # Create comparison table
        header = f"{'Model':<40}"
        for file_name in files_with_dataset:
            header += f" {file_name:<15}"
        header += f" {'Rank_Range':<12} {'Consistency':<12}"
        print(header)
        print("-" * 100)
        
        model_stats = {}
        
        for model in sorted(common_models):
            ranks = []
            scores = []
            row = f"{model:<40}"
            
            for file_name in files_with_dataset:
                if model in file_data[file_name]:
                    rank = file_data[file_name][model]['rank']
                    score = file_data[file_name][model]['score']
                    ranks.append(rank)
                    scores.append(score)
                    row += f" {rank:2d}({score:.3f})"
                else:
                    row += f" {'N/A':<15}"
            
            # Calculate consistency metrics
            if len(ranks) >= 2:
                rank_range = max(ranks) - min(ranks)
                rank_std = np.std(ranks)
                consistency = "High" if rank_range <= 1 else "Med" if rank_range <= 3 else "Low"
                
                model_stats[model] = {
                    'ranks': ranks,
                    'scores': scores,
                    'range': rank_range,
                    'std': rank_std,
                    'avg_rank': np.mean(ranks)
                }
                
                row += f" {rank_range:<12} {consistency:<12}"
            
            print(row)
        
        # Pairwise file comparisons for this dataset
        print(f"\n🔗 PAIRWISE FILE COMPARISONS for {dataset_name}:")
        print("-" * 80)
        
        correlations = []
        total_swaps = 0
        total_weighted_swaps = 0.0
        total_comparisons = 0
        
        # Track first and last place consistency
        first_place_models = []
        last_place_models = []
        
        for file_name in files_with_dataset:
            file_rankings = file_data[file_name]
            if file_rankings:
                # Find first place (rank 1)
                first_place = [model for model, data in file_rankings.items() if data['rank'] == 1]
                if first_place:
                    first_place_models.extend(first_place)
                
                # Find last place (highest rank number)
                max_rank = max(data['rank'] for data in file_rankings.values())
                last_place = [model for model, data in file_rankings.items() if data['rank'] == max_rank]
                if last_place:
                    last_place_models.extend(last_place)
        
        for i, file1 in enumerate(files_with_dataset):
            for j, file2 in enumerate(files_with_dataset[i+1:], i+1):
                # Get models common to both files
                models_file1 = set(file_data[file1].keys())
                models_file2 = set(file_data[file2].keys())
                common_in_pair = models_file1.intersection(models_file2)
                
                if len(common_in_pair) >= 3:
                    # Get rankings for comparison
                    ranks1 = []
                    ranks2 = []
                    
                    for model in common_in_pair:
                        ranks1.append(file_data[file1][model]['rank'])
                        ranks2.append(file_data[file2][model]['rank'])
                    
                    # Calculate correlation
                    corr, p_value = stats.spearmanr(ranks1, ranks2)
                    correlations.append(corr)
                    
                    # Count ranking swaps with penalty for distance
                    swaps = 0
                    weighted_swaps = 0.0
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
                                    
                                    # Calculate weighted penalty based on rank distance
                                    # Bigger rank changes get higher penalties
                                    distance1 = abs(rank1_m1 - rank1_m2)
                                    distance2 = abs(rank2_m1 - rank2_m2)
                                    avg_distance = (distance1 + distance2) / 2
                                    
                                    # Penalty increases quadratically with distance
                                    penalty = avg_distance ** 1.5
                                    weighted_swaps += penalty
                    
                    swaps = swaps // 2  # Each swap counted twice
                    weighted_swaps = weighted_swaps / 2  # Each weighted swap counted twice
                    total_swaps += swaps
                    total_weighted_swaps += weighted_swaps
                    total_comparisons += 1
                    
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    
                    print(f"   {file1:<25} vs {file2:<25}")
                    print(f"      Correlation: {corr:.3f} (p={p_value:.3f}){significance}")
                    print(f"      Rank swaps:  {swaps} (weighted: {weighted_swaps:.1f})")
                    print(f"      Common models: {len(common_in_pair)}")
                    print()
        
        # Dataset summary
        if correlations:
            avg_corr = np.mean(correlations)
            avg_swaps = total_swaps / total_comparisons if total_comparisons > 0 else 0
            avg_weighted_swaps = total_weighted_swaps / total_comparisons if total_comparisons > 0 else 0
            
            print(f"📊 SUMMARY for {dataset_name}:")
            print(f"   Average correlation: {avg_corr:.3f}")
            print(f"   Average swaps: {avg_swaps:.1f} (weighted: {avg_weighted_swaps:.1f})")
            
            # First and last place consistency
            if first_place_models:
                first_place_counts = {}
                for model in first_place_models:
                    first_place_counts[model] = first_place_counts.get(model, 0) + 1
                
                print(f"   First place consistency:")
                for model, count in sorted(first_place_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(files_with_dataset)) * 100
                    print(f"      {model}: {count}/{len(files_with_dataset)} times ({percentage:.1f}%)")
            
            if last_place_models:
                last_place_counts = {}
                for model in last_place_models:
                    last_place_counts[model] = last_place_counts.get(model, 0) + 1
                
                print(f"   Last place consistency:")
                for model, count in sorted(last_place_counts.items(), key=lambda x: x[1], reverse=True):
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
                
                print(f"   Most consistent model: {most_consistent[0]} (range: {most_consistent[1]['range']})")
                print(f"   Least consistent model: {least_consistent[0]} (range: {least_consistent[1]['range']})")
        
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
        all_weighted_swaps = []
        dataset_summary = []
        
        # Collect all correlations and weighted swaps across datasets
        for dataset_name in datasets_analyzed:
            file_data = dataset_rankings[dataset_name]
            files_with_dataset = list(file_data.keys())
            
            if len(files_with_dataset) >= 2:
                dataset_correlations = []
                dataset_weighted_swaps = []
                
                for i, file1 in enumerate(files_with_dataset):
                    for j, file2 in enumerate(files_with_dataset[i+1:], i+1):
                        models_file1 = set(file_data[file1].keys())
                        models_file2 = set(file_data[file2].keys())
                        common_in_pair = models_file1.intersection(models_file2)
                        
                        if len(common_in_pair) >= 3:
                            # Calculate correlation
                            ranks1 = [file_data[file1][model]['rank'] for model in common_in_pair]
                            ranks2 = [file_data[file2][model]['rank'] for model in common_in_pair]
                            corr, p_value = stats.spearmanr(ranks1, ranks2)
                            dataset_correlations.append(corr)
                            all_correlations.append(corr)
                            
                            # Calculate weighted swaps
                            weighted_swaps = 0.0
                            for m1 in common_in_pair:
                                for m2 in common_in_pair:
                                    if m1 != m2:
                                        rank1_m1 = file_data[file1][m1]['rank']
                                        rank1_m2 = file_data[file1][m2]['rank']
                                        rank2_m1 = file_data[file2][m1]['rank']
                                        rank2_m2 = file_data[file2][m2]['rank']
                                        
                                        if ((rank1_m1 < rank1_m2 and rank2_m1 > rank2_m2) or
                                            (rank1_m1 > rank1_m2 and rank2_m1 < rank2_m2)):
                                            distance1 = abs(rank1_m1 - rank1_m2)
                                            distance2 = abs(rank2_m1 - rank2_m2)
                                            avg_distance = (distance1 + distance2) / 2
                                            penalty = avg_distance ** 1.5
                                            weighted_swaps += penalty
                            
                            weighted_swaps = weighted_swaps / 2
                            dataset_weighted_swaps.append(weighted_swaps)
                            all_weighted_swaps.append(weighted_swaps)
                
                if dataset_correlations:
                    avg_corr = np.mean(dataset_correlations)
                    avg_weighted_swaps = np.mean(dataset_weighted_swaps)
                    dataset_summary.append({
                        'dataset': dataset_name,
                        'avg_correlation': avg_corr,
                        'avg_weighted_swaps': avg_weighted_swaps,
                        'num_comparisons': len(dataset_correlations)
                    })
        
        if all_correlations:
            overall_avg_corr = np.mean(all_correlations)
            overall_std_corr = np.std(all_correlations)
            overall_avg_weighted_swaps = np.mean(all_weighted_swaps)
            overall_std_weighted_swaps = np.std(all_weighted_swaps)
            
            print(f"🔍 OVERALL STATISTICS:")
            print(f"   Total pairwise comparisons: {len(all_correlations)}")
            print(f"   Average correlation: {overall_avg_corr:.3f} ± {overall_std_corr:.3f}")
            print(f"   Average weighted swaps: {overall_avg_weighted_swaps:.1f} ± {overall_std_weighted_swaps:.1f}")
            
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
            
            # Dataset-specific breakdown
            print(f"\n📊 DATASET-SPECIFIC BREAKDOWN:")
            for ds in sorted(dataset_summary, key=lambda x: x['avg_correlation'], reverse=True):
                print(f"   {ds['dataset']:<20}: r={ds['avg_correlation']:.3f}, weighted_swaps={ds['avg_weighted_swaps']:.1f} ({ds['num_comparisons']} comparisons)")
        
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