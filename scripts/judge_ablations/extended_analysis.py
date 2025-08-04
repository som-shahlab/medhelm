import pandas as pd
import os
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict
import argparse
from itertools import combinations

MODEL_NUM = 9
DATASET_NUM = 14
NUM_CSV_FILES = 7

def calculate_effect_size(scores1, scores2):
    """Calculate Cohen's d effect size between two score distributions."""
    pooled_std = np.sqrt(((len(scores1) - 1) * np.var(scores1, ddof=1) + 
                         (len(scores2) - 1) * np.var(scores2, ddof=1)) / 
                        (len(scores1) + len(scores2) - 2))
    if pooled_std == 0:
        return 0
    return abs(np.mean(scores1) - np.mean(scores2)) / pooled_std

def analyze_score_gaps(rankings_data):
    """Analyze the actual score gaps between adjacent rankings."""
    gaps = []
    for model_data in rankings_data.values():
        if 'score' in model_data:
            gaps.append(model_data['score'])
    
    if len(gaps) < 2:
        return {}
    
    gaps.sort(reverse=True)  # Sort descending
    adjacent_gaps = [gaps[i] - gaps[i+1] for i in range(len(gaps)-1)]
    
    return {
        'mean_gap': np.mean(adjacent_gaps),
        'median_gap': np.median(adjacent_gaps),
        'min_gap': min(adjacent_gaps),
        'max_gap': max(adjacent_gaps),
        'std_gap': np.std(adjacent_gaps),
        'gaps': adjacent_gaps
    }

def calculate_meaningful_disagreement(file_data, min_meaningful_diff=0.01):
    """Calculate disagreement rates considering only meaningful score differences."""
    
    files = list(file_data.keys())
    meaningful_disagreements = 0
    total_comparisons = 0
    
    for file1, file2 in combinations(files, 2):
        data1 = file_data[file1]
        data2 = file_data[file2]
        
        # Get common models
        common_models = set(data1.keys()) & set(data2.keys())
        
        for model1, model2 in combinations(common_models, 2):
            # Get scores for both models in both files
            score1_file1 = data1[model1]['score']
            score1_file2 = data2[model1]['score']
            score2_file1 = data1[model2]['score']
            score2_file2 = data2[model2]['score']
            
            # Check if the difference is meaningful in both files
            diff1 = abs(score1_file1 - score2_file1)
            diff2 = abs(score1_file2 - score2_file2)
            
            if diff1 > min_meaningful_diff and diff2 > min_meaningful_diff:
                total_comparisons += 1
                
                # Check if the relative order is consistent
                order1 = score1_file1 > score2_file1
                order2 = score1_file2 > score2_file2
                
                if order1 != order2:
                    meaningful_disagreements += 1
    
    if total_comparisons == 0:
        return 0, 0, 0
    
    return meaningful_disagreements, total_comparisons, meaningful_disagreements / total_comparisons

def confidence_based_ranking(rankings_data, confidence_threshold=0.02):
    """Create confidence-based rankings where close scores are treated as ties."""
    
    # Sort by score
    sorted_items = sorted(rankings_data.items(), key=lambda x: x[1]['score'], reverse=True)
    
    confident_ranks = {}
    current_rank = 1
    
    for i, (model, data) in enumerate(sorted_items):
        if i == 0:
            confident_ranks[model] = current_rank
        else:
            prev_score = sorted_items[i-1][1]['score']
            current_score = data['score']
            
            # If the difference is smaller than threshold, it's a tie
            if abs(prev_score - current_score) < confidence_threshold:
                confident_ranks[model] = confident_ranks[sorted_items[i-1][0]]
            else:
                current_rank = i + 1
                confident_ranks[model] = current_rank
    
    return confident_ranks

def bootstrap_consistency_analysis(dataset_rankings, n_bootstrap=1000):
    """Bootstrap analysis to assess consistency stability."""
    
    files = list(dataset_rankings.keys())
    if len(files) < 3:
        return {}
    
    bootstrap_correlations = []
    
    for _ in range(n_bootstrap):
        # Sample subset of files
        sample_files = np.random.choice(files, size=max(2, len(files)//2), replace=False)
        
        # Calculate correlation for this subset
        subset_correlations = []
        for file1, file2 in combinations(sample_files, 2):
            data1 = dataset_rankings[file1]
            data2 = dataset_rankings[file2]
            
            common_models = set(data1.keys()) & set(data2.keys())
            if len(common_models) < 2:
                continue
                
            ranks1 = [data1[model]['rank'] for model in common_models]
            ranks2 = [data2[model]['rank'] for model in common_models]
            
            corr, _ = stats.spearmanr(ranks1, ranks2)
            if not np.isnan(corr):
                subset_correlations.append(corr)
        
        if subset_correlations:
            bootstrap_correlations.append(np.mean(subset_correlations))
    
    if bootstrap_correlations:
        return {
            'mean': np.mean(bootstrap_correlations),
            'std': np.std(bootstrap_correlations),
            'ci_lower': np.percentile(bootstrap_correlations, 2.5),
            'ci_upper': np.percentile(bootstrap_correlations, 97.5)
        }
    
    return {}

def analyze_systematic_disagreements(dataset_rankings):
    """Identify if disagreements are systematic or random."""
    
    files = list(dataset_rankings.keys())
    if len(files) < 2:
        return {}
    
    # Track which models consistently disagree
    model_disagreement_patterns = defaultdict(list)
    
    for file1, file2 in combinations(files, 2):
        data1 = dataset_rankings[file1]
        data2 = dataset_rankings[file2]
        
        common_models = set(data1.keys()) & set(data2.keys())
        
        for model in common_models:
            rank1 = data1[model]['rank']
            rank2 = data2[model]['rank']
            disagreement = abs(rank1 - rank2)
            
            model_disagreement_patterns[model].append(disagreement)
    
    # Calculate consistency metrics per model
    model_consistency = {}
    for model, disagreements in model_disagreement_patterns.items():
        if disagreements:
            model_consistency[model] = {
                'mean_disagreement': np.mean(disagreements),
                'std_disagreement': np.std(disagreements),
                'max_disagreement': max(disagreements),
                'is_systematic': np.std(disagreements) < np.mean(disagreements)  # Low variance suggests systematic bias
            }
    
    return model_consistency

def enhanced_leaderboard_analysis(directory_path):
    """Enhanced analysis with focus on meaningful differences and confidence."""
    
    # Store rankings organized by dataset name (reusing structure from original)
    dataset_rankings = defaultdict(dict)
    all_models = set()
    
    # Get all CSV files in directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return
    
    assert len(csv_files) == NUM_CSV_FILES, f"Expected {NUM_CSV_FILES} CSV files, found {len(csv_files)}"
    
    print(f"Found {len(csv_files)} CSV files to analyze")
    print("="*80)
    
    # Process each CSV file (similar to original but storing more info)
    for csv_file in csv_files:
        file_path = os.path.join(directory_path, csv_file)
        
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()
            
            model_col = df.columns[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                continue

            assert len(df) == MODEL_NUM, f"Expected {MODEL_NUM} models in {csv_file}, found {len(df)}"
            assert len(numeric_cols) == DATASET_NUM, f"Expected {DATASET_NUM} datasets in {csv_file}, found {len(numeric_cols)}"
            
            # Process each dataset
            for dataset in numeric_cols:
                sorted_df = df.sort_values(by=dataset, ascending=False)
                
                file_rankings = {}
                for idx, row in sorted_df.iterrows():
                    model = row[model_col]
                    score = row[dataset]
                    rank = sorted_df.index.get_loc(idx) + 1
                    file_rankings[model] = {'rank': rank, 'score': score}
                    all_models.add(model)
                
                dataset_rankings[dataset][csv_file] = file_rankings
                
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    # Enhanced dataset-specific analysis
    print("\n" + "="*80)
    print("🔍 ENHANCED CONSISTENCY ANALYSIS")
    print("="*80)
    
    all_models = sorted(list(all_models))
    dataset_summaries = []
    
    for dataset_name, file_data in dataset_rankings.items():
        files_with_dataset = list(file_data.keys())
        
        if len(files_with_dataset) < 2:
            continue
        
        print(f"\n📊 DATASET: {dataset_name}")
        print("="*60)
        
        # 1. Score Gap Analysis
        print("1️⃣ SCORE GAP ANALYSIS:")
        gap_stats = []
        for file_name, rankings in file_data.items():
            gaps = analyze_score_gaps(rankings)
            if gaps:
                gap_stats.append(gaps)
                print(f"   {file_name}: Mean gap = {gaps['mean_gap']:.4f}, "
                      f"Min gap = {gaps['min_gap']:.4f}, Max gap = {gaps['max_gap']:.4f}")
        
        if gap_stats:
            overall_mean_gap = np.mean([g['mean_gap'] for g in gap_stats])
            overall_min_gap = np.mean([g['min_gap'] for g in gap_stats])
            print(f"   📈 Overall: Mean gap = {overall_mean_gap:.4f}, "
                  f"Typical min gap = {overall_min_gap:.4f}")
        
        # 2. Meaningful Disagreement Analysis
        print("\n2️⃣ MEANINGFUL DISAGREEMENT ANALYSIS:")
        for threshold in [0.001, 0.01, 0.05]:
            disagreements, total, rate = calculate_meaningful_disagreement(file_data, threshold)
            print(f"   Threshold {threshold:.3f}: {disagreements}/{total} disagreements ({rate:.1%})")
        
        # 3. Confidence-based Rankings
        print("\n3️⃣ CONFIDENCE-BASED RANKING CONSISTENCY:")
        confidence_correlations = []
        for file1, file2 in combinations(files_with_dataset, 2):
            conf_ranks1 = confidence_based_ranking(file_data[file1])
            conf_ranks2 = confidence_based_ranking(file_data[file2])
            
            common_models = set(conf_ranks1.keys()) & set(conf_ranks2.keys())
            if len(common_models) >= 2:
                ranks1 = [conf_ranks1[model] for model in common_models]
                ranks2 = [conf_ranks2[model] for model in common_models]
                
                corr, _ = stats.spearmanr(ranks1, ranks2)
                if not np.isnan(corr):
                    confidence_correlations.append(corr)
        
        if confidence_correlations:
            conf_corr = np.mean(confidence_correlations)
            print(f"   Confidence-based correlation: {conf_corr:.3f}")
        
        # 4. Bootstrap Consistency Analysis
        print("\n4️⃣ BOOTSTRAP CONSISTENCY ANALYSIS:")
        bootstrap_stats = bootstrap_consistency_analysis(file_data)
        if bootstrap_stats:
            print(f"   Bootstrap correlation: {bootstrap_stats['mean']:.3f} ± {bootstrap_stats['std']:.3f}")
            print(f"   95% CI: [{bootstrap_stats['ci_lower']:.3f}, {bootstrap_stats['ci_upper']:.3f}]")
        
        # 5. Systematic Disagreement Analysis
        print("\n5️⃣ SYSTEMATIC DISAGREEMENT ANALYSIS:")
        model_consistency = analyze_systematic_disagreements(file_data)
        systematic_models = [m for m, stats in model_consistency.items() if stats['is_systematic']]
        random_models = [m for m, stats in model_consistency.items() if not stats['is_systematic']]
        
        print(f"   Models with systematic disagreements: {len(systematic_models)}")
        print(f"   Models with random disagreements: {len(random_models)}")
        
        if systematic_models:
            print(f"   Most systematic: {systematic_models[:3]}")
        
        # 6. First/Last Place Analysis with Effect Sizes
        print("\n6️⃣ FIRST/LAST PLACE ANALYSIS:")
        first_last_consistency = {}
        
        for file_name, rankings in file_data.items():
            # Find first and last place
            sorted_models = sorted(rankings.items(), key=lambda x: x[1]['score'], reverse=True)
            first_model = sorted_models[0][0]
            last_model = sorted_models[-1][0]
            
            first_last_consistency[file_name] = {
                'first': first_model,
                'last': last_model,
                'first_score': sorted_models[0][1]['score'],
                'last_score': sorted_models[-1][1]['score'],
                'gap': sorted_models[0][1]['score'] - sorted_models[-1][1]['score']
            }
        
        # Calculate first/last place consistency
        first_models = [data['first'] for data in first_last_consistency.values()]
        last_models = [data['last'] for data in first_last_consistency.values()]
        
        from collections import Counter
        first_counts = Counter(first_models)
        last_counts = Counter(last_models)
        
        print(f"   First place consistency: {first_counts.most_common(1)[0][1]}/{len(files_with_dataset)} ({first_counts.most_common(1)[0][1]/len(files_with_dataset):.1%})")
        print(f"   Last place consistency: {last_counts.most_common(1)[0][1]}/{len(files_with_dataset)} ({last_counts.most_common(1)[0][1]/len(files_with_dataset):.1%})")
        
        # Calculate effect sizes for first vs last
        first_last_gaps = [data['gap'] for data in first_last_consistency.values()]
        print(f"   Average first-last gap: {np.mean(first_last_gaps):.4f} ± {np.std(first_last_gaps):.4f}")
        
        # 7. Overall Assessment
        print("\n7️⃣ OVERALL ASSESSMENT:")
        
        # Calculate traditional correlation
        traditional_correlations = []
        for file1, file2 in combinations(files_with_dataset, 2):
            data1 = file_data[file1]
            data2 = file_data[file2]
            
            common_models = set(data1.keys()) & set(data2.keys())
            ranks1 = [data1[model]['rank'] for model in common_models]
            ranks2 = [data2[model]['rank'] for model in common_models]
            
            corr, _ = stats.spearmanr(ranks1, ranks2)
            if not np.isnan(corr):
                traditional_correlations.append(corr)
        
        if traditional_correlations:
            trad_corr = np.mean(traditional_correlations)
            
            # Decision framework
            print(f"   Traditional correlation: {trad_corr:.3f}")
            
            if confidence_correlations:
                print(f"   Confidence-based correlation: {conf_corr:.3f}")
            
            if overall_min_gap < 0.01:
                print("   🚨 WARNING: Very small score gaps detected - many differences may not be meaningful")
            
            # Decision logic
            if trad_corr > 0.8 and (not confidence_correlations or conf_corr > 0.8):
                recommendation = "✅ JURIES ARE CONSISTENT - Current setup is sufficient"
            elif trad_corr > 0.6 and overall_min_gap > 0.01:
                recommendation = "✅ JURIES ARE REASONABLY CONSISTENT - Consider current setup sufficient"
            elif trad_corr > 0.4 and overall_min_gap < 0.01:
                recommendation = "⚠️ MIXED SIGNALS - Low correlation but small gaps suggest noise rather than real disagreement"
            elif len(systematic_models) > len(random_models):
                recommendation = "🔄 SYSTEMATIC DISAGREEMENTS DETECTED - Consider adding models that capture missing aspects"
            else:
                recommendation = "🔄 INCONSISTENT JURIES - Consider adding new models or investigating evaluation criteria"
            
            print(f"   {recommendation}")
        
        # Store summary for overall analysis
        dataset_summaries.append({
            'dataset': dataset_name,
            'traditional_correlation': trad_corr if traditional_correlations else 0,
            'confidence_correlation': conf_corr if confidence_correlations else 0,
            'mean_gap': overall_mean_gap if gap_stats else 0,
            'min_gap': overall_min_gap if gap_stats else 0,
            'systematic_models': len(systematic_models),
            'total_models': len(model_consistency)
        })
    
    # Overall recommendations
    print("\n" + "="*80)
    print("🎯 OVERALL RECOMMENDATIONS")
    print("="*80)
    
    if dataset_summaries:
        avg_trad_corr = np.mean([d['traditional_correlation'] for d in dataset_summaries])
        avg_conf_corr = np.mean([d['confidence_correlation'] for d in dataset_summaries if d['confidence_correlation'] > 0])
        avg_min_gap = np.mean([d['min_gap'] for d in dataset_summaries if d['min_gap'] > 0])
        
        print(f"📊 Overall Statistics:")
        print(f"   Average traditional correlation: {avg_trad_corr:.3f}")
        if avg_conf_corr:
            print(f"   Average confidence-based correlation: {avg_conf_corr:.3f}")
        print(f"   Average minimum score gap: {avg_min_gap:.4f}")
        
        # Final recommendation
        if avg_trad_corr > 0.7 and avg_min_gap > 0.01:
            final_rec = "✅ CURRENT JURY SETUP IS SUFFICIENT"
        elif avg_trad_corr > 0.5 and avg_min_gap < 0.01:
            final_rec = "⚠️ APPARENT INCONSISTENCY MAY BE DUE TO NOISE - CONSIDER CURRENT SETUP ADEQUATE"
        else:
            final_rec = "🔄 CONSIDER ADDING NEW JURY MODELS"
        
        print(f"\n🎯 FINAL RECOMMENDATION: {final_rec}")
        
        # Specific guidance
        print(f"\n📋 SPECIFIC GUIDANCE:")
        if avg_min_gap < 0.01:
            print("   • Very small score differences suggest many rankings are essentially ties")
            print("   • Consider using confidence-based rankings or minimum meaningful difference thresholds")
        
        if avg_trad_corr < 0.6:
            print("   • Low correlation suggests genuine disagreement between juries")
            print("   • Investigate what aspects current juries might be missing")
        
        systematic_ratio = np.mean([d['systematic_models']/d['total_models'] for d in dataset_summaries if d['total_models'] > 0])
        if systematic_ratio > 0.5:
            print("   • High systematic disagreement suggests missing evaluation criteria")
            print("   • Consider adding models that capture different aspects of quality")

def main():
    """Main function to run the enhanced analysis."""
    parser = argparse.ArgumentParser(description='Enhanced leaderboard consistency analysis')
    parser.add_argument('directory', help='Path to directory containing CSV files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        return
    
    enhanced_leaderboard_analysis(args.directory)

if __name__ == "__main__":
    # If no command line arguments, use current directory
    directory = "../medhelm/judge_ablations"
    enhanced_leaderboard_analysis(directory)