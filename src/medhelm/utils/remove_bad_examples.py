import json
import numpy as np
from typing import Dict, List, Any
import statistics
import os


def analyze_scores(data: List[Dict[str, Any]], gold_standard: Dict[str, int], target_dataset: str) -> None:
    # Extract scores before filtering
    scores_before = []
    for entry in data:
        for stat in entry.get("stats", []):
            if target_dataset in stat["name"]["name"]:
                scores_before.append(stat.get("sum"))
    if len(scores_before)==0:
        raise ValueError(f"No scores found for {target_dataset}")
    # Filter out entries based on gold standard
    filtered_data_bad = [
        entry for entry in data 
        if gold_standard[entry["instance_id"]] == 0
    ]
    filtered_data_good = [
        entry for entry in data 
        if gold_standard[entry["instance_id"]] == 1
    ]
    
    # Extract scores after filtering
    scores_after_bad = []
    scores_after_good = []
    for entry in filtered_data_bad:
        for stat in entry.get("stats", []):
            if target_dataset in stat.get("name", {}).get("name", ""):
                scores_after_bad.append(stat.get("sum"))
    for entry in filtered_data_good:
        for stat in entry.get("stats", []):
            if target_dataset in stat.get("name", {}).get("name", ""):
                scores_after_good.append(stat.get("sum"))

    return scores_before, scores_after_bad, scores_after_good

def calc_stats(scores):
    return (
        np.mean(scores),
        np.median(scores),
        np.std(scores) if len(scores) > 1 else 0
    )

def main():
    # Load the gold standard evaluations
    target_dataset = "mimic_rrs"
    PRIVACY="PRIVATE"
    directory = f"../medhelm/data/benchmark_output/runs/MED-HELM-{PRIVACY}"
    all_outputs=[[], [], []]
    with open(f"../medhelm/data/filtering/gold_standard_evaluations_{target_dataset}.json", "r") as f:
        gold_standard = json.load(f)
    number_of_models=0
    for folder in os.listdir(directory):
        if target_dataset in folder:
            stats_file = os.path.join(directory, folder, "per_instance_stats.json")
            with open(stats_file, "r") as f:
                data = json.load(f)
            outputs=analyze_scores(data, gold_standard, target_dataset)
            all_outputs[0].extend(outputs[0])
            all_outputs[1].extend(outputs[1])
            all_outputs[2].extend(outputs[2])
            number_of_models+=1
        # Calculate statistics for all outputs
    before_mean, before_median, before_std = calc_stats(all_outputs[0])
    after_mean_bad, after_median_bad, after_std_bad = calc_stats(all_outputs[1]) 
    after_mean_good, after_median_good, after_std_good = calc_stats(all_outputs[2])

    print("\nScore Statistics:")
    print("Before filtering:")
    print(f"  Mean: {before_mean:.3f}")
    print(f"  Median: {before_median:.3f}") 
    print(f"  Std Dev: {before_std:.3f}")
    
    print("\nAfter filtering (Bad examples):")
    print(f"  Mean: {after_mean_bad:.3f}")
    print(f"  Median: {after_median_bad:.3f}")
    print(f"  Std Dev: {after_std_bad:.3f}")
    
    print("\nAfter filtering (Good examples):")
    print(f"  Mean: {after_mean_good:.3f}") 
    print(f"  Median: {after_median_good:.3f}")
    print(f"  Std Dev: {after_std_good:.3f}")

    print(f"Before filtering: {len(all_outputs[0])/number_of_models}")
    print(f"After filtering (Bad examples): {len(all_outputs[1])/number_of_models}")
    print(f"After filtering (Good examples): {len(all_outputs[2])/number_of_models}")
    

if __name__ == "__main__":
    main()
