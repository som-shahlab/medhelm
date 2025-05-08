import json
import numpy as np
from typing import Dict, List, Any, Tuple, Union
import statistics
import os
import pandas as pd
import argparse

# Number of models that should be processed
NUMBER_OF_MODELS = 9

def return_flagged_and_filtered_scores(data: List[Dict[str, Any]], gold_standard: Dict[str, int], target_dataset: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Filter data based on gold standard scores and extract performance metrics.
    
    Args:
        data: List of dictionaries containing model outputs and statistics
        gold_standard: Dictionary mapping instance IDs to gold standard scores (0=bad, 1=good)
        target_dataset: Name of the dataset to filter scores for
        
    Returns:
        Tuple containing:
        - List of scores before filtering
        - List of scores for examples marked as bad in gold standard
        - List of scores for examples marked as good in gold standard
        
    Raises:
        ValueError: If no scores are found for the target dataset
    """
    # Extract scores before filtering
    scores_before = {}
    for entry in data:
        for stat in entry.get("stats", []):
            if target_dataset in stat["name"]["name"]:
                scores_before[entry["instance_id"]] = stat.get("sum")
    if len(scores_before)==0:
        raise ValueError(f"No scores found for {target_dataset}")
        
    # Filter out entries based on gold standard
    filtered_data_bad = [
        entry for entry in data 
        if gold_standard[entry["instance_id"]] == 0 or gold_standard[entry["instance_id"]] == None
    ]
    filtered_data_good = [
        entry for entry in data 
        if gold_standard[entry["instance_id"]] == 1
    ]
    
    # Extract scores after filtering
    scores_after_bad = {}
    scores_after_good = {}
    for entry in filtered_data_bad:
        for stat in entry.get("stats", []):
            if target_dataset in stat.get("name", {}).get("name", ""):
                scores_after_bad[entry["instance_id"]] = stat.get("sum")
    for entry in filtered_data_good:
        for stat in entry.get("stats", []):
            if target_dataset in stat.get("name", {}).get("name", ""):
                scores_after_good[entry["instance_id"]] = stat.get("sum")

    return scores_before, scores_after_bad, scores_after_good

def calc_stats(scores: List[float]) -> Tuple[float, float, float]:
    """
    Calculate basic statistics for a list of scores.
    
    Args:
        scores: List of numeric scores
        
    Returns:
        Tuple containing mean, median and standard deviation
        If only one score, std dev will be 0
    """
    return (
        np.mean(scores),
        np.median(scores),
        np.std(scores) if len(scores) > 1 else 0
    )

def return_double_flagged_ids(outliers: str, gold_standard: Dict[str, int], target_dataset: str) -> Tuple[List[str], List[str]]:
    """
    Check which outliers were correctly and incorrectly flagged according to gold standard.
    
    Args:
        outliers: Path to CSV file containing outlier information
        gold_standard: Dictionary mapping instance IDs to gold standard scores
        target_dataset: Name of dataset to filter outliers for
        
    Returns:
        Tuple containing:
        - List of instance IDs that were flagged as "abnormal" from LLM judge and outlier analysis
        - List of instance IDs that were flagged as "normal" from LLM judge but were flagged as "abnormal" by outlier analysis
    """
    # Filter to only rows matching the folder
    outliers_df = pd.read_csv(outliers)
    outliers_df = outliers_df[outliers_df['scenario'] == target_dataset]
    if len(outliers_df) == 0:
        return [], []
    outliers_flagged = []
    outliers_not_flagged = []
    
    for _, row in outliers_df.iterrows():
        instance_id = row['id']
        if instance_id in gold_standard:
            if gold_standard[instance_id] == 0 or gold_standard[instance_id] == None:
                outliers_flagged.append(instance_id)
            elif gold_standard[instance_id] == 1:
                outliers_not_flagged.append(instance_id)
                
    return outliers_flagged, outliers_not_flagged

def remove_ids(data: List[Dict[str, Any]], ids: List[str]) -> List[Dict[str, Any]]:
    """
    Remove entries with specified IDs from data.
    
    Args:
        data: List of dictionaries containing model outputs
        ids: List of instance IDs to remove
        
    Returns:
        Filtered list with specified IDs removed
    """
    return [entry for entry in data if entry['instance_id'] not in ids]

def get_score(data: List[Dict[str, Any]], target_dataset: str) -> List[float]:
    """
    Extract scores for a specific dataset from model outputs.
    
    Args:
        data: List of dictionaries containing model outputs
        target_dataset: Name of dataset to extract scores for
        
    Returns:
        List of scores for the target dataset
    """
    scores_before = []
    for entry in data:
        for stat in entry.get("stats", []):
            if target_dataset in stat["name"]["name"]:
                scores_before.append(stat.get("sum"))
    return scores_before

def return_examples(scenario_state_file: str, instance_ids: str, scores: Dict[str, float], target_dataset: str, output_file: str) -> List[Dict[str, Any]]:
    """
    Extract examples from scenario state.
    
    Args:
        scenario_state: Dictionary containing scenario state
        
    Returns:
        List of examples
    """
    examples={}
    with open(scenario_state_file, "r") as f:
        scenario_state = json.load(f)
    instructions = scenario_state["adapter_spec"]["instructions"]
    for entry in scenario_state["request_states"]:
        if entry["instance"]["id"] in instance_ids:
            text = entry["instance"]["input"]["text"]
            reference = entry["instance"]["references"][0]["output"]["text"]
            examples[entry["instance"]["id"]] = {
                "instructions": instructions,
                "text": text,
                "reference": reference,
                "score": scores[entry["instance"]["id"]]
            }
            
    # Save examples as jsonl
    with open(output_file, "w") as f:
        for id, example in examples.items():
            example["id"] = id
            json.dump(example, f)
            f.write("\n")
    return None
  

def summarize_results(
    all_scores: Dict[str, List[float]],
    scores_double_flagged: List[float],
    number_of_models: int,
    TARGET_DATASET: str
) -> Dict[str, Dict[str, Any]]:
    def format_stats(scores: List[float]) -> Dict[str, float]:
        mean, median, std = calc_stats(scores)
        return {
            "mean": round(mean, 3),
            "median": round(median, 3),
            "std_dev": round(std, 3),
            "n_examples": len(scores) / number_of_models
        }

    results = {
        "before_filtering": format_stats(all_scores["all"]),
        "after_filtering_bad": format_stats(all_scores["flagged"]),
        "after_filtering_good": format_stats(all_scores["not_flagged"]),
        "after_filtering_double_flagged": format_stats(scores_double_flagged)
    }
    with open(f"../medhelm/data/filtering/stats/stats_{TARGET_DATASET}.json", "w") as f:
        json.dump(results, f)

    return results

def main() -> None:
    """
    Main function to analyze model performance with and without filtering.
    Processes multiple model outputs, filters based on gold standard evaluations,
    and prints statistics about model performance under different filtering conditions.
    """
    parser = argparse.ArgumentParser(description='Filter and analyze model performance')
    parser.add_argument('--target_dataset', type=str, required=True, help='Target dataset to analyze')
    args = parser.parse_args()
    TARGET_DATASET = args.target_dataset
    # Load the gold standard evaluations
    if TARGET_DATASET == "mtsamples_procedures" or TARGET_DATASET == "mtsamples_replicate":
        PRIVACY="PUBLIC"
    else:
        PRIVACY="PRIVATE"
    DIRECTORY_REASONING = f"../medhelm/data/benchmark_output/runs/MED-HELM-{PRIVACY}-REASONING"
    DIRECTORY_STANDARD = f"../medhelm/data/benchmark_output/runs/MED-HELM-{PRIVACY}"
    OUTLIERS= "../medhelm/results/outliers.csv"

    all_scores={"all": [], "flagged": [], "not_flagged": []}
    outlier_ids_flagged = []
    outlier_ids_not_flagged = []
    data_filtered_double_flagged = []

    with open(f"../medhelm/data/filtering/gold_standard_evaluations_{TARGET_DATASET}.json", "r") as f:
        gold_standard_scores = json.load(f)
    number_of_models=0

    for directory in [DIRECTORY_REASONING, DIRECTORY_STANDARD]:
        for folder in os.listdir(directory):
            if TARGET_DATASET in folder:
                stats_file = os.path.join(directory, folder, "per_instance_stats.json")
                examples_file = os.path.join(directory, folder, "scenario_state.json")
                with open(stats_file, "r") as f:
                    raw_data = json.load(f)

                original_scores, flagged_scores, not_flagged_scores=return_flagged_and_filtered_scores(raw_data, gold_standard_scores, TARGET_DATASET)
                outliers_flagged_ids, outliers_not_flagged_ids = return_double_flagged_ids(OUTLIERS, gold_standard_scores, TARGET_DATASET)
                all_scores["all"].extend(original_scores.values())
                all_scores["flagged"].extend(flagged_scores.values())
                all_scores["not_flagged"].extend(not_flagged_scores.values())

                outlier_ids_flagged.extend(outliers_flagged_ids)    
                outlier_ids_not_flagged.extend(outliers_not_flagged_ids)
                data_filtered_double_flagged.extend(remove_ids(raw_data, outlier_ids_flagged))

                number_of_models+=1

    assert (number_of_models==NUMBER_OF_MODELS)

    all_flagged_examples = [key for key in gold_standard_scores.keys() if gold_standard_scores[key]==0]
    double_flagged_examples = outliers_flagged_ids
    return_examples(examples_file, all_flagged_examples, flagged_scores, TARGET_DATASET, output_file=f"../medhelm/data/filtering/examples/all_flagged_examples_{TARGET_DATASET}.jsonl")
    return_examples(examples_file, double_flagged_examples, flagged_scores, TARGET_DATASET, output_file=f"../medhelm/data/filtering/examples/double_flagged_examples_{TARGET_DATASET}.jsonl")

    scores_double_flagged = get_score(data_filtered_double_flagged, TARGET_DATASET)

    print(summarize_results(all_scores, scores_double_flagged, number_of_models, TARGET_DATASET))

if __name__ == "__main__":
    main()
