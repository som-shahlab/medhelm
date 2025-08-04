import os
import json
import statistics
import csv
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_stats(values: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard deviation for a list of values."""
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def process_predictions_file(filepath: str, exclude_judges: List[str]) -> Tuple[Dict[str, Dict[str, List[float]]], str, str]:
    """
    Extract judge scores for all judge models and metrics across all instances in the file.
    Returns: (Dict[judge][metric] -> List of scores, dataset_name, model_name)
    """
    judge_metric_scores = defaultdict(lambda: defaultdict(list))
    
    # Extract dataset and model from filepath
    parts = filepath.split(os.sep)
    dataset = None
    model = None
    
    for i, part in enumerate(parts):
        if "model" in part:
            dataset = part.split(":")[0]  # Get text before colon
            model= part.split("model_deployment=")[-1]
            break

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        for instance in data:
            annotations = instance.get("annotations", {})
            if not annotations:
                continue
            task_key = next(iter(annotations))
            task_annotations = annotations.get(task_key, {})

            for judge, metrics in task_annotations.items():
                if judge not in exclude_judges:
                    for metric in ["accuracy", "completeness", "clarity", "structure"]:
                        try:
                            score = float(metrics[metric]["score"])
                            judge_metric_scores[judge][metric].append(score)
                        except (KeyError, TypeError, ValueError):
                            continue
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error processing file {filepath}: {e}")

    if len(judge_metric_scores.keys()) != 3-len(exclude_judges) and len(judge_metric_scores.keys()):
        print(f"{filepath} has {len(judge_metric_scores.keys())} judges")
        assert len(judge_metric_scores.keys()) == 3-len(exclude_judges), "All 3 judges should be present"
    return judge_metric_scores, dataset, model


def find_and_process_files(root_path: str, exclude_judges: List[str], output_file: str) -> Dict:
    """
    Process all 'display_predictions.json' files and compute per-file and aggregate stats.
    Also creates a leaderboard CSV with models vs datasets.
    """
    all_scores = defaultdict(lambda: defaultdict(list))  # judge -> metric -> [scores]
    all_scores_flat = []
    model_dataset_scores = defaultdict(lambda: defaultdict(list))  # model -> dataset -> [scores]

    file_stats = {}

    for root, _, files in os.walk(root_path):
        if "display_predictions.json" in files:
            filepath = os.path.join(root, "display_predictions.json")
            judge_scores, dataset, model = process_predictions_file(filepath, exclude_judges)
            
            if dataset and model and len(judge_scores.keys()) == 3-len(exclude_judges):
                # Calculate average score across all judges and metrics for this file
                all_file_scores = []
                for judge_model in judge_scores.keys():
                    for judge_metrics in judge_scores[judge_model].values():
                        assert len(judge_scores[judge_model].keys()) == 3, "All 3 metrics should be present"
                        all_file_scores.extend(judge_metrics)
                    if all_file_scores:
                        avg_score = statistics.mean(all_file_scores)
                        model_dataset_scores[model][dataset].append(avg_score)
                        print(f"Model: {model}, Dataset: {dataset}, Score: {avg_score}")

            # Flatten all scores in file
            file_scores_flat = []
            per_judge_metric_stats = {}

            for judge, metric_scores in judge_scores.items():
                for metric, scores in metric_scores.items():
                    all_scores[judge][metric].extend(scores)
                    file_scores_flat.extend(scores)
                    per_judge_metric_stats[(judge, metric)] = calculate_stats(scores)

            if file_scores_flat:
                file_stats[filepath] = {
                    "mean": calculate_stats(file_scores_flat)[0],
                    "std": calculate_stats(file_scores_flat)[1],
                    "per_judge_metric_stats": per_judge_metric_stats
                }

            all_scores_flat.extend(file_scores_flat)

    # Create leaderboard CSV
    datasets = sorted(set(dataset for scores in model_dataset_scores.values() for dataset in scores.keys()))
    models = sorted(model_dataset_scores.keys())
    # breakpoint()
    
    # with open(output_file, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Model'] + datasets)
    #     for model in models:
    #         row = [model]
    #         for dataset in datasets:
    #             scores = model_dataset_scores[model].get(dataset, [])
    #             avg = statistics.mean(scores) if scores else ''
    #             row.append(f"{avg:.3f}" if avg != '' else '')
    #         writer.writerow(row)

    # Aggregate across all files per judge-metric
    aggregate_stats = {}
    for judge, metrics in all_scores.items():
        aggregate_stats[judge] = {}
        for metric, values in metrics.items():
            mean, std = calculate_stats(values)
            aggregate_stats[judge][metric] = {
                "mean": mean,
                "std": std,
                "n_samples": len(values)
            }

    # Global (all judges, all metrics)
    total_mean, total_std = calculate_stats(all_scores_flat)
    return {
        "per_file": file_stats,
        "aggregate": aggregate_stats,
        "total_scores": {
            "mean": total_mean,
            "std": total_std,
            "n_samples": len(all_scores_flat)
        }
    }


def main():
    root_path = "../medhelm/data/benchmark_output/runs"
    exclude_judges = [] #llama, claude
    results = find_and_process_files(root_path, exclude_judges=exclude_judges, output_file="/share/pi/nigam/users/aunell/medhelm/judge_ablations/leaderboard_judge_only_claude.csv")

    print("\nPer-file Statistics:")
    for filepath, stats in results['per_file'].items():
        print(f"\n{filepath}")
        print(f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        for (judge, metric), (mean, std) in stats["per_judge_metric_stats"].items():
            print(f"  {judge}:{metric} -> Mean = {mean:.3f}, Std = {std:.3f}")

    print("\nAggregate Statistics (Per Judge and Metric):")
    for judge, metric_stats in results["aggregate"].items():
        print(f"\nJudge: {judge}")
        for metric, stats in metric_stats.items():
            print(f"  {metric}: Mean = {stats['mean']:.3f}, Std = {stats['std']:.3f}, N = {stats['n_samples']}")

    print("\nOverall Statistics (All Scores Combined):")
    print(f"Mean: {results['total_scores']['mean']:.3f}")
    print(f"Std: {results['total_scores']['std']:.3f}")
    print(f"N: {results['total_scores']['n_samples']}")


if __name__ == "__main__":
    main()


#todo
#find the model x dataset score to replicate leaderboard by 1) averaging across judges vs 2) looking at single judge

#