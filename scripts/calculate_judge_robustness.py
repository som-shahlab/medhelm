import os
import json
import statistics
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_stats(values: List[float]) -> Tuple[float, float]:
    """Calculate mean and standard deviation for a list of values."""
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def process_predictions_file(filepath: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract judge scores for all judge models and metrics across all instances in the file.
    Returns: Dict[judge][metric] -> List of scores
    """
    judge_metric_scores = defaultdict(lambda: defaultdict(list))

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
                for metric in ["accuracy", "structure", "clarity"]:
                    try:
                        score = float(metrics[metric]["score"])
                        judge_metric_scores[judge][metric].append(score)
                    except (KeyError, TypeError, ValueError):
                        continue

    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error processing file {filepath}: {e}")

    return judge_metric_scores


def find_and_process_files(root_path: str) -> Dict:
    """
    Process all 'display_predictions.json' files and compute per-file and aggregate stats.
    """
    all_scores = defaultdict(lambda: defaultdict(list))  # judge -> metric -> [scores]
    all_scores_flat = []

    file_stats = {}

    for root, _, files in os.walk(root_path):
        if "display_predictions.json" in files:
            filepath = os.path.join(root, "display_predictions.json")
            judge_scores = process_predictions_file(filepath)

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
    results = find_and_process_files(root_path)

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
