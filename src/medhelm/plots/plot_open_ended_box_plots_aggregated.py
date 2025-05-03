import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Any
from tqdm import tqdm

from helm.benchmark.run_spec import RunSpec
from helm.common.codec import from_json
from helm.common.hierarchical_logger import hlog
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.metrics.statistic import Stat

from medhelm.utils.constants import (
    BENCHMARK_METRICS, 
    OPEN_ENDED_BENCHMARKS, 
    BENCHMARK_NAME_MAPPING
)

PER_INSTANCE_STATS_FILE_NAME = "per_instance_stats.json"
SCENARIO_FILE_NAME = "scenario.json"
RUN_SPEC_FILE_NAME = "run_spec.json"
TEST = "test"
MAIN_METRIC_RANGE = [1, 5]


def read_per_instance_stats(per_instance_stats_path: str) -> List[PerInstanceStats]:
    if not os.path.exists(per_instance_stats_path):
        raise ValueError(f"Could not load [PerInstanceStats] from {per_instance_stats_path}")
    with open(per_instance_stats_path) as f:
        return from_json(f.read(), List[PerInstanceStats])
   

def read_run_spec(run_spec_path: str) -> RunSpec:
    if not os.path.exists(run_spec_path):
        raise ValueError(f"Could not load RunSpec from {run_spec_path}")
    with open(run_spec_path) as f:
        return from_json(f.read(), RunSpec)


def read_scenario(scenario_path: str) -> Dict[str, Any]:
    with open(scenario_path) as scenario_file:
        scenario = json.load(scenario_file)
    return scenario


def get_metric_stats(per_instance_stats_list: List[PerInstanceStats], metric_name: str) -> Dict[str, float]:
    instance_to_value = {}
    for per_instance_stats in per_instance_stats_list:
        for stat in per_instance_stats.stats:
            if stat.name.name == metric_name and stat.name.split == TEST:
                instance_to_value[per_instance_stats.instance_id] = stat.sum
    return instance_to_value


def generate_aggregated_plot(runs_path: str, output_dir: str) -> None:
    benchmark_instance_scores_main: Dict[str, Dict[str, List[float]]] = {}
    benchmark_instance_scores_secondary: Dict[str, Dict[str, List[float]]] = {}

    suite_names = [
        p for p in os.listdir(runs_path)
        if os.path.isdir(os.path.join(runs_path, p))
    ]
    run_dir_names = []
    processed = {}
    for suite in suite_names:
        run_suite_path = os.path.join(runs_path, suite)
        for p in os.listdir(run_suite_path):
            full_path = os.path.join(run_suite_path, p)
            if p not in {"eval_cache", "groups"} and os.path.isdir(full_path):
                run_dir_names.append(os.path.join(suite, p))
    run_dir_names.sort()
    for run_dir_name in tqdm(run_dir_names, disable=None):
        per_instance_stats_path = os.path.join(runs_path, run_dir_name, PER_INSTANCE_STATS_FILE_NAME)
        scenario_path = os.path.join(runs_path, run_dir_name, SCENARIO_FILE_NAME)
        run_spec_path = os.path.join(runs_path, run_dir_name, RUN_SPEC_FILE_NAME)
        run_spec = read_run_spec(run_spec_path)
        model = run_spec.adapter_spec.model.split("/")[-1].replace("-", "_")
        if not (os.path.exists(per_instance_stats_path) and os.path.exists(scenario_path) and os.path.exists(run_spec_path)):
            hlog(f"WARNING: Missing files in {run_dir_name}, skipping")
            continue
        
        scenario = read_scenario(scenario_path)
        scenario_name = scenario["name"]
        if scenario_name not in OPEN_ENDED_BENCHMARKS:
            continue
        main_metric_name = BENCHMARK_METRICS.get(scenario_name)
        secondary_metric_name = "BERTScore-F"
        if not main_metric_name:
            hlog(f"WARNING: No benchmark metric defined for scenario {scenario_name}, skipping")
            continue

        per_instance_stats_list = read_per_instance_stats(per_instance_stats_path)
        main_scores = get_metric_stats(per_instance_stats_list, main_metric_name)
        secondary_scores = get_metric_stats(per_instance_stats_list, secondary_metric_name)

        if f"{scenario_name},{model}" not in processed:
            processed[f"{scenario_name},{model}"] = 0
        if scenario_name not in benchmark_instance_scores_main:
            benchmark_instance_scores_main[scenario_name] = {}
        if scenario_name not in benchmark_instance_scores_secondary:
            benchmark_instance_scores_secondary[scenario_name] = {}
        for instance_id, score in main_scores.items():
            benchmark_instance_scores_main[scenario_name].setdefault(f"{instance_id}-{processed.get(f'{scenario_name},{model}')}" if processed.get(f"{scenario_name},{model}") > 0 else instance_id, []).append(score)
        for instance_id, score in secondary_scores.items():
            benchmark_instance_scores_secondary[scenario_name].setdefault(f"{instance_id}-{processed.get(f'{scenario_name},{model}')}" if processed.get(f"{scenario_name},{model}") > 0 else instance_id, []).append(score)
        processed[f"{scenario_name},{model}"] += 1

    # Average and normalize scores
    data_main = []
    data_secondary = []
    labels = []

    for scenario_name in sorted(benchmark_instance_scores_main.keys()):
        main_scores = []
        
        for id, scores in benchmark_instance_scores_main[scenario_name].items():
            avg_score = sum(scores) / len(scores)
            min_val, max_val = MAIN_METRIC_RANGE
            norm_score = (avg_score - min_val) / (max_val - min_val)
            main_scores.append(norm_score)
        data_main.append(main_scores)

        secondary_scores = []
        for scores in benchmark_instance_scores_secondary[scenario_name].values():
            avg_score = sum(scores) / len(scores)
            secondary_scores.append(avg_score)
        data_secondary.append(secondary_scores)

        label = BENCHMARK_NAME_MAPPING.get(scenario_name, scenario_name)
        labels.append(label)

    # Create plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Combine data for both metrics
    combined_data = []
    positions = []
    for i in range(len(labels)):
        combined_data.append(data_main[i])
        combined_data.append(data_secondary[i])
        positions.extend([2 * i + 1, 2 * i + 1.7])  # Reduced distance between main and secondary metrics

    # Create the plot
    box = plt.boxplot(combined_data, patch_artist=True, positions=positions, showfliers=True)

    # Set alternating colors for the two metrics
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for main metric, orange for secondary metric
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(colors[i % 2])
    for median in box["medians"]:
        median.set_color("black")

    # Adjust x-ticks to be centered between the two box plots for each benchmark
    x_ticks = [1.35 + 2 * i for i in range(len(labels))]
    plt.xticks(x_ticks, labels, rotation=45, ha='right')

    # Add a legend for the colors
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", lw=4, label="Main Metric"),
            plt.Line2D([0], [0], color="#ff7f0e", lw=4, label="Secondary Metric"),
        ],
        loc="lower right",
    )

    plt.ylabel("Normalized Score")
    plt.xlabel("Benchmark")
    plt.title("Average Performance on Open Ended Benchmarks Across all Models")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "box_plot_aggregated.png")
    plt.savefig(output_path)
    hlog(f"Saved figure to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", "-b", type=str, required=True, help="Path to the directory containing run outputs")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for the figure")
    args = parser.parse_args()

    generate_aggregated_plot(f"{args.benchmark_path}/runs", args.output_dir)
