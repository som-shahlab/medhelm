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
    BENCHMARK_NAME_MAPPING, 
    MODEL_NAME_MAPPING
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


def get_metric_stats(
    per_instance_stats_list: List[PerInstanceStats], main_metric_name: str
) -> List[Stat]:
    main_metric_stats: List[Stat] = []
    for per_instance_stats in per_instance_stats_list:
        for stat in per_instance_stats.stats:
            if stat.name.name == main_metric_name and stat.name.split == TEST:
                main_metric_stats.append(stat)
    main_metric_stats.sort(key=lambda x: x.sum)
    return main_metric_stats


def compute_statistics(
    main_metric_stats: List[float]
) -> Dict[str, float]:
    numpy_array = np.array(main_metric_stats)
    mean= np.mean(numpy_array)
    median = np.median(numpy_array)
    stddev = np.std(numpy_array)
    variance = np.var(numpy_array)
    minimum = main_metric_stats[0]
    maximum = main_metric_stats[-1]
    return {
        "mean": mean,
        "median": median,
        "stddev": stddev,
        "variance": variance,
        "minimum": minimum,
        "maximum": maximum
    }
        

def generate_plots(
    runs_path: str,
    output_dir: str
) -> None:
    metric_data_main: Dict[str, Dict[str, List[float]]] = {}
    metric_data_secondary: Dict[str, Dict[str, List[float]]] = {}
    suite_names = [
        p for p in os.listdir(runs_path)
        if os.path.isdir(os.path.join(runs_path, p))
    ]

    run_dir_names = []

    for suite in suite_names:
        run_suite_path = os.path.join(runs_path, suite)
        for p in os.listdir(run_suite_path):
            full_path = os.path.join(run_suite_path, p)
            if p not in {"eval_cache", "groups"} and os.path.isdir(full_path):
                run_dir_names.append(os.path.join(suite, p))  # Keeping track of suite/run

    run_dir_names.sort()
    
    for run_dir_name in tqdm(run_dir_names, disable=None):
        per_instance_stats_path: str = os.path.join(runs_path, run_dir_name, PER_INSTANCE_STATS_FILE_NAME)
        scenario_path: str = os.path.join(runs_path, run_dir_name, SCENARIO_FILE_NAME)
        run_spec_path: str = os.path.join(runs_path, run_dir_name, RUN_SPEC_FILE_NAME)
        if not os.path.exists(per_instance_stats_path):
            hlog(f"WARNING: {run_dir_name} doesn't have {PER_INSTANCE_STATS_FILE_NAME}, skipping")
            continue
        if not os.path.exists(scenario_path):
            hlog(f"WARNING: {run_dir_name} doesn't have {SCENARIO_FILE_NAME}, skipping")
            continue
        if not os.path.exists(run_spec_path):
            hlog(f"WARNING: {run_dir_name} doesn't have {RUN_SPEC_FILE_NAME}, skipping")
            continue
        scenario = read_scenario(scenario_path)
        scenario_name = scenario["name"]
        if scenario_name not in OPEN_ENDED_BENCHMARKS:
            continue
        run_spec = read_run_spec(run_spec_path)
        model = run_spec.adapter_spec.model.split("/")[-1].replace("-", "_")
        if model not in metric_data_main:
            metric_data_main[model] = {}
        if model not in metric_data_secondary:
            metric_data_secondary[model] = {}
        
        main_metric_name = BENCHMARK_METRICS.get(scenario_name)
        secondary_metric_name = "BERTScore-F"

        if not main_metric_name:
            hlog(f"WARNING: No benchmark metric defined for scenario {scenario_name}, skipping")
            continue

        per_instance_stats_list = read_per_instance_stats(per_instance_stats_path)
        main_metric_stats = get_metric_stats(per_instance_stats_list, main_metric_name)
        secondary_metric_stats = get_metric_stats(per_instance_stats_list, secondary_metric_name)

        if scenario_name not in metric_data_main[model]:
            metric_data_main[model][scenario_name] = []
        if scenario_name not in metric_data_secondary[model]:
            metric_data_secondary[model][scenario_name] = []
        min_val, max_val = MAIN_METRIC_RANGE
        metric_data_main[model][scenario_name].extend([(stat.sum - min_val) / (max_val - min_val) for stat in main_metric_stats])
        metric_data_secondary[model][scenario_name].extend([stat.sum for stat in secondary_metric_stats])
        print(f"Metric statistics for {model},{scenario_name}:")
        print(compute_statistics(metric_data_main[model][scenario_name]))
        print(f"Secondary metric statistics for {model},{scenario_name}:")
        print(compute_statistics(metric_data_secondary[model][scenario_name]))

    for model in metric_data_main.keys():
        generate_box_plots(metric_data_main, metric_data_secondary, output_dir, model)


def generate_box_plots(
    metric_data_main: Dict[str, Dict[str, List[float]]],
    metric_data_secondary: Dict[str, Dict[str, List[float]]],
    output_dir: str,
    model: str
):
    """Generate and save box plots for both main and secondary metrics in the same figure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenarios = list(metric_data_main[model].keys())
    data_main = [metric_data_main[model][scenario] for scenario in scenarios]
    data_secondary = [metric_data_secondary[model][scenario] for scenario in scenarios]
    scenario_labels = [BENCHMARK_NAME_MAPPING[scenario] for scenario in scenarios]

    # Combine data for both metrics
    combined_data = []
    positions = []
    for i, (main_data, secondary_data) in enumerate(zip(data_main, data_secondary)):
        combined_data.append(main_data)
        combined_data.append(secondary_data)
        positions.extend([2 * i + 1, 2 * i + 1.7])  # Reduce the distance between main and secondary metrics

    # Create the plot
    plt.figure(figsize=(15, 8))
    # box = plt.boxplot(combined_data, patch_artist=True, positions=positions, showfliers=False)
    box = plt.boxplot(combined_data, patch_artist=True, positions=positions, showfliers=True)

    # Set alternating colors for the two metrics
    colors = ["#1f77b4", "#ff7f0e"]  # Blue for main metric, orange for secondary metric
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(colors[i % 2])
    for median in box["medians"]:
        median.set_color("black")

    # Adjust x-ticks to be centered between the two box plots for each benchmark
    x_ticks = [1.35 + 2 * i for i in range(len(scenarios))]  # Adjust x-tick positions to match the new spacing
    plt.xticks(x_ticks, scenario_labels, rotation=45, ha='right')

    # Add a legend for the colors
    plt.legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", lw=4, label="Jury Score"),
            plt.Line2D([0], [0], color="#ff7f0e", lw=4, label="BERTScore-F"),
        ],
        loc="lower right",
    )

    plt.ylabel("Normalized Score")
    plt.xlabel("Benchmark")
    plt.title(f"Performance on open-ended benchmarks ({MODEL_NAME_MAPPING[model]})")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"box_plot_{model}.png")
    plt.savefig(plot_path)
    hlog(f"Box plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--benchmark-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Where to save the plots", default="plots"
    )
    args = parser.parse_args()
    benchmark_path = args.benchmark_path
    output_dir = args.output_dir
    runs_path = os.path.join(benchmark_path, "runs")
    generate_plots(runs_path, output_dir)


if __name__ == "__main__":
    main()
